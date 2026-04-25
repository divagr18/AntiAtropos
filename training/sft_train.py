#!/usr/bin/env python3
"""
sft_train.py — AntiAtropos SFT Training (QLoRA via Unsloth)

Phase 1: teach the model strict SRE action grammar, DAG physics intuition,
and JSON-only output format via supervised fine-tuning on heuristic +
expert-generated trajectories.

Model: Qwen/Qwen3.5-4B (4-bit QLoRA via Unsloth)

Usage:
  python training/sft_train.py

  # Override defaults:
  python training/sft_train.py --episodes-per-task 20 --max-steps 60 --epochs 3

  # Skip data generation if JSONL files already exist:
  python training/sft_train.py --skip-data-gen

  # Skip quality-gate evaluation:
  python training/sft_train.py --skip-eval
"""

import argparse
import json
import math
import os
import random
import re
import sys
import textwrap
import time
from collections import Counter
from enum import Enum
from pathlib import Path

import requests
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split

# ---- Torch-dependent imports (deferred for data-gen-only mode) ----
# Unsloth MUST be imported before torch/transformers/peft/trl.
# If torch/CUDA is misconfigured, data generation still works.
try:
    import unsloth  # noqa: F401 — patches torch before trl/peft
    from unsloth import FastLanguageModel  # noqa: F401
    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    _TORCH_OK = True
    _TORCH_ERR = None
except ImportError as _e:
    _TORCH_OK = False
    _TORCH_ERR = str(_e)
    torch = None  # type: ignore[assignment]
    FastLanguageModel = None  # type: ignore[assignment]
    Dataset = None  # type: ignore[assignment]
    SFTTrainer = None  # type: ignore[assignment]
    SFTConfig = None  # type: ignore[assignment]


# ════════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════════

VALID_ACTIONS = ["NO_OP", "SCALE_UP", "SCALE_DOWN", "REROUTE_TRAFFIC", "SHED_LOAD"]
VALID_NODES = ["node-0", "node-1", "node-2", "node-3", "node-4"]
MAX_QUEUE_NORM = 200.0
MAX_LATENCY_NORM = 1000.0
MAX_REQUEST_RATE_NORM = 100.0


TASK_BRIEFS = {
    "task-1": "Traffic ramps linearly every tick. Scale up proactively — new capacity takes 5 ticks to boot. Keep latency under SLA (200ms) while minimizing cost. Scale down when queues are safe.",
    "task-2": "One node (node-1 through node-4) will fail permanently. Wait until you SEE a FAILED node — do NOT pre-scale. Once a node shows status=FAILED: reroute traffic FROM the failed node to healthy peers, and scale up any starved children. Do NOT scale node-0 unless node-4 failed independently. SCALE_DOWN cancels pending boots and reduces cost. If reward is falling, stop scaling.",
    "task-3": "A surge (~75 req/tick) will hit node-1 and node-2 via a side channel bypassing node-0. Do NOT scale node-0 — it is NOT affected. ONLY scale node-1 or node-2 when their queue_depth rises. Do NOT pre-scale. 3-4 SCALE_UPs on each is sufficient. SCALE_DOWN cancels pending boots and reduces cost — use it when queues are safe. If reward is falling, STOP scaling and SCALE_DOWN to recover.",
}

SYSTEM_PROMPT = textwrap.dedent("""
    You are an autonomous SRE controller managing a five-node microservice cluster.

    CRITICAL: /no_think mode. DO NOT use <think> or </think> tags. NO reasoning blocks. Output ONLY your action directly as plain text.

    CLUSTER TOPOLOGY (traffic flows parent → children):
      node-0 → node-1, node-2
      node-2 → node-3
      node-4 (independent ingress)
    FAILED nodes have outflow=0 — their children are starved.
    Backpressure: overloaded children reduce parent capacity.

    ACTIONS (new capacity takes 5 ticks to boot):
      SCALE_UP <node> <amount>   — add capacity (0.3-0.5 normal, 0.6-0.8 heavy surge), clears DEGRADED
      SCALE_DOWN <node> <amount>  — cancel pending boots first, then remove active capacity (0.2-0.4 safe, 0.5-0.7 aggressive)
      REROUTE_TRAFFIC <node> <fraction> — reduce THIS node capacity, redistribute to peers (0.3-0.5)
      SHED_LOAD <node> <fraction>  — drop incoming traffic (0.3-0.5), NEVER on node-0 (payment gateway)
      NO_OP                           — do nothing

    REWARD PRIORITIES (in order):
      1. Avoid SLA violations (latency > 200ms or error rate > 5%)
      2. Keep queues low (growing queues = destabilizing system)
      3. Don't over-provision (excess capacity costs money)

    REWARD SIGNAL: Each step returns a reward [0,1].
      > 0.5 = good. 0.15–0.5 = acceptable. < 0.15 = you are making things worse.
      If reward is falling, STOP the current strategy — try a different action or NO_OP.
      Repeating the same action when reward < 0.1 is always wrong.

    Scale when your observations demand it, not preemptively.
    Boot delay is 5 ticks — factor this into your timing.
    Scale back down when safe to save cost.

    Return exactly one JSON object:
    {
      "action_type": "SCALE_UP" | "SCALE_DOWN" | "REROUTE_TRAFFIC" | "SHED_LOAD" | "NO_OP",
      "target_node_id": "node-0" | "node-1" | "node-2" | "node-3" | "node-4",
      "parameter": 0.0
    }
""").strip()


# ════════════════════════════════════════════════════════════════════════════════
# Action types (re-defined to avoid dependency on models.py)
# ════════════════════════════════════════════════════════════════════════════════

class ActionType(str, Enum):
    NO_OP = "NO_OP"
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    REROUTE_TRAFFIC = "REROUTE_TRAFFIC"
    SHED_LOAD = "SHED_LOAD"


# ════════════════════════════════════════════════════════════════════════════════
# HF Space HTTP client
# ════════════════════════════════════════════════════════════════════════════════

_session = requests.Session()


def env_reset(hf_space_url, task_id="task-1", seed=None):
    payload = {"task_id": task_id}
    if seed is not None:
        payload["seed"] = seed
    resp = _session.post(f"{hf_space_url}/reset", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(hf_space_url, action_type, target_node_id, parameter):
    payload = {
        "action": {
            "action_type": action_type,
            "target_node_id": target_node_id,
            "parameter": parameter,
        }
    }
    resp = _session.post(f"{hf_space_url}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ════════════════════════════════════════════════════════════════════════════════
# Observation formatting
# ════════════════════════════════════════════════════════════════════════════════

def format_observation(obs_dict, task_id, step, max_steps, reward=0.0, sla_violations=0):
    """Build user prompt aligned with inference.py: task brief + compact JSON observation."""
    brief = TASK_BRIEFS.get(task_id, "Maintain SLA, stability, and efficient cost.")

    # Synthesize cluster summary (matches inference.py build_user_prompt)
    cost_hour = obs_dict.get("current_cost_per_hour", 0.0)
    cost_dev = "low" if cost_hour < 1.2 else ("high" if cost_hour > 1.8 else "baseline")
    queue_backlog = obs_dict.get("total_queue_backlog", 0.0)
    queue_trend = "rising" if queue_backlog > 0.3 else ("stable" if queue_backlog < 0.1 else "moderate")
    sla_note = f" ({sla_violations} violations)" if sla_violations > 0 else ""
    r_tag = "GOOD" if reward > 0.5 else ("OK" if reward > 0.2 else ("BAD" if reward > 0.05 else "STOP-SCALING"))
    cluster_summary = f"Cost: {cost_dev} (${cost_hour:.2f}/hr) | Queues: {queue_trend}{sla_note} | Reward: {reward:.2f}={r_tag}"

    # Build compact observation dict (mirrors inference.py observation_for_model)
    nodes_data = []
    for n in obs_dict.get("nodes", []):
        nodes_data.append({
            "node_id": n.get("node_id"),
            "status": n.get("status", "HEALTHY"),
            "queue_depth": n.get("queue_depth", 0),
            "latency_ms": n.get("latency_ms", 0),
            "incoming_request_rate": n.get("incoming_request_rate", 0),
            "cpu_utilization": n.get("cpu_utilization", 0),
            "capacity": n.get("capacity", 0),
            "pending_capacity": n.get("pending_capacity", 0),
            "outflow_rate": n.get("outflow_rate", 0),
            "upstream_pressure": n.get("upstream_pressure", 0),
        })

    obs_compact = {
        "task_id": task_id,
        "step": step,
        "max_steps": max_steps,
        "failed_nodes": [n["node_id"] for n in obs_dict.get("nodes", []) if n.get("status") == "FAILED"],
        "degraded_nodes": [n["node_id"] for n in obs_dict.get("nodes", []) if n.get("status") == "DEGRADED"],
        "average_latency_ms": obs_dict.get("average_latency_ms", 0),
        "error_rate": obs_dict.get("error_rate", 0),
        "total_queue_backlog": obs_dict.get("total_queue_backlog", 0),
        "current_cost_per_hour": obs_dict.get("current_cost_per_hour", 0),
        "sla_violations": sla_violations,
        "nodes": nodes_data,
    }

    return textwrap.dedent(f"""
        Task: {task_id}
        Objective: {brief}
        Step: {step}
        Status: {cluster_summary}

        Current state:
        {json.dumps(obs_compact, separators=(',',':'))}

        Choose the next SRE action.
        """).strip()


# ════════════════════════════════════════════════════════════════════════════════
# Heuristic policy
# ════════════════════════════════════════════════════════════════════════════════

CRITICAL_NODES = {"node-0", "node-1", "node-2"}


def heuristic_action(obs_dict, task_id, step=0, max_steps=60, episode_reward=0.0):
    """Task-aware, reward-aware heuristic with balanced action distribution.

    Key design principles:
    - Task-specific: task-2 generates REROUTE_TRAFFIC on failures,
      task-3 targets node-1/node-2 for surges, NOT node-0.
    - Phase-aware: early steps tend to NO_OP, late steps tend to SCALE_DOWN.
    - Reward-aware: high reward → NO_OP/SCALE_DOWN, low reward → active fix.
    - Node-balanced: prefer downstream nodes for SCALE_UP, reserve node-0
      for last resort.
    """
    nodes = obs_dict.get("nodes", [])
    if not nodes:
        return ActionType.NO_OP, "node-0", 0.0
    node_map = {n["node_id"]: n for n in nodes}

    # Aggregate metrics
    total_queue = sum(n["queue_depth"] * 200 for n in nodes)
    avg_latency = sum(n["latency_ms"] for n in nodes) / len(nodes)
    error_rate = obs_dict.get("error_rate", 0)
    cost_per_hour = obs_dict.get("current_cost_per_hour", 0)
    failed_nodes = [n for n in nodes if n.get("status") == "FAILED"]
    degraded_nodes = [n for n in nodes if n.get("status") == "DEGRADED"]

    # Phase detection (0-1 progress through episode)
    progress = step / max_steps if max_steps > 0 else 0
    early = progress < 0.15
    late = progress > 0.65

    # ── TASK-2: Fault tolerance ──────────────────────────────────────
    if task_id == "task-2":
        if failed_nodes:
            fn = failed_nodes[0]
            # Alternate between REROUTE (drain failed) and SCALE_UP (feed children)
            starved_children = [
                n for n in nodes
                if n.get("status") == "DEGRADED"
                and n["node_id"] not in CRITICAL_NODES
            ]
            if starved_children and step % 3 != 0:
                target = max(starved_children, key=lambda n: n["queue_depth"])
                return ActionType.SCALE_UP, target["node_id"], 0.5
            return ActionType.REROUTE_TRAFFIC, fn["node_id"], 0.7

        # Reward is good and no failures → wind down excess capacity
        if episode_reward > 0.5 and avg_latency < 0.04:
            non_vips = [n for n in nodes
                        if not n.get("is_vip", False)
                        and n.get("status") != "FAILED"]
            overprov = [n for n in non_vips if n.get("capacity", 0) > 0.7]
            if overprov:
                target = max(overprov, key=lambda n: n.get("capacity", 0))
                return ActionType.SCALE_DOWN, target["node_id"], 0.3
            return ActionType.NO_OP, "node-0", 0.0

        # Moderate stress: scale up downstream
        if avg_latency > 0.04 or total_queue > 100:
            downstream = [n for n in nodes
                          if n["node_id"] != "node-0"
                          and n.get("status") != "FAILED"]
            if downstream:
                target = max(downstream, key=lambda n: (
                    n.get("status") == "DEGRADED", n["queue_depth"]))
                return ActionType.SCALE_UP, target["node_id"], 0.4

        return ActionType.NO_OP, "node-0", 0.0

    # ── TASK-3: Surge on node-1/2 (NOT node-0) ──────────────────────
    if task_id == "task-3":
        n1 = node_map.get("node-1", {})
        n2 = node_map.get("node-2", {})
        n3 = node_map.get("node-3", {})
        n4 = node_map.get("node-4", {})

        # Scale up node-1/2 when their queues rise
        if n1.get("queue_depth", 0) > 0.3:
            param = 0.6 if n1["queue_depth"] > 0.7 else 0.4
            return ActionType.SCALE_UP, "node-1", param
        if n2.get("queue_depth", 0) > 0.3:
            param = 0.6 if n2["queue_depth"] > 0.7 else 0.4
            return ActionType.SCALE_UP, "node-2", param

        # Shed load on overloaded non-critical downstream
        for nid, nd in [("node-3", n3), ("node-4", n4)]:
            if nd.get("queue_depth", 0) > 0.5 and nd.get("status") != "FAILED":
                return ActionType.SHED_LOAD, nid, 0.4

        # Scale down node-1/2 when overprovisioned and queues safe
        if avg_latency < 0.04 and total_queue < 80:
            for nid in ["node-1", "node-2"]:
                n = node_map.get(nid, {})
                if n.get("capacity", 0) > 0.8:
                    return ActionType.SCALE_DOWN, nid, 0.3

        # NO_OP when stable
        if episode_reward > 0.5 or (avg_latency < 0.04 and total_queue < 80):
            return ActionType.NO_OP, "node-0", 0.0

        # Mild stress on downstream
        if total_queue > 60:
            for nid in ["node-1", "node-2"]:
                n = node_map.get(nid, {})
                if n.get("queue_depth", 0) > 0.15 and n.get("status") != "FAILED":
                    return ActionType.SCALE_UP, nid, 0.3

        return ActionType.NO_OP, "node-0", 0.0

    # ── TASK-1: Traffic ramp (general) ────────────────────────────────

    # Early phase: traffic is still low → NO_OP
    if early and avg_latency < 0.03 and total_queue < 60:
        return ActionType.NO_OP, "node-0", 0.0

    # High reward → cluster is healthy, NO_OP or SCALE_DOWN
    if episode_reward > 0.55 and avg_latency < 0.04 and total_queue < 100:
        non_vips = [n for n in nodes
                    if not n.get("is_vip", False)
                    and n.get("status") != "FAILED"]
        overprov = [n for n in non_vips if n.get("capacity", 0) > 0.7]
        if overprov and total_queue < 60:
            target = max(overprov, key=lambda n: n.get("capacity", 0))
            return ActionType.SCALE_DOWN, target["node_id"], 0.3
        return ActionType.NO_OP, "node-0", 0.0

    # Late phase: traffic plateaued, consider SCALE_DOWN
    if late and avg_latency < 0.035 and total_queue < 80:
        non_vips = [n for n in nodes
                    if not n.get("is_vip", False)
                    and n.get("status") != "FAILED"]
        overprov = [n for n in non_vips if n.get("capacity", 0) > 0.7]
        if overprov:
            target = max(overprov, key=lambda n: n.get("capacity", 0))
            return ActionType.SCALE_DOWN, target["node_id"], 0.3
        return ActionType.NO_OP, "node-0", 0.0

    # SHED_LOAD on non-critical overloaded nodes
    non_critical_overloaded = [
        n for n in nodes
        if n["queue_depth"] > 0.5 and n["node_id"] not in CRITICAL_NODES
        and n.get("status") != "FAILED"
    ]
    if non_critical_overloaded and avg_latency > 0.05:
        target = non_critical_overloaded[0]
        return ActionType.SHED_LOAD, target["node_id"], 0.4

    # SCALE_UP — prefer downstream nodes, NOT node-0
    if avg_latency > 0.04 or total_queue > 100:
        downstream = [n for n in nodes
                      if n["node_id"] != "node-0"
                      and n.get("status") != "FAILED"]
        if downstream:
            target = max(downstream, key=lambda n: (
                n.get("status") == "DEGRADED",
                n["queue_depth"],
            ))
        else:
            # Only node-0 left — scale as last resort
            target = node_map.get("node-0", nodes[0])
        param = 0.6 if target["queue_depth"] > 0.75 else 0.4
        return ActionType.SCALE_UP, target["node_id"], param

    # DEFAULT: NO_OP when healthy
    return ActionType.NO_OP, "node-0", 0.0


def make_assistant_text(action_type, target_node_id, parameter):
    """Build the assistant response: JSON only, matching inference.py format."""
    return json.dumps({
        "action_type": action_type.value,
        "target_node_id": target_node_id,
        "parameter": round(float(parameter), 4),
    })


# ════════════════════════════════════════════════════════════════════════════════
# Expert augmentation examples
# ════════════════════════════════════════════════════════════════════════════════

def build_expert_examples():
    examples = []

    # Expert: REROUTE_TRAFFIC on FAILED node (task-2 scenario)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-2\nObjective: " + TASK_BRIEFS["task-2"] + "\nStep: 25\nStatus: Cost: baseline ($1.50/hr) | Queues: moderate (3 violations) | Reward: 0.32=BAD\n\nCurrent state:\n{\"task_id\":\"task-2\",\"step\":25,\"max_steps\":60,\"failed_nodes\":[\"node-1\"],\"degraded_nodes\":[],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.275,\"capacity\":0.6,\"incoming_request_rate\":0.47,\"latency_ms\":0.025,\"outflow_rate\":0.45},{\"node_id\":\"node-1\",\"status\":\"FAILED\",\"queue_depth\":0.0,\"capacity\":0.0,\"incoming_request_rate\":0.23,\"latency_ms\":1.0,\"outflow_rate\":0.0},{\"node_id\":\"node-2\",\"status\":\"HEALTHY\",\"queue_depth\":0.09,\"capacity\":0.6,\"incoming_request_rate\":0.23,\"latency_ms\":0.022,\"outflow_rate\":0.23},{\"node_id\":\"node-3\",\"status\":\"HEALTHY\",\"queue_depth\":0.01,\"capacity\":0.6,\"incoming_request_rate\":0.12,\"latency_ms\":0.02,\"outflow_rate\":0.12},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.06,\"capacity\":0.6,\"incoming_request_rate\":0.47,\"latency_ms\":0.021,\"outflow_rate\":0.45}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "REROUTE_TRAFFIC", "target_node_id": "node-1", "parameter": 0.7}',
    })

    # Expert: SHED_LOAD on non-critical node (task-3 surge)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-3\nObjective: " + TASK_BRIEFS["task-3"] + "\nStep: 35\nStatus: Cost: baseline ($1.50/hr) | Queues: rising (4 violations) | Reward: 0.25=BAD\n\nCurrent state:\n{\"task_id\":\"task-3\",\"step\":35,\"max_steps\":60,\"failed_nodes\":[],\"degraded_nodes\":[\"node-2\",\"node-3\"],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.05,\"capacity\":0.6,\"incoming_request_rate\":0.3,\"latency_ms\":0.02,\"outflow_rate\":0.3},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.225,\"capacity\":0.6,\"incoming_request_rate\":0.75,\"latency_ms\":0.022,\"outflow_rate\":0.45},{\"node_id\":\"node-2\",\"status\":\"DEGRADED\",\"queue_depth\":0.8,\"capacity\":0.6,\"incoming_request_rate\":0.75,\"latency_ms\":0.055,\"outflow_rate\":0.45},{\"node_id\":\"node-3\",\"status\":\"DEGRADED\",\"queue_depth\":0.7,\"capacity\":0.6,\"incoming_request_rate\":0.45,\"latency_ms\":0.048,\"outflow_rate\":0.45},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.025,\"capacity\":0.6,\"incoming_request_rate\":0.3,\"latency_ms\":0.02,\"outflow_rate\":0.3}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "SHED_LOAD", "target_node_id": "node-3", "parameter": 0.4}',
    })

    # Expert: SCALE_UP on downstream bottleneck (not node-0)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-1\nObjective: " + TASK_BRIEFS["task-1"] + "\nStep: 45\nStatus: Cost: baseline ($1.50/hr) | Queues: rising (2 violations) | Reward: 0.35=BAD\n\nCurrent state:\n{\"task_id\":\"task-1\",\"step\":45,\"max_steps\":100,\"failed_nodes\":[],\"degraded_nodes\":[\"node-2\",\"node-3\"],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.15,\"capacity\":0.6,\"incoming_request_rate\":0.44,\"latency_ms\":0.021,\"outflow_rate\":0.42},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.04,\"capacity\":0.6,\"incoming_request_rate\":0.22,\"latency_ms\":0.02,\"outflow_rate\":0.22},{\"node_id\":\"node-2\",\"status\":\"DEGRADED\",\"queue_depth\":0.925,\"capacity\":0.6,\"incoming_request_rate\":0.22,\"latency_ms\":0.075,\"outflow_rate\":0.22},{\"node_id\":\"node-3\",\"status\":\"DEGRADED\",\"queue_depth\":0.85,\"capacity\":0.6,\"incoming_request_rate\":0.22,\"latency_ms\":0.065,\"outflow_rate\":0.22},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.05,\"capacity\":0.6,\"incoming_request_rate\":0.43,\"latency_ms\":0.02,\"outflow_rate\":0.43}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "SCALE_UP", "target_node_id": "node-2", "parameter": 0.6}',
    })

    # Expert: SCALE_DOWN on idle node
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-1\nObjective: " + TASK_BRIEFS["task-1"] + "\nStep: 80\nStatus: Cost: high ($2.40/hr) | Queues: stable | Reward: 0.65=GOOD\n\nCurrent state:\n{\"task_id\":\"task-1\",\"step\":80,\"max_steps\":100,\"failed_nodes\":[],\"degraded_nodes\":[],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.025,\"capacity\":1.0,\"incoming_request_rate\":0.35,\"latency_ms\":0.02,\"outflow_rate\":0.35},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.015,\"capacity\":0.8,\"incoming_request_rate\":0.18,\"latency_ms\":0.02,\"outflow_rate\":0.18},{\"node_id\":\"node-2\",\"status\":\"HEALTHY\",\"queue_depth\":0.02,\"capacity\":1.0,\"incoming_request_rate\":0.18,\"latency_ms\":0.02,\"outflow_rate\":0.18},{\"node_id\":\"node-3\",\"status\":\"HEALTHY\",\"queue_depth\":0.01,\"capacity\":0.8,\"incoming_request_rate\":0.09,\"latency_ms\":0.02,\"outflow_rate\":0.09},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.015,\"capacity\":1.2,\"incoming_request_rate\":0.35,\"latency_ms\":0.02,\"outflow_rate\":0.35}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "SCALE_DOWN", "target_node_id": "node-4", "parameter": 0.4}',
    })

    # Expert: NO_OP on healthy cluster
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-1\nObjective: " + TASK_BRIEFS["task-1"] + "\nStep: 10\nStatus: Cost: baseline ($1.50/hr) | Queues: stable | Reward: 0.70=GOOD\n\nCurrent state:\n{\"task_id\":\"task-1\",\"step\":10,\"max_steps\":100,\"failed_nodes\":[],\"degraded_nodes\":[],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.04,\"capacity\":0.6,\"incoming_request_rate\":0.42,\"latency_ms\":0.02,\"outflow_rate\":0.42},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.025,\"capacity\":0.6,\"incoming_request_rate\":0.21,\"latency_ms\":0.02,\"outflow_rate\":0.21},{\"node_id\":\"node-2\",\"status\":\"HEALTHY\",\"queue_depth\":0.03,\"capacity\":0.6,\"incoming_request_rate\":0.21,\"latency_ms\":0.02,\"outflow_rate\":0.21},{\"node_id\":\"node-3\",\"status\":\"HEALTHY\",\"queue_depth\":0.015,\"capacity\":0.6,\"incoming_request_rate\":0.1,\"latency_ms\":0.02,\"outflow_rate\":0.1},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.035,\"capacity\":0.6,\"incoming_request_rate\":0.42,\"latency_ms\":0.02,\"outflow_rate\":0.42}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "NO_OP", "target_node_id": "node-0", "parameter": 0.0}',
    })

    # Expert: REROUTE_TRAFFIC on failed node-2 (task-2)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-2\nObjective: " + TASK_BRIEFS["task-2"] + "\nStep: 20\nStatus: Cost: baseline ($1.50/hr) | Queues: rising (2 violations) | Reward: 0.38=BAD\n\nCurrent state:\n{\"task_id\":\"task-2\",\"step\":20,\"max_steps\":60,\"failed_nodes\":[\"node-2\"],\"degraded_nodes\":[\"node-3\"],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.18,\"capacity\":0.6,\"incoming_request_rate\":0.45,\"latency_ms\":0.024,\"outflow_rate\":0.43},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.05,\"capacity\":0.6,\"incoming_request_rate\":0.23,\"latency_ms\":0.02,\"outflow_rate\":0.23},{\"node_id\":\"node-2\",\"status\":\"FAILED\",\"queue_depth\":0.0,\"capacity\":0.0,\"incoming_request_rate\":0.23,\"latency_ms\":1.0,\"outflow_rate\":0.0},{\"node_id\":\"node-3\",\"status\":\"DEGRADED\",\"queue_depth\":0.45,\"capacity\":0.6,\"incoming_request_rate\":0.23,\"latency_ms\":0.038,\"outflow_rate\":0.15},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.04,\"capacity\":0.6,\"incoming_request_rate\":0.42,\"latency_ms\":0.02,\"outflow_rate\":0.42}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "REROUTE_TRAFFIC", "target_node_id": "node-2", "parameter": 0.7}',
    })

    # Expert: SHED_LOAD on node-4 during surge (task-3)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-3\nObjective: " + TASK_BRIEFS["task-3"] + "\nStep: 30\nStatus: Cost: high ($1.80/hr) | Queues: rising (5 violations) | Reward: 0.22=BAD\n\nCurrent state:\n{\"task_id\":\"task-3\",\"step\":30,\"max_steps\":60,\"failed_nodes\":[],\"degraded_nodes\":[\"node-4\"],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.05,\"capacity\":0.6,\"incoming_request_rate\":0.3,\"latency_ms\":0.02,\"outflow_rate\":0.3},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.15,\"capacity\":0.8,\"incoming_request_rate\":0.75,\"latency_ms\":0.022,\"outflow_rate\":0.6},{\"node_id\":\"node-2\",\"status\":\"HEALTHY\",\"queue_depth\":0.12,\"capacity\":0.8,\"incoming_request_rate\":0.75,\"latency_ms\":0.021,\"outflow_rate\":0.6},{\"node_id\":\"node-3\",\"status\":\"HEALTHY\",\"queue_depth\":0.08,\"capacity\":0.6,\"incoming_request_rate\":0.45,\"latency_ms\":0.02,\"outflow_rate\":0.45},{\"node_id\":\"node-4\",\"status\":\"DEGRADED\",\"queue_depth\":0.65,\"capacity\":0.6,\"incoming_request_rate\":0.55,\"latency_ms\":0.045,\"outflow_rate\":0.4}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "SHED_LOAD", "target_node_id": "node-4", "parameter": 0.4}',
    })

    # Expert: SCALE_DOWN on node-1 when overprovisioned (task-1 late)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-1\nObjective: " + TASK_BRIEFS["task-1"] + "\nStep: 75\nStatus: Cost: high ($2.20/hr) | Queues: stable | Reward: 0.68=GOOD\n\nCurrent state:\n{\"task_id\":\"task-1\",\"step\":75,\"max_steps\":100,\"failed_nodes\":[],\"degraded_nodes\":[],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.02,\"capacity\":1.0,\"incoming_request_rate\":0.35,\"latency_ms\":0.02,\"outflow_rate\":0.35},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.01,\"capacity\":1.0,\"incoming_request_rate\":0.18,\"latency_ms\":0.02,\"outflow_rate\":0.18},{\"node_id\":\"node-2\",\"status\":\"HEALTHY\",\"queue_depth\":0.015,\"capacity\":0.8,\"incoming_request_rate\":0.18,\"latency_ms\":0.02,\"outflow_rate\":0.18},{\"node_id\":\"node-3\",\"status\":\"HEALTHY\",\"queue_depth\":0.01,\"capacity\":0.6,\"incoming_request_rate\":0.09,\"latency_ms\":0.02,\"outflow_rate\":0.09},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.02,\"capacity\":0.8,\"incoming_request_rate\":0.35,\"latency_ms\":0.02,\"outflow_rate\":0.35}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "SCALE_DOWN", "target_node_id": "node-1", "parameter": 0.3}',
    })

    # Expert: SCALE_UP on node-3 (downstream child, not node-0)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-1\nObjective: " + TASK_BRIEFS["task-1"] + "\nStep: 50\nStatus: Cost: baseline ($1.50/hr) | Queues: rising (3 violations) | Reward: 0.30=BAD\n\nCurrent state:\n{\"task_id\":\"task-1\",\"step\":50,\"max_steps\":100,\"failed_nodes\":[],\"degraded_nodes\":[\"node-3\"],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.1,\"capacity\":0.8,\"incoming_request_rate\":0.5,\"latency_ms\":0.021,\"outflow_rate\":0.48},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.05,\"capacity\":0.6,\"incoming_request_rate\":0.25,\"latency_ms\":0.02,\"outflow_rate\":0.25},{\"node_id\":\"node-2\",\"status\":\"HEALTHY\",\"queue_depth\":0.08,\"capacity\":0.8,\"incoming_request_rate\":0.25,\"latency_ms\":0.022,\"outflow_rate\":0.23},{\"node_id\":\"node-3\",\"status\":\"DEGRADED\",\"queue_depth\":0.6,\"capacity\":0.6,\"incoming_request_rate\":0.23,\"latency_ms\":0.042,\"outflow_rate\":0.2},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.04,\"capacity\":0.6,\"incoming_request_rate\":0.48,\"latency_ms\":0.02,\"outflow_rate\":0.48}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "SCALE_UP", "target_node_id": "node-3", "parameter": 0.5}',
    })

    # Expert: NO_OP on healthy task-2 cluster (no failures visible)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-2\nObjective: " + TASK_BRIEFS["task-2"] + "\nStep: 8\nStatus: Cost: baseline ($1.50/hr) | Queues: stable | Reward: 0.72=GOOD\n\nCurrent state:\n{\"task_id\":\"task-2\",\"step\":8,\"max_steps\":60,\"failed_nodes\":[],\"degraded_nodes\":[],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.035,\"capacity\":0.6,\"incoming_request_rate\":0.47,\"latency_ms\":0.02,\"outflow_rate\":0.45},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.02,\"capacity\":0.6,\"incoming_request_rate\":0.23,\"latency_ms\":0.02,\"outflow_rate\":0.23},{\"node_id\":\"node-2\",\"status\":\"HEALTHY\",\"queue_depth\":0.025,\"capacity\":0.6,\"incoming_request_rate\":0.23,\"latency_ms\":0.02,\"outflow_rate\":0.23},{\"node_id\":\"node-3\",\"status\":\"HEALTHY\",\"queue_depth\":0.01,\"capacity\":0.6,\"incoming_request_rate\":0.12,\"latency_ms\":0.02,\"outflow_rate\":0.12},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.03,\"capacity\":0.6,\"incoming_request_rate\":0.47,\"latency_ms\":0.02,\"outflow_rate\":0.45}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "NO_OP", "target_node_id": "node-0", "parameter": 0.0}',
    })

    # Expert: SCALE_DOWN on node-2 when overprovisioned in task-3
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": "Task: task-3\nObjective: " + TASK_BRIEFS["task-3"] + "\nStep: 50\nStatus: Cost: high ($2.10/hr) | Queues: stable | Reward: 0.62=GOOD\n\nCurrent state:\n{\"task_id\":\"task-3\",\"step\":50,\"max_steps\":60,\"failed_nodes\":[],\"degraded_nodes\":[],\"nodes\":[{\"node_id\":\"node-0\",\"status\":\"HEALTHY\",\"queue_depth\":0.03,\"capacity\":0.6,\"incoming_request_rate\":0.3,\"latency_ms\":0.02,\"outflow_rate\":0.3},{\"node_id\":\"node-1\",\"status\":\"HEALTHY\",\"queue_depth\":0.02,\"capacity\":1.0,\"incoming_request_rate\":0.55,\"latency_ms\":0.02,\"outflow_rate\":0.55},{\"node_id\":\"node-2\",\"status\":\"HEALTHY\",\"queue_depth\":0.015,\"capacity\":1.0,\"incoming_request_rate\":0.55,\"latency_ms\":0.02,\"outflow_rate\":0.55},{\"node_id\":\"node-3\",\"status\":\"HEALTHY\",\"queue_depth\":0.01,\"capacity\":0.6,\"incoming_request_rate\":0.28,\"latency_ms\":0.02,\"outflow_rate\":0.28},{\"node_id\":\"node-4\",\"status\":\"HEALTHY\",\"queue_depth\":0.02,\"capacity\":0.6,\"incoming_request_rate\":0.3,\"latency_ms\":0.02,\"outflow_rate\":0.3}]}\n\nChoose the next SRE action.",
        "assistant": '{"action_type": "SCALE_DOWN", "target_node_id": "node-2", "parameter": 0.3}',
    })

    return examples


# ════════════════════════════════════════════════════════════════════════════════
# Dataset generation
# ════════════════════════════════════════════════════════════════════════════════

def generate_dataset(hf_space_url, output_dir, seed, episodes_per_task, max_steps, no_op_cap):
    """Run heuristic episodes against HF Space and build SFT dataset."""
    all_examples = []
    action_counts = Counter()
    node_counts = Counter()
    reward_sum = 0.0
    reward_count = 0

    for task_id in ["task-1", "task-2", "task-3"]:
        print(f"\n--- Generating episodes for {task_id} ---")
        for ep_idx in range(episodes_per_task):
            ep_seed = seed + ep_idx * 100 + hash(task_id) % 1000

            reset_resp = env_reset(hf_space_url, task_id=task_id, seed=ep_seed)
            obs_dict = reset_resp.get("observation", reset_resp)
            sla_violations = obs_dict.get("sla_violations", 0)
            episode_reward = 0.0

            for step in range(1, max_steps + 1):
                obs_text = format_observation(obs_dict, task_id, step, max_steps, episode_reward, sla_violations)
                action_type, target_node_id, parameter = heuristic_action(
                    obs_dict, task_id, step=step, max_steps=max_steps,
                    episode_reward=episode_reward,
                )
                assistant_text = make_assistant_text(action_type, target_node_id, parameter)

                all_examples.append({
                    "system": SYSTEM_PROMPT,
                    "user": obs_text,
                    "assistant": assistant_text,
                    "task_id": task_id,
                    "action_type": action_type.value,
                    "target_node_id": target_node_id,
                    "reward": episode_reward,
                })

                action_counts[action_type.value] += 1
                node_counts[target_node_id] += 1

                step_resp = env_step(hf_space_url, action_type.value, target_node_id, parameter)
                obs_dict = step_resp.get("observation", step_resp)
                episode_reward = step_resp.get("reward", episode_reward)
                reward_sum += episode_reward
                reward_count += 1
                sla_violations = obs_dict.get("sla_violations", sla_violations)

                if step_resp.get("done", False):
                    break

            if (ep_idx + 1) % 5 == 0:
                print(f"  Episode {ep_idx+1}/{episodes_per_task} done")

    print(f"\nHeuristic episodes generated: {len(all_examples)} examples")

    # ---- Add expert examples ----
    for ex in build_expert_examples():
        at_match = re.search(r'"action_type":\s*"(\w+)"', ex["assistant"])
        tn_match = re.search(r'"target_node_id":\s*"(node-\d+)"', ex["assistant"])
        if at_match and tn_match:
            all_examples.append({
                "system": ex["system"],
                "user": ex["user"],
                "assistant": ex["assistant"],
                "task_id": "expert",
                "action_type": at_match.group(1),
                "target_node_id": tn_match.group(1),
                "reward": 0.0,
            })
            action_counts[at_match.group(1)] += 1
            node_counts[tn_match.group(1)] += 1

    print(f"After expert augmentation: {len(all_examples)} examples")

    # ---- Oversample underrepresented action types ----
    total = len(all_examples)
    target_min_pct = 0.05  # Each action type should be ≥5% of dataset
    target_min_count = max(20, int(total * target_min_pct))
    for at in VALID_ACTIONS:
        existing = [e for e in all_examples if e["action_type"] == at]
        if len(existing) < target_min_count and existing:
            multiplier = target_min_count // len(existing)
            remainder = target_min_count % len(existing)
            oversampled = existing * multiplier + random.sample(existing, remainder)
            all_examples.extend(oversampled)
            print(f"Oversampled {at}: {len(existing)} → {len(existing) + len(oversampled)}")

    # ---- Oversample underrepresented target nodes ----
    target_node_min_pct = 0.08  # Each node should be ≥8% of dataset
    target_node_min_count = max(20, int(len(all_examples) * target_node_min_pct))
    for nid in VALID_NODES:
        existing = [e for e in all_examples if e["target_node_id"] == nid]
        if len(existing) < target_node_min_count and existing:
            multiplier = target_node_min_count // len(existing)
            remainder = target_node_min_count % len(existing)
            oversampled = existing * multiplier + random.sample(existing, remainder)
            all_examples.extend(oversampled)
            print(f"Oversampled {nid}: {len(existing)} → {len(existing) + len(oversampled)}")

    print(f"After oversampling: {len(all_examples)} examples")

    # ---- Filter: cap NO_OP ----
    noop_examples = [e for e in all_examples if e["action_type"] == "NO_OP"]
    non_noop_examples = [e for e in all_examples if e["action_type"] != "NO_OP"]
    max_noop = int(len(all_examples) * no_op_cap)
    if len(noop_examples) > max_noop:
        random.shuffle(noop_examples)
        noop_examples = noop_examples[:max_noop]
        all_examples = non_noop_examples + noop_examples
        print(f"NO_OP capped: keeping {max_noop}")

    # ---- Deduplicate ----
    seen = set()
    unique_examples = []
    for e in all_examples:
        key = (e["user"][:100], e["action_type"], e["target_node_id"])
        if key not in seen:
            seen.add(key)
            unique_examples.append(e)
    all_examples = unique_examples
    print(f"After dedup: {len(all_examples)} examples")

    # ---- Shuffle ----
    random.shuffle(all_examples)

    # ---- Train/Test/Val split (70/15/15) ----
    train_ex, temp_ex = train_test_split(all_examples, test_size=0.30, random_state=seed)
    test_ex, val_ex = train_test_split(temp_ex, test_size=0.50, random_state=seed)

    print(f"\nDataset split:")
    print(f"  Train: {len(train_ex)}")
    print(f"  Test:  {len(test_ex)}")
    print(f"  Val:   {len(val_ex)}")

    # ---- Save as JSONL ----
    def save_jsonl(examples, path):
        with open(path, "w") as f:
            for ex in examples:
                record = {
                    "messages": [
                        {"role": "system", "content": ex["system"]},
                        {"role": "user", "content": ex["user"]},
                        {"role": "assistant", "content": ex["assistant"]},
                    ],
                    "task_id": ex.get("task_id", ""),
                    "action_type": ex.get("action_type", ""),
                    "target_node_id": ex.get("target_node_id", ""),
                }
                f.write(json.dumps(record) + "\n")

    save_jsonl(train_ex, output_dir / "sft_train.jsonl")
    save_jsonl(test_ex, output_dir / "sft_test.jsonl")
    save_jsonl(val_ex, output_dir / "sft_val.jsonl")

    # ---- Print statistics ----
    final_action_counts = Counter(e["action_type"] for e in all_examples)
    final_node_counts = Counter(e["target_node_id"] for e in all_examples)
    total = len(all_examples)

    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS ({total} total examples)")
    print(f"{'='*60}")
    print(f"\nAction type distribution:")
    for at in VALID_ACTIONS:
        cnt = final_action_counts.get(at, 0)
        pct = 100 * cnt / total if total > 0 else 0
        flag = " <-- UNDERREPRESENTED" if pct < 5 else ""
        print(f"  {at:20s}: {cnt:5d} ({pct:5.1f}%){flag}")

    print(f"\nTarget node distribution:")
    for nid in VALID_NODES:
        cnt = final_node_counts.get(nid, 0)
        pct = 100 * cnt / total if total > 0 else 0
        flag = " <-- UNDERREPRESENTED" if pct < 5 else ""
        print(f"  {nid:10s}: {cnt:5d} ({pct:5.1f}%){flag}")

    if reward_count > 0:
        print(f"\nAverage reward: {reward_sum / reward_count:.4f}")

    print(f"\nFiles saved to {output_dir}:")
    for f in sorted(output_dir.glob("sft_*.jsonl")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")


# ════════════════════════════════════════════════════════════════════════════════
# Model loading
# ════════════════════════════════════════════════════════════════════════════════

def load_model(model_name, max_seq_length, lora_rank, lora_alpha, load_in_4bit, seed):
    """Load base model + attach LoRA adapters via Unsloth."""
    if not _TORCH_OK:
        raise ImportError(
            f"Cannot load model — torch/unsloth import failed:\n"
            f"  {_TORCH_ERR}\n"
            f"Ensure torch+torchvision are installed with matching CUDA version, e.g.:\n"
            f"  pip install torch==X.Y.Z+cuNNN torchvision==A.B.C+cuNNN "
            f"--index-url https://download.pytorch.org/whl/cuNNN"
        )
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect (bf16 on A100, fp16 on T4+)
        trust_remote_code=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    # VRAM report
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM used: {vram_used:.2f} GiB / {vram_total:.2f} GiB")
        print(f"VRAM free: {vram_total - vram_used:.2f} GiB")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {model_name}")
    print(f"LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"Max seq length: {max_seq_length}")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params: {total:,}")

    return model, tokenizer


# ════════════════════════════════════════════════════════════════════════════════
# SFT Training
# ════════════════════════════════════════════════════════════════════════════════

def run_training(model, tokenizer, output_dir, args):
    """Run SFT training with Unsloth + SFTTrainer."""
    def load_jsonl(path):
        records = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    train_records = load_jsonl(output_dir / "sft_train.jsonl")
    val_records = load_jsonl(output_dir / "sft_val.jsonl")

    # Pre-format into "text" field using chat template
    def apply_template(record):
        return {"text": tokenizer.apply_chat_template(
            record["messages"], tokenize=False, add_generation_prompt=False,
            enable_thinking=False
        )}

    train_dataset = Dataset.from_list(train_records).map(apply_template)
    val_dataset = Dataset.from_list(val_records).map(apply_template)

    print(f"Train: {len(train_dataset)}  Val: {len(val_dataset)}")

    adapter_dir = str(output_dir / "sft_lora_adapter")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.learning_rate,
            num_train_epochs=args.epochs,
            max_seq_length=args.max_seq_length,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_total_limit=3,
            bf16=args.bf16,
            optim="adamw_8bit",
            output_dir=adapter_dir,
            dataset_text_field="text",
            packing=False,
        ),
    )

    print(f"\nEffective batch size: {args.batch_size * args.grad_accum}")
    print(f"Total epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output dir: {adapter_dir}")
    print("\nStarting training...")

    train_result = trainer.train()

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Train loss: {train_result.training_loss:.4f}")
    print(f"Train runtime: {train_result.metrics['train_runtime']:.1f}s")

    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"\nLoRA adapter saved to: {adapter_dir}")

    eval_result = trainer.evaluate()
    print(f"\nEval loss: {eval_result.get('eval_loss', 'N/A')}")
    if "eval_loss" in eval_result:
        print(f"Train/Eval loss ratio: {train_result.training_loss / eval_result['eval_loss']:.3f}")

    return model, tokenizer, adapter_dir


# ════════════════════════════════════════════════════════════════════════════════
# Quality Gate — Live evaluation via HF Space API
# ════════════════════════════════════════════════════════════════════════════════

def parse_model_action(text):
    """Extract action from model output text."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None, "no JSON found"
        obj = json.loads(text[start:end+1])
        at = str(obj.get("action_type", "")).upper()
        if at not in VALID_ACTIONS:
            return None, f"invalid action_type: {at}"
        nid = str(obj.get("target_node_id", ""))
        if nid not in VALID_NODES:
            return None, f"invalid target_node_id: {nid}"
        param = float(obj.get("parameter", 0.0))
        return {
            "action_type": at,
            "target_node_id": nid,
            "parameter": param,
        }, "ok"
    except Exception as e:
        return None, str(e)


def run_eval_episode_with_model(hf_space_url, model, tokenizer, task_id, max_steps, episode_reward=0.0):
    """Run one evaluation episode using the SFT model via HF Space API."""
    FastLanguageModel.for_inference(model)

    reset_resp = env_reset(hf_space_url, task_id=task_id, seed=None)
    obs_dict = reset_resp.get("observation", reset_resp)
    sla_violations = obs_dict.get("sla_violations", 0)
    rewards = []
    crashes = 0
    action_log = []

    for step in range(1, max_steps + 1):
        obs_text = format_observation(obs_dict, task_id, step, max_steps, episode_reward, sla_violations)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                temperature=1.0,
            )
        generated = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Strip TRACE if present
        import re
        generated = re.sub(
            '\x3cthink\x3e.*?\x3c/think\x3e', '',
            generated, flags=re.DOTALL
        ).strip()

        action, err = parse_model_action(generated)
        if action is None:
            crashes += 1
            # Fallback to NO_OP
            action = {"action_type": "NO_OP", "target_node_id": "node-0", "parameter": 0.0}
            action_log.append(f"step={step} INVALID ({err})")
        else:
            action_log.append(f"step={step} {action['action_type']} {action['target_node_id']} p={action['parameter']:.2f}")

        step_resp = env_step(hf_space_url, action["action_type"], action["target_node_id"], action["parameter"])
        obs_dict = step_resp.get("observation", step_resp)
        step_reward = step_resp.get("reward", 0.0)
        rewards.append(step_reward)
        episode_reward = step_reward
        sla_violations = obs_dict.get("sla_violations", sla_violations)

        if step_resp.get("done", False):
            break

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return {
        "avg_reward": avg_reward,
        "crashes": crashes,
        "action_log": action_log,
    }


def run_eval_episode_with_heuristic(hf_space_url, task_id, max_steps):
    """Run one evaluation episode using the heuristic baseline via HF Space API."""
    reset_resp = env_reset(hf_space_url, task_id=task_id, seed=None)
    obs_dict = reset_resp.get("observation", reset_resp)
    rewards = []

    for step in range(1, max_steps + 1):
        action_type, target_node_id, parameter = heuristic_action(obs_dict, task_id)
        step_resp = env_step(hf_space_url, action_type.value, target_node_id, parameter)
        obs_dict = step_resp.get("observation", step_resp)
        rewards.append(step_resp.get("reward", 0.0))

        if step_resp.get("done", False):
            break

    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return {"avg_reward": avg_reward}


def run_quality_gate(hf_space_url, model, tokenizer, eval_episodes, eval_steps):
    """Compare SFT model vs heuristic baseline on live episodes."""
    print(f"\nRunning {eval_episodes} episodes per task ({eval_steps} steps each)...")
    print(f"{'='*70}")

    results = {}
    for task_id in ["task-1", "task-2", "task-3"]:
        sft_rewards = []
        heuristic_rewards = []
        total_crashes = 0

        for ep in range(eval_episodes):
            sft_result = run_eval_episode_with_model(hf_space_url, model, tokenizer, task_id, eval_steps)
            sft_rewards.append(sft_result["avg_reward"])
            total_crashes += sft_result["crashes"]

            heur_result = run_eval_episode_with_heuristic(hf_space_url, task_id, eval_steps)
            heuristic_rewards.append(heur_result["avg_reward"])

        sft_avg = sum(sft_rewards) / len(sft_rewards)
        heur_avg = sum(heuristic_rewards) / len(heuristic_rewards)

        results[task_id] = {
            "sft_avg_reward": sft_avg,
            "heuristic_avg_reward": heur_avg,
            "crashes": total_crashes,
            "sft_beats_heuristic": sft_avg >= heur_avg,
        }

        status = "SFT WINS" if sft_avg >= heur_avg else "HEURISTIC WINS"
        print(f"\n{task_id}:")
        print(f"  SFT avg reward:       {sft_avg:.4f}")
        print(f"  Heuristic avg reward: {heur_avg:.4f}")
        print(f"  Result: {status}")
        print(f"  Crashes: {total_crashes}")

    tasks_won = sum(1 for r in results.values() if r["sft_beats_heuristic"])
    total_crashes = sum(r["crashes"] for r in results.values())

    print(f"\n{'='*70}")
    print(f"QUALITY GATE SUMMARY")
    print(f"{'='*70}")
    print(f"SFT beats heuristic on: {tasks_won}/3 tasks")
    print(f"Total crashes: {total_crashes}")
    gate_pass = total_crashes == 0 and tasks_won >= 2
    print(f"\nGate results:")
    print(f"  Zero crashes: {'PASS' if total_crashes == 0 else 'FAIL'}")
    print(f"  Beat heuristic >= 2/3: {'PASS' if tasks_won >= 2 else 'FAIL'}")
    print(f"\nOverall: {'GATE PASSED' if gate_pass else 'GATE FAILED — consider more SFT epochs or larger dataset'}")


# ════════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AntiAtropos SFT Training — QLoRA via Unsloth"
    )

    # ---- Data generation ----
    parser.add_argument("--hf-space-url", type=str,
                        default="https://pranavkk-antiatropos.hf.space",
                        help="HF Space URL for environment API")
    parser.add_argument("--episodes-per-task", type=int, default=20,
                        help="Heuristic episodes per task for data gen (default: 20)")
    parser.add_argument("--max-steps", type=int, default=60,
                        help="Max steps per episode for data gen (default: 60)")
    parser.add_argument("--no-op-cap", type=float, default=0.40,
                        help="Max fraction of NO_OP examples (default: 0.40)")
    parser.add_argument("--skip-data-gen", action="store_true",
                        help="Skip data generation (use existing JSONL files)")

    # ---- Model ----
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3.5-4B",
                        help="Base model name (default: Qwen/Qwen3.5-4B)")
    parser.add_argument("--max-seq-length", type=int, default=2048,
                        help="Max sequence length (default: 2048)")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha (default: 16)")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Load model in 4-bit (default: True)")

    # ---- Training ----
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Per-device train batch size (default: 1)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation steps (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2)")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Warmup steps (default: 50)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--logging-steps", type=int, default=10,
                        help="Log every N steps (default: 10)")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="Save checkpoint every N steps (default: 100)")
    parser.add_argument("--eval-steps", type=int, default=100,
                        help="Eval every N steps (default: 100)")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bf16 precision (default: True)")

    # ---- Eval / quality gate ----
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip quality-gate evaluation after training")
    parser.add_argument("--eval-episodes", type=int, default=3,
                        help="Episodes per task for quality gate (default: 3)")
    parser.add_argument("--qg-max-steps", type=int, default=60,
                        help="Max steps per quality gate episode (default: 60)")

    # ---- General ----
    parser.add_argument("--output-dir", type=str, default="./sft_output",
                        help="Output directory (default: ./sft_output)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    # ---- Setup ----
    random.seed(args.seed)
    if _TORCH_OK:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    else:
        print(f"WARNING: torch not available ({_TORCH_ERR})")
        print("  Data generation will work, but model training requires torch+unsloth.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" AntiAtropos SFT Training")
    print("=" * 60)
    print(f"  Model:          {args.model_name}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  HF Space:       {args.hf_space_url}")
    print(f"  Episodes/task:  {args.episodes_per_task}")
    print(f"  Max steps:      {args.max_steps}")
    print(f"  Epochs:         {args.epochs}")
    print(f"  Batch size:     {args.batch_size} x {args.grad_accum} accum = {args.batch_size * args.grad_accum}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Output dir:     {output_dir}")
    print(f"  Seed:           {args.seed}")
    print("=" * 60)

    # ---- Verify HF Space ----
    print("\nVerifying HF Space connectivity...")
    try:
        test = env_reset(args.hf_space_url, "task-1")
        step_result = env_step(args.hf_space_url, "NO_OP", "node-0", 0.0)
        print(f"Reset OK — task_id={test.get('observation', {}).get('task_id')}")
        print(f"Step OK — reward={step_result.get('reward')}, done={step_result.get('done')}")
    except Exception as e:
        print(f"FATAL: Cannot reach HF Space at {args.hf_space_url}: {e}")
        sys.exit(1)

    # ---- Data generation ----
    if args.skip_data_gen:
        # Check that JSONL files exist
        for fname in ["sft_train.jsonl", "sft_val.jsonl", "sft_test.jsonl"]:
            if not (output_dir / fname).exists():
                print(f"ERROR: --skip-data-gen but {fname} not found in {output_dir}")
                sys.exit(1)
        print("\nSkipping data generation — using existing JSONL files.")
    else:
        print("\nGenerating SFT dataset...")
        generate_dataset(
            hf_space_url=args.hf_space_url,
            output_dir=output_dir,
            seed=args.seed,
            episodes_per_task=args.episodes_per_task,
            max_steps=args.max_steps,
            no_op_cap=args.no_op_cap,
        )

    # ---- Load model ----
    print("\nLoading model...")
    model, tokenizer = load_model(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        seed=args.seed,
    )

    # ---- Train ----
    model, tokenizer, adapter_dir = run_training(model, tokenizer, output_dir, args)

    # ---- Quality gate ----
    if args.skip_eval:
        print("\nSkipping quality-gate evaluation (--skip-eval).")
    else:
        run_quality_gate(
            hf_space_url=args.hf_space_url,
            model=model,
            tokenizer=tokenizer,
            eval_episodes=args.eval_episodes,
            eval_steps=args.qg_max_steps,
        )

    print(f"\nDone. Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
