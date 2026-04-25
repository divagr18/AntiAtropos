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
import time
from collections import Counter
from enum import Enum
from pathlib import Path

import requests
import torch
from datasets import Dataset
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer


# ════════════════════════════════════════════════════════════════════════════════
# Constants
# ════════════════════════════════════════════════════════════════════════════════

VALID_ACTIONS = ["NO_OP", "SCALE_UP", "SCALE_DOWN", "REROUTE_TRAFFIC", "SHED_LOAD"]
VALID_NODES = ["node-0", "node-1", "node-2", "node-3", "node-4"]
MAX_QUEUE_NORM = 200.0
MAX_LATENCY_NORM = 1000.0
MAX_REQUEST_RATE_NORM = 100.0


SYSTEM_PROMPT = """You are an autonomous SRE controller managing a five-node microservice cluster.

CLUSTER TOPOLOGY (traffic flows parent -> children):
  node-0 (VIP payment gateway) -> node-1, node-2
  node-2 (catalog) -> node-3 (inventory)
  node-4 (auth, independent ingress)
FAILED nodes have outflow=0 — their children are starved.
Backpressure: overloaded children reduce parent capacity.

ACTIONS (new capacity takes 5 ticks to boot):
  SCALE_UP <node> <amount>   — add capacity (0.3-0.5 normal, 0.6-0.8 heavy surge)
  SCALE_DOWN <node> <amount>  — remove capacity (0.2-0.4 safe, 0.5-0.7 aggressive)
  REROUTE_TRAFFIC <node> <fraction> — move traffic AWAY from this node to healthy peers (0.3-0.7)
  SHED_LOAD <node> <fraction>  — drop incoming traffic (0.3-0.5), NEVER on node-0 (VIP)
  NO_OP — do nothing when cluster is healthy

CRITICAL RULES:
  - node-0 is the VIP payment gateway — NEVER shed its traffic
  - REROUTE_TRAFFIC moves traffic AWAY FROM the target node
  - SCALE_UP clears DEGRADED status on the target node
  - Boot delay is 5 ticks — plan ahead for scaling
  - Use English for reasoning, JSON for the action

REWARD PRIORITIES (in order):
  1. Avoid SLA violations (latency > 200ms or error rate > 5%)
  2. Keep queues low (growing queues = destabilizing system)
  3. Don't over-provision (excess capacity costs money)

You MUST respond with one sentence of English reasoning, then a JSON action.
The JSON must use EXACTLY these keys: action_type, target_node_id, parameter.
action_type must be one of: SCALE_UP, SCALE_DOWN, REROUTE_TRAFFIC, SHED_LOAD, NO_OP.
target_node_id must be one of: node-0, node-1, node-2, node-3, node-4.
parameter must be a float between 0.0 and 10.0."""


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
    """Convert API observation dict to natural-language string."""
    lines = [f"Task: {task_id}  Step: {step}/{max_steps}  Reward: {reward:.3f}  SLA violations: {sla_violations}"]
    lines.append("")
    lines.append("Node states:")
    for n in obs_dict.get("nodes", []):
        vip = " (VIP)" if n.get("is_vip") else ""
        status = n.get("status", "HEALTHY")
        q = n.get("queue_depth", 0) * 200
        cap = n.get("capacity", 0) * 5
        pending = n.get("pending_capacity", 0) * 5
        inc = n.get("incoming_request_rate", 0) * 100
        lat = n.get("latency_ms", 0) * 1000
        outflow = n.get("outflow_rate", 0) * 100
        failed = " [FAILED, outflow=0]" if status == "FAILED" else ""
        degraded = " [DEGRADED]" if status == "DEGRADED" else ""
        pending_str = f" (+{pending:.0f} booting)" if pending > 0 else ""
        lines.append(
            f"  {n['node_id']}{vip}: queue={int(q)}, capacity={cap:.0f}{pending_str}, "
            f"incoming={inc:.0f}, latency={lat:.0f}ms, outflow={outflow:.0f}{failed}{degraded}"
        )
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════════
# Heuristic policy
# ════════════════════════════════════════════════════════════════════════════════

def heuristic_action(obs_dict, task_id):
    """Enhanced heuristic that covers all action types. Works with normalized API data."""
    nodes = obs_dict.get("nodes", [])

    total_queue = sum(n["queue_depth"] * 200 for n in nodes)
    avg_latency = sum(n["latency_ms"] for n in nodes) / len(nodes) if nodes else 0
    failed_nodes = [n for n in nodes if n.get("status") == "FAILED"]

    # PRIORITY 1: REROUTE away from FAILED nodes
    if failed_nodes:
        target = failed_nodes[0]
        return ActionType.REROUTE_TRAFFIC, target["node_id"], 0.7

    # PRIORITY 2: SHED_LOAD on non-critical overloaded nodes
    non_critical_overloaded = [
        n for n in nodes
        if n["queue_depth"] > 0.6 and n["node_id"] != "node-0"
        and n.get("status") != "FAILED"
    ]
    if non_critical_overloaded and avg_latency > 0.05:
        shed_candidates = [n for n in non_critical_overloaded if n["node_id"] in ["node-3", "node-4"]]
        target = shed_candidates[0] if shed_candidates else non_critical_overloaded[0]
        return ActionType.SHED_LOAD, target["node_id"], 0.4

    # PRIORITY 3: SCALE_UP stressed nodes
    if avg_latency > 0.03 or total_queue > 200:
        target = max(nodes, key=lambda n: n["queue_depth"])
        param = 0.6 if target["queue_depth"] > 0.75 else 0.4
        return ActionType.SCALE_UP, target["node_id"], param

    # PRIORITY 4: SCALE_DOWN idle/overprovisioned nodes
    non_vips = [n for n in nodes if not n.get("is_vip", False) and n.get("status") != "FAILED"]
    if non_vips and avg_latency < 0.025 and total_queue < 50:
        overprov = [n for n in non_vips if n.get("capacity", 0) > 0.6]
        if overprov:
            target = max(overprov, key=lambda n: n.get("capacity", 0))
            return ActionType.SCALE_DOWN, target["node_id"], 0.3

    # PRIORITY 5: NO_OP when healthy
    target = max(nodes, key=lambda n: n["queue_depth"])
    return ActionType.NO_OP, target["node_id"], 0.0


def generate_reasoning(action_type, target_node_id, parameter, obs_dict, task_id):
    """Generate an English reasoning sentence for the chosen action."""
    nodes = obs_dict.get("nodes", [])
    node_map = {n["node_id"]: n for n in nodes}
    target = node_map.get(target_node_id, {})
    q = target.get("queue_depth", 0) * 200

    if action_type == ActionType.SCALE_UP:
        if q > 150:
            return f"Node {target_node_id} is near FATAL with queue={int(q)}, scaling up immediately to prevent cascading failure."
        elif q > 80:
            return f"Node {target_node_id} is DEGRADED with queue={int(q)}, scaling up to restore service capacity."
        else:
            return f"Node {target_node_id} queue={int(q)} is rising, proactively scaling up to stay ahead of demand."
    elif action_type == ActionType.SCALE_DOWN:
        return f"Cluster is stable with low queues, scaling down {target_node_id} to reduce infrastructure cost."
    elif action_type == ActionType.REROUTE_TRAFFIC:
        status = target.get("status", "HEALTHY")
        if status == "FAILED":
            return f"Node {target_node_id} has FAILED with outflow=0, rerouting its traffic to healthy peers to prevent request loss."
        else:
            return f"Node {target_node_id} is overloaded with queue={int(q)}, rerouting traffic to healthy peers to reduce pressure."
    elif action_type == ActionType.SHED_LOAD:
        return f"Node {target_node_id} is overloaded but non-critical, shedding {parameter:.0%} of incoming traffic to protect cluster stability."
    elif action_type == ActionType.NO_OP:
        return "All nodes are healthy with low queues, no intervention needed."
    return "Taking action based on current cluster state."


# ════════════════════════════════════════════════════════════════════════════════
# Expert augmentation examples
# ════════════════════════════════════════════════════════════════════════════════

def build_expert_examples():
    examples = []

    # Expert: REROUTE_TRAFFIC on FAILED node (task-2 scenario)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": """Task: task-2  Step: 25/60  Reward: 0.320  SLA violations: 3

Node states:
  node-0 (VIP): queue=55, capacity=3, incoming=47, latency=25ms, outflow=45
  node-1: queue=0, capacity=0, incoming=23, latency=inf, outflow=0 [FAILED, outflow=0]
  node-2: queue=18, capacity=3, incoming=23, latency=22ms, outflow=23
  node-3: queue=2, capacity=3, incoming=12, latency=20ms, outflow=12
  node-4: queue=12, capacity=3, incoming=47, latency=21ms, outflow=45""",
        "assistant": 'Node node-1 has FAILED with outflow=0, rerouting its traffic to healthy peers to prevent request loss. {"action_type": "REROUTE_TRAFFIC", "target_node_id": "node-1", "parameter": 0.7}',
    })

    examples.append({
        "system": SYSTEM_PROMPT,
        "user": """Task: task-2  Step: 30/60  Reward: 0.280  SLA violations: 5

Node states:
  node-0 (VIP): queue=65, capacity=3, incoming=47, latency=28ms, outflow=43
  node-1: queue=0, capacity=0, incoming=24, latency=inf, outflow=0 [FAILED, outflow=0]
  node-2: queue=90, capacity=3, incoming=24, latency=35ms, outflow=24
  node-3: queue=5, capacity=3, incoming=12, latency=21ms, outflow=12
  node-4: queue=15, capacity=3, incoming=47, latency=22ms, outflow=45""",
        "assistant": 'Node node-1 is FAILED and node-0 is still sending 50% traffic to it, rerouting away from node-1 to save that wasted traffic. {"action_type": "REROUTE_TRAFFIC", "target_node_id": "node-1", "parameter": 0.8}',
    })

    # Expert: SHED_LOAD on non-critical node (task-3 surge)
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": """Task: task-3  Step: 35/60  Reward: 0.250  SLA violations: 4

Node states:
  node-0 (VIP): queue=10, capacity=3, incoming=30, latency=20ms, outflow=30
  node-1: queue=45, capacity=3, incoming=75, latency=22ms, outflow=45
  node-2: queue=160, capacity=3, incoming=75, latency=55ms, outflow=45
  node-3: queue=140, capacity=3, incoming=45, latency=48ms, outflow=45
  node-4: queue=5, capacity=3, incoming=30, latency=20ms, outflow=30""",
        "assistant": 'Node node-3 (inventory) is non-critical and overloaded at queue=140, shedding 40% of its traffic to prevent cascade while scaling handles the surge upstream. {"action_type": "SHED_LOAD", "target_node_id": "node-3", "parameter": 0.4}',
    })

    examples.append({
        "system": SYSTEM_PROMPT,
        "user": """Task: task-3  Step: 38/60  Reward: 0.220  SLA violations: 6

Node states:
  node-0 (VIP): queue=8, capacity=3, incoming=30, latency=20ms, outflow=30
  node-1: queue=50, capacity=5, incoming=80, latency=21ms, outflow=75
  node-2: queue=170, capacity=4, incoming=80, latency=62ms, outflow=60
  node-3: queue=155, capacity=3, incoming=60, latency=52ms, outflow=45
  node-4: queue=130, capacity=3, incoming=75, latency=35ms, outflow=45""",
        "assistant": 'Node node-4 (auth) is independent and overloaded at queue=130, shedding 35% of its incoming traffic since it is not on the critical DAG path. {"action_type": "SHED_LOAD", "target_node_id": "node-4", "parameter": 0.35}',
    })

    # Expert: SCALE_DOWN on idle node
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": """Task: task-1  Step: 80/100  Reward: 0.650  SLA violations: 0

Node states:
  node-0 (VIP): queue=5, capacity=5, incoming=35, latency=20ms, outflow=35
  node-1: queue=3, capacity=4, incoming=18, latency=20ms, outflow=18
  node-2: queue=4, capacity=5, incoming=18, latency=20ms, outflow=18
  node-3: queue=2, capacity=4, incoming=9, latency=20ms, outflow=9
  node-4: queue=3, capacity=6, incoming=35, latency=20ms, outflow=35""",
        "assistant": 'Cluster is very stable with all queues under 10, scaling down node-4 which has 6 capacity units but only needs 3 for its traffic. {"action_type": "SCALE_DOWN", "target_node_id": "node-4", "parameter": 0.4}',
    })

    # Expert: DAG bottleneck — scale downstream, not ingress
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": """Task: task-1  Step: 45/100  Reward: 0.350  SLA violations: 2

Node states:
  node-0 (VIP): queue=30, capacity=3, incoming=44, latency=21ms, outflow=42
  node-1: queue=8, capacity=3, incoming=22, latency=20ms, outflow=22
  node-2: queue=185, capacity=3, incoming=22, latency=75ms, outflow=22
  node-3: queue=170, capacity=3, incoming=22, latency=65ms, outflow=22
  node-4: queue=10, capacity=3, incoming=43, latency=20ms, outflow=43""",
        "assistant": 'The bottleneck is downstream at node-2 (queue=185) and node-3 (queue=170), not at ingress node-0. Backpressure from node-2/3 is throttling node-0. Scaling up node-2 to break the backpressure chain. {"action_type": "SCALE_UP", "target_node_id": "node-2", "parameter": 0.6}',
    })

    # Expert: NO_OP on healthy cluster
    examples.append({
        "system": SYSTEM_PROMPT,
        "user": """Task: task-1  Step: 10/100  Reward: 0.700  SLA violations: 0

Node states:
  node-0 (VIP): queue=8, capacity=3, incoming=42, latency=20ms, outflow=42
  node-1: queue=5, capacity=3, incoming=21, latency=20ms, outflow=21
  node-2: queue=6, capacity=3, incoming=21, latency=20ms, outflow=21
  node-3: queue=3, capacity=3, incoming=10, latency=20ms, outflow=10
  node-4: queue=7, capacity=3, incoming=42, latency=20ms, outflow=42""",
        "assistant": 'All nodes are healthy with low queues and normal latency, no intervention needed. {"action_type": "NO_OP", "target_node_id": "node-0", "parameter": 0.0}',
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
                action_type, target_node_id, parameter = heuristic_action(obs_dict, task_id)
                reasoning = generate_reasoning(action_type, target_node_id, parameter, obs_dict, task_id)

                action_json = json.dumps({
                    "action_type": action_type.value,
                    "target_node_id": target_node_id,
                    "parameter": round(float(parameter), 4),
                })
                assistant_text = f"{reasoning} {action_json}"

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
        flag = " <-- UNDERREPRESENTED" if pct < 3 and at != "NO_OP" else ""
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
    from unsloth import FastLanguageModel

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
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9
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
            record["messages"], tokenize=False, add_generation_prompt=False
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
    from unsloth import FastLanguageModel
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
            messages, tokenize=False, add_generation_prompt=True
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
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
