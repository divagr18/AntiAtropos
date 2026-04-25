"""
openenv_loop.py — Environment interaction via OpenEnv HTTP API.

Handles:
  - env_reset / env_step HTTP calls to the AntiAtropos HF Space
  - Model-guided rollouts (generate action, step env, collect reward)
  - Heuristic baseline rollouts (for comparison)
  - Observation formatting for the LLM

Everything goes through the HTTP API — no local simulator imports needed.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch


# ────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────

class ActionType(str, Enum):
    NO_OP = "NO_OP"
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    REROUTE_TRAFFIC = "REROUTE_TRAFFIC"
    SHED_LOAD = "SHED_LOAD"


VALID_ACTIONS = [a.value for a in ActionType]
VALID_NODES = ["node-0", "node-1", "node-2", "node-3", "node-4"]

TASK_BRIEFS = {
    "task-1": "Traffic ramps linearly every tick. Scale up proactively — new capacity takes 5 ticks to boot. Keep latency under SLA (200ms) while minimizing cost. Scale down when queues are safe.",
    "task-2": "One node (node-1 through node-4) will fail permanently. Wait until you SEE a FAILED node — do NOT pre-scale. Once a node shows status=FAILED: reroute traffic FROM the failed node to healthy peers, and scale up any starved children. Do NOT scale node-0 unless node-4 failed independently. SCALE_DOWN cancels pending boots and reduces cost. If reward is falling, stop scaling.",
    "task-3": "A surge (~75 req/tick) will hit node-1 and node-2 via a side channel bypassing node-0. Do NOT scale node-0 — it is NOT affected. ONLY scale node-1 or node-2 when their queue_depth rises. Do NOT pre-scale. 3-4 SCALE_UPs on each is sufficient. SCALE_DOWN cancels pending boots and reduces cost — use it when queues are safe. If reward is falling, STOP scaling and SCALE_DOWN to recover.",
}

SYSTEM_PROMPT = """You are an autonomous SRE controller managing a five-node microservice cluster.

CRITICAL: You are running in NO-THINK mode (/no_think). DO NOT output `</think>` or ` 
` tags. DO NOT generate reasoning blocks. DO NOT use  
 
 or . Output ONLY your action directly as plain text.

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
}"""


# ────────────────────────────────────────────────
# HTTP Client
# ────────────────────────────────────────────────

class OpenEnvClient:
    """HTTP client for the AntiAtropos OpenEnv environment."""

    def __init__(self, env_url: str):
        self.env_url = env_url.rstrip("/")
        self._session = requests.Session()
        self._session.mount("https://", requests.adapters.HTTPAdapter(
            pool_maxsize=1, max_retries=3
        ))

    def reset(self, task_id: str = "task-1",
              seed: Optional[int] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        resp = self._session.post(
            f"{self.env_url}/reset", json=payload, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action_type: str, target_node_id: str,
             parameter: float) -> Dict[str, Any]:
        payload = {
            "action": {
                "action_type": action_type,
                "target_node_id": target_node_id,
                "parameter": parameter,
            }
        }
        resp = self._session.post(
            f"{self.env_url}/step", json=payload, timeout=30
        )
        resp.raise_for_status()
        return resp.json()

    def verify(self) -> bool:
        """Smoke-test connectivity. Returns True if OK."""
        try:
            r = self.reset("task-1", seed=0)
            obs = r.get("observation", r)
            step_r = self.step("NO_OP", "node-0", 0.0)
            print(f"[openenv] Connectivity OK — "
                  f"task_id={obs.get('task_id')}, reward={step_r.get('reward')}")
            return True
        except Exception as e:
            print(f"[openenv] Connectivity FAILED: {e}")
            return False


# ────────────────────────────────────────────────
# Observation Formatting
# ────────────────────────────────────────────────

def format_observation(obs_dict: Dict, task_id: str, step: int,
                       max_steps: int, reward: float = 0.0,
                       sla_violations: int = 0) -> str:
    """Convert API observation dict to user prompt aligned with inference.py."""
    import textwrap
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


# ────────────────────────────────────────────────
# Action Parsing
# ────────────────────────────────────────────────

@dataclass
class ParsedAction:
    action_type: str
    target_node_id: str
    parameter: float
    raw_text: str = ""
    is_valid: bool = True
    parse_error: str = ""


def parse_action(text: str) -> ParsedAction:
    """Extract action from model output text."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            return ParsedAction("NO_OP", "node-0", 0.0, text,
                                False, "no JSON found")

        obj = json.loads(text[start:end + 1])
        at = str(obj.get("action_type", "")).upper()
        nid = str(obj.get("target_node_id", "") or "node-0")
        param = float(obj.get("parameter") or 0.0)

        if at not in VALID_ACTIONS:
            return ParsedAction("NO_OP", "node-0", 0.0, text,
                                False, f"invalid action_type: {at}")
        if nid not in VALID_NODES:
            return ParsedAction("NO_OP", "node-0", 0.0, text,
                                False, f"invalid target_node_id: {nid}")

        return ParsedAction(at, nid, param, text, True, "")
    except Exception as e:
        return ParsedAction("NO_OP", "node-0", 0.0, text, False, str(e))


# ────────────────────────────────────────────────
# Rollout Data
# ────────────────────────────────────────────────

@dataclass
class Transition:
    """Single step in an episode rollout."""
    obs_text: str              # Formatted observation (LLM input)
    input_ids: Any             # Tokenized input IDs (tensor)
    attention_mask: Any        # Tokenized attention mask (tensor)
    action: ParsedAction       # The action taken
    reward: float              # Reward from environment
    log_prob: float = 0.0     # Log probability of action under policy


@dataclass
class Episode:
    """Complete episode rollout."""
    task_id: str
    transitions: List[Transition] = field(default_factory=list)
    total_reward: float = 0.0
    avg_reward: float = 0.0
    num_invalid: int = 0
    done: bool = False

    def finalize(self) -> None:
        if self.transitions:
            self.total_reward = sum(t.reward for t in self.transitions)
            self.avg_reward = self.total_reward / len(self.transitions)


# ────────────────────────────────────────────────
# Model-Guided Rollout
# ────────────────────────────────────────────────

def rollout_episode(
    client: OpenEnvClient,
    model,
    tokenizer,
    task_id: str,
    max_steps: int,
    cfg: Dict[str, Any],
    seed: Optional[int] = None,
) -> Episode:
    """Run one episode using the model to generate actions.

    The model generates text → we parse the JSON action → step the env →
    collect the reward. We also compute log_probs for REINFORCE.
    """
    episode = Episode(task_id=task_id)

    # Reset environment
    reset_resp = client.reset(task_id=task_id, seed=seed)
    obs_dict = reset_resp.get("observation", reset_resp)
    episode_reward = 0.0
    sla_violations = obs_dict.get("sla_violations", 0)

    # Generation config
    max_new_tokens = cfg.get("generation_max_new_tokens", 80)
    temperature = cfg.get("generation_temperature", 0.7)
    top_p = cfg.get("generation_top_p", 0.9)
    do_sample = cfg.get("generation_do_sample", True)

    for step in range(1, max_steps + 1):
        # Format observation for the LLM
        obs_text = format_observation(
            obs_dict, task_id, step, max_steps,
            episode_reward, sla_violations
        )

        # Build chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": obs_text},
        ]

        # Tokenize
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(
            outputs[0][input_len:], skip_special_tokens=True
        )

        # Strip TRACE
        generated_text = re.sub(
            '\x3cthink\x3e.*?\x3c/think\x3e', '',
            generated_text, flags=re.DOTALL
        ).strip()

        # Parse action
        action = parse_action(generated_text)

        # Compute log_prob for the generated tokens (for REINFORCE)
        # We'll compute this properly in the training loop using the
        # full sequence. For now, store the generated token IDs.
        # The train.py will compute log_probs during the loss step.
        generated_ids = outputs[0][input_len:]

        # Step environment (even if parse failed — NO_OP fallback)
        step_resp = client.step(
            action.action_type, action.target_node_id, action.parameter
        )
        obs_dict = step_resp.get("observation", step_resp)
        step_reward = step_resp.get("reward", 0.0)
        episode_reward = step_reward
        done = step_resp.get("done", False)
        sla_violations = obs_dict.get("sla_violations", sla_violations)

        # Record transition
        transition = Transition(
            obs_text=obs_text,
            input_ids=inputs["input_ids"].squeeze(0),
            attention_mask=inputs["attention_mask"].squeeze(0),
            action=action,
            reward=step_reward,
        )
        episode.transitions.append(transition)

        if not action.is_valid:
            episode.num_invalid += 1

        if done:
            episode.done = True
            break

    episode.finalize()
    return episode


# ────────────────────────────────────────────────
# Heuristic Baseline
# ────────────────────────────────────────────────

CRITICAL_NODES = {"node-0", "node-1", "node-2"}


def heuristic_action(obs_dict: Dict, task_id: str, step: int = 0,
                    max_steps: int = 60,
                    episode_reward: float = 0.0) -> Tuple[str, str, float]:
    """Task-aware, reward-aware heuristic with balanced action distribution."""
    nodes = obs_dict.get("nodes", [])
    if not nodes:
        return "NO_OP", "node-0", 0.0
    node_map = {n["node_id"]: n for n in nodes}

    total_queue = sum(n["queue_depth"] * 200 for n in nodes)
    avg_latency = sum(n["latency_ms"] for n in nodes) / len(nodes)
    failed_nodes = [n for n in nodes if n.get("status") == "FAILED"]
    degraded_nodes = [n for n in nodes if n.get("status") == "DEGRADED"]

    progress = step / max_steps if max_steps > 0 else 0
    early = progress < 0.15
    late = progress > 0.65

    # ── TASK-2: Fault tolerance ──
    if task_id == "task-2":
        if failed_nodes:
            fn = failed_nodes[0]
            starved_children = [
                n for n in nodes
                if n.get("status") == "DEGRADED" and n["node_id"] not in CRITICAL_NODES
            ]
            if starved_children and step % 3 != 0:
                target = max(starved_children, key=lambda n: n["queue_depth"])
                return "SCALE_UP", target["node_id"], 0.5
            return "REROUTE_TRAFFIC", fn["node_id"], 0.7

        if episode_reward > 0.5 and avg_latency < 0.04:
            non_vips = [n for n in nodes
                        if not n.get("is_vip", False) and n.get("status") != "FAILED"]
            overprov = [n for n in non_vips if n.get("capacity", 0) > 0.7]
            if overprov:
                target = max(overprov, key=lambda n: n.get("capacity", 0))
                return "SCALE_DOWN", target["node_id"], 0.3
            return "NO_OP", "node-0", 0.0

        if avg_latency > 0.04 or total_queue > 100:
            downstream = [n for n in nodes
                          if n["node_id"] != "node-0" and n.get("status") != "FAILED"]
            if downstream:
                target = max(downstream, key=lambda n: (
                    n.get("status") == "DEGRADED", n["queue_depth"]))
                return "SCALE_UP", target["node_id"], 0.4

        return "NO_OP", "node-0", 0.0

    # ── TASK-3: Surge on node-1/2 ──
    if task_id == "task-3":
        n1 = node_map.get("node-1", {})
        n2 = node_map.get("node-2", {})
        n3 = node_map.get("node-3", {})
        n4 = node_map.get("node-4", {})

        if n1.get("queue_depth", 0) > 0.3:
            param = 0.6 if n1["queue_depth"] > 0.7 else 0.4
            return "SCALE_UP", "node-1", param
        if n2.get("queue_depth", 0) > 0.3:
            param = 0.6 if n2["queue_depth"] > 0.7 else 0.4
            return "SCALE_UP", "node-2", param

        for nid, nd in [("node-3", n3), ("node-4", n4)]:
            if nd.get("queue_depth", 0) > 0.5 and nd.get("status") != "FAILED":
                return "SHED_LOAD", nid, 0.4

        if avg_latency < 0.04 and total_queue < 80:
            for nid in ["node-1", "node-2"]:
                n = node_map.get(nid, {})
                if n.get("capacity", 0) > 0.8:
                    return "SCALE_DOWN", nid, 0.3

        if episode_reward > 0.5 or (avg_latency < 0.04 and total_queue < 80):
            return "NO_OP", "node-0", 0.0

        if total_queue > 60:
            for nid in ["node-1", "node-2"]:
                n = node_map.get(nid, {})
                if n.get("queue_depth", 0) > 0.15 and n.get("status") != "FAILED":
                    return "SCALE_UP", nid, 0.3

        return "NO_OP", "node-0", 0.0

    # ── TASK-1: Traffic ramp ──
    if early and avg_latency < 0.03 and total_queue < 60:
        return "NO_OP", "node-0", 0.0

    if episode_reward > 0.55 and avg_latency < 0.04 and total_queue < 100:
        non_vips = [n for n in nodes
                    if not n.get("is_vip", False) and n.get("status") != "FAILED"]
        overprov = [n for n in non_vips if n.get("capacity", 0) > 0.7]
        if overprov and total_queue < 60:
            target = max(overprov, key=lambda n: n.get("capacity", 0))
            return "SCALE_DOWN", target["node_id"], 0.3
        return "NO_OP", "node-0", 0.0

    if late and avg_latency < 0.035 and total_queue < 80:
        non_vips = [n for n in nodes
                    if not n.get("is_vip", False) and n.get("status") != "FAILED"]
        overprov = [n for n in non_vips if n.get("capacity", 0) > 0.7]
        if overprov:
            target = max(overprov, key=lambda n: n.get("capacity", 0))
            return "SCALE_DOWN", target["node_id"], 0.3
        return "NO_OP", "node-0", 0.0

    non_critical_overloaded = [
        n for n in nodes
        if n["queue_depth"] > 0.5 and n["node_id"] not in CRITICAL_NODES
        and n.get("status") != "FAILED"
    ]
    if non_critical_overloaded and avg_latency > 0.05:
        target = non_critical_overloaded[0]
        return "SHED_LOAD", target["node_id"], 0.4

    if avg_latency > 0.04 or total_queue > 100:
        downstream = [n for n in nodes
                      if n["node_id"] != "node-0" and n.get("status") != "FAILED"]
        if downstream:
            target = max(downstream, key=lambda n: (
                n.get("status") == "DEGRADED", n["queue_depth"]))
        else:
            target = node_map.get("node-0", nodes[0])
        param = 0.6 if target["queue_depth"] > 0.75 else 0.4
        return "SCALE_UP", target["node_id"], param

    return "NO_OP", "node-0", 0.0


def rollout_heuristic_episode(
    client: OpenEnvClient,
    task_id: str,
    max_steps: int,
    seed: Optional[int] = None,
) -> Episode:
    """Run one episode using the heuristic baseline."""
    episode = Episode(task_id=task_id)

    reset_resp = client.reset(task_id=task_id, seed=seed)
    obs_dict = reset_resp.get("observation", reset_resp)
    episode_reward = 0.0

    for step in range(1, max_steps + 1):
        action_type, target_node_id, parameter = heuristic_action(
            obs_dict, task_id, step=step, max_steps=max_steps,
            episode_reward=episode_reward,
        )
        step_resp = client.step(action_type, target_node_id, parameter)
        obs_dict = step_resp.get("observation", step_resp)
        step_reward = step_resp.get("reward", 0.0)
        episode_reward = step_reward
        done = step_resp.get("done", False)

        action = ParsedAction(action_type, target_node_id, parameter)
        episode.transitions.append(Transition(
            obs_text="", input_ids=None, attention_mask=None,
            action=action, reward=step_reward,
        ))

        if done:
            episode.done = True
            break

    episode.finalize()
    return episode
