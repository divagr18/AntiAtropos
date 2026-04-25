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
    """Convert API observation dict to natural-language string for LLM."""
    lines = [f"Task: {task_id}  Step: {step}/{max_steps}  "
             f"Reward: {reward:.3f}  SLA violations: {sla_violations}"]
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
            messages, tokenize=False, add_generation_prompt=True
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

def heuristic_action(obs_dict: Dict, task_id: str) -> Tuple[str, str, float]:
    """Rule-based heuristic for baseline comparison."""
    nodes = obs_dict.get("nodes", [])
    total_queue = sum(n["queue_depth"] * 200 for n in nodes)
    avg_latency = sum(n["latency_ms"] for n in nodes) / len(nodes) if nodes else 0
    failed_nodes = [n for n in nodes if n.get("status") == "FAILED"]

    if failed_nodes:
        return "REROUTE_TRAFFIC", failed_nodes[0]["node_id"], 0.7

    non_critical_overloaded = [
        n for n in nodes
        if n["queue_depth"] > 0.6 and n["node_id"] != "node-0"
        and n.get("status") != "FAILED"
    ]
    if non_critical_overloaded and avg_latency > 0.05:
        shed = [n for n in non_critical_overloaded
                if n["node_id"] in ["node-3", "node-4"]]
        target = shed[0] if shed else non_critical_overloaded[0]
        return "SHED_LOAD", target["node_id"], 0.4

    if avg_latency > 0.03 or total_queue > 200:
        target = max(nodes, key=lambda n: n["queue_depth"])
        param = 0.6 if target["queue_depth"] > 0.75 else 0.4
        return "SCALE_UP", target["node_id"], param

    non_vips = [n for n in nodes if not n.get("is_vip", False)
                and n.get("status") != "FAILED"]
    if non_vips and avg_latency < 0.025 and total_queue < 50:
        overprov = [n for n in non_vips if n.get("capacity", 0) > 0.6]
        if overprov:
            target = max(overprov, key=lambda n: n.get("capacity", 0))
            return "SCALE_DOWN", target["node_id"], 0.3

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
        action_type, target_node_id, parameter = heuristic_action(obs_dict, task_id)
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
