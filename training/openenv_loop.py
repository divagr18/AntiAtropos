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
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
import torch

try:
    from .chat_utils import render_no_think_chat, tokenize_text_only
except ImportError:
    from chat_utils import render_no_think_chat, tokenize_text_only


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
CRITICAL_NODES = {"node-0", "node-1", "node-2"}

TASK_BRIEFS = {
    "task-1": (
        "Traffic ramps linearly — queues build progressively on node-1, node-2, and node-3. "
        "Your job: keep latency under SLA while controlling cost. "
        "Relevant actions: SCALE_UP stressed downstream nodes before queues overflow; "
        "SCALE_DOWN nodes that are over-provisioned after the ramp stabilises; NO_OP when healthy. "
        "Boot delay is 5 ticks — act before queues peak, not after. "
        "All three downstream nodes (node-1, node-2, node-3) may need attention at different times. "
        "node-0 is rarely the bottleneck here."
    ),
    "task-2": (
        "One of node-1..node-4 will permanently FAIL mid-episode. "
        "Your job: reroute traffic away from the failed node and stabilise its downstream neighbours. "
        "Relevant actions: REROUTE_TRAFFIC from the failed node; SCALE_UP starved children; "
        "SCALE_DOWN where excess capacity accumulates post-reroute; NO_OP once stable. "
        "The failed node itself cannot be scaled — reroute away from it. "
        "REROUTE parameter controls what fraction of traffic is redirected — higher is more aggressive. "
        "Do not pre-scale before a failure is observed."
    ),
    "task-3": (
        "A traffic surge hits node-1 and node-2 via a side-channel bypassing node-0. "
        "node-0 is not affected — focus on node-1, node-2, and their downstream node-3. "
        "Relevant actions: SCALE_UP the surging nodes; SHED_LOAD on overloaded non-critical nodes; "
        "SCALE_DOWN when queues recover to reclaim cost; NO_OP when stable. "
        "Over-scaling wastes cost and reduces reward — scale proportionally to pressure. "
        "node-4 may also become overloaded from auth traffic side-effects."
    ),
}

SYSTEM_PROMPT = """You are an autonomous SRE controller for a five-node microservice cluster.

OUTPUT: One JSON object only. No thinking tags. No explanation.

━━━ TOPOLOGY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  node-0 (VIP Ingress) → node-1 (Checkout), node-2 (Catalog)
  node-2 → node-3 (Database)
  node-4 (Auth) — independent ingress
  Backpressure: overloaded children throttle their parent's outflow.

━━━ OBSERVATION KEYS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  fn=failed_nodes  dn=degraded_nodes  al=avg_latency  er=error_rate
  qb=queue_backlog  co=cost/hr  sv=sla_violations
  per node: n=id  s=H/D/F  q=queue_depth  l=latency  r=inflow  c=capacity  pc=pending  o=outflow

━━━ ACTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  SCALE_UP        target=stressed node    param=0.3-0.8  (adds capacity; 5-tick boot delay)
  SCALE_DOWN      target=idle node        param=0.2-0.5  (cancels pending first, then reduces)
  REROUTE_TRAFFIC target=failed/hot node  param=0.5-1.0  (redirects traffic to healthy peers)
  SHED_LOAD       target=non-critical     param=0.3-0.6  (drops traffic fraction; NEVER node-0)
  NO_OP           target=node-0           param=0.0      (hold position)

━━━ NODE ROLES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  node-0: VIP ingress — almost never the scaling target; target it only for REROUTE if it FAILs
  node-1: Checkout — downstream; scales independently from node-0
  node-2: Catalog  — downstream; feeds node-3; failure starves the database
  node-3: Database — leaf node; can become bottleneck under node-2 backpressure
  node-4: Auth     — independent; failure affects auth traffic separately from checkout/catalog

━━━ PARAMETER SCALE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Low (0.2-0.3): mild adjustment — conservative scaling or light reroute
  Mid (0.4-0.6): meaningful intervention — moderate queue pressure or partial reroute
  High (0.7-1.0): aggressive — severe overload, FAILED node, or emergency reroute

━━━ PRIORITIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Respond to the most urgent signal first:
  failed nodes > degraded nodes > rising latency > excess capacity > stable

━━━ ANTI-PATTERNS (reduce reward) ━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✗ Scaling node-0 when downstream nodes are the actual bottleneck
  ✗ Repeating NO_OP while queues or latency are visibly rising
  ✗ Using parameter=0.0 for SCALE_UP or REROUTE_TRAFFIC (has no effect)
  ✗ Same action on the same node many steps in a row with no improvement

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{"action_type":"SCALE_UP","target_node_id":"node-2","parameter":0.5}

action_type MUST be one of: SCALE_UP SCALE_DOWN REROUTE_TRAFFIC SHED_LOAD NO_OP
Choose the action that best addresses the current cluster state."""


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
              seed: Optional[int] = None,
              mode: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"task_id": task_id}
        if mode is not None:
            payload["mode"] = mode
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

    # Build compact observation dict (trimmed for speed: 40% fewer tokens)
    nodes_data = []
    for n in obs_dict.get("nodes", []):
        nodes_data.append({
            "n": n.get("node_id"),
            "s": n.get("status", "HEALTHY")[0],  # H/D/F = Healthy/Degraded/Failed
            "q": n.get("queue_depth", 0),
            "l": n.get("latency_ms", 0),
            "r": n.get("incoming_request_rate", 0),
            "c": n.get("capacity", 0),
            "pc": n.get("pending_capacity", 0),
            "o": n.get("outflow_rate", 0),
        })
    
    obs_compact = {
        "t": task_id,
        "st": step,
        "mx": max_steps,
        "fn": [n["node_id"] for n in obs_dict.get("nodes", []) if n.get("status") == "FAILED"],
        "dn": [n["node_id"] for n in obs_dict.get("nodes", []) if n.get("status") == "DEGRADED"],
        "al": obs_dict.get("average_latency_ms", 0),
        "er": obs_dict.get("error_rate", 0),
        "qb": obs_dict.get("total_queue_backlog", 0),
        "co": obs_dict.get("current_cost_per_hour", 0),
        "sv": sla_violations,
        "nd": nodes_data,
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


def repair_action(action_type: str, target_node_id: str, parameter: float) -> Tuple[str, str, float, str]:
    """Normalize generated JSON so the environment validator accepts it."""
    at = str(action_type).upper()
    nid = str(target_node_id or "node-0")

    if at not in VALID_ACTIONS or nid not in VALID_NODES:
        return "NO_OP", "node-0", 0.0, "invalid action schema"

    try:
        param = float(parameter)
    except (TypeError, ValueError):
        param = 0.0

    if not math.isfinite(param):
        param = 0.0

    repair_notes = []

    if at == "NO_OP":
        return at, "node-0", 0.0, ""

    if at in {"REROUTE_TRAFFIC", "SHED_LOAD"}:
        clamped = min(1.0, max(0.0, param))
        if clamped != param:
            repair_notes.append(f"clamped {at} parameter to [0,1]")
        param = clamped

    if at in {"SCALE_UP", "SCALE_DOWN"}:
        clamped = min(10.0, max(0.0, param))
        if clamped != param:
            repair_notes.append(f"clamped {at} parameter to [0,10]")
        param = clamped

    if at == "SHED_LOAD" and nid in CRITICAL_NODES:
        at = "SCALE_UP"
        param = min(0.8, max(0.3, param or 0.4))
        repair_notes.append("rewrote critical-node SHED_LOAD to SCALE_UP")

    return at, nid, round(float(param), 4), "; ".join(repair_notes)


def parse_action(text: str) -> ParsedAction:
    """Extract action from model output text.

    Uses raw_decode so that extra content after the first JSON object
    (e.g. duplicate actions, trailing text) is silently ignored.
    """
    try:
        start = text.find("{")
        if start == -1:
            return ParsedAction("NO_OP", "node-0", 0.0, text,
                                False, "no JSON found")

        # Decode only the first complete JSON value (ignore extra data)
        decoder = json.JSONDecoder()
        obj, end_pos = decoder.raw_decode(text, start)

        at_raw = obj.get("action_type", "") or ""
        at = str(at_raw).strip().upper()
        if not at:
            return ParsedAction("NO_OP", "node-0", 0.0, text,
                                False, "invalid action_type: (empty)")
        nid = str(obj.get("target_node_id", "") or "node-0")
        param = float(obj.get("parameter") or 0.0)

        if at not in VALID_ACTIONS:
            return ParsedAction("NO_OP", "node-0", 0.0, text,
                                False, f"invalid action_type: {at}")
        if nid not in VALID_NODES:
            return ParsedAction("NO_OP", "node-0", 0.0, text,
                                False, f"invalid target_node_id: {nid}")

        at, nid, param, repair_note = repair_action(at, nid, param)
        extracted = text[start:end_pos]
        return ParsedAction(at, nid, param, extracted, True, repair_note)
    except json.JSONDecodeError as e:
        return ParsedAction("NO_OP", "node-0", 0.0, text, False, str(e))
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
    env_mode = cfg.get("env_mode", "simulated")
    reset_resp = client.reset(task_id=task_id, seed=seed, mode=env_mode)
    obs_dict = reset_resp.get("observation", reset_resp)
    episode_reward = 0.0
    sla_violations = obs_dict.get("sla_violations", 0)

    # Generation config (reduced for speed)
    max_new_tokens = cfg.get("generation_max_new_tokens", 50)
    temperature = cfg.get("generation_temperature", 0.85)  # 0.85 > 0.7: more exploration
    top_p = cfg.get("generation_top_p", 0.9)
    do_sample = cfg.get("generation_do_sample", True)
    invalid_penalty = cfg.get("invalid_action_penalty", 0.15)  # reward penalty for empty/bad JSON

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

        # Render via the Qwen Jinja template with thinking disabled, then
        # tokenize explicitly as text so Qwen-VL processors do not load images.
        input_text = render_no_think_chat(
            tokenizer, messages, add_generation_prompt=True
        )
        inputs = tokenize_text_only(tokenizer, input_text, model.device)
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

        # Invalid action penalty — teaches the model that malformed JSON hurts.
        # Without this, the model gets full env reward even for empty action_type
        # (which falls back to NO_OP), so it never learns to generate valid JSON.
        if not action.is_valid:
            penalty = invalid_penalty if "empty" in action.parse_error else invalid_penalty * 0.5
            step_reward = max(0.0, step_reward - penalty)

        episode_reward = step_reward
        done = step_resp.get("done", False)
        sla_violations = obs_dict.get("sla_violations", sla_violations)

        # Per-step log
        if not action.is_valid:
            notes = f"INVALID: {action.parse_error}"
        elif action.parse_error:
            notes = action.parse_error
        else:
            notes = ""
        action_str = f"{action.action_type:11s} {action.target_node_id} p={action.parameter:.2f}"
        print(f"  S{step:2d}  | {action_str:30s} | {step_reward:.4f}  | {notes}", flush=True)

        # Record transition (with adjusted reward)
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
# Batch Rollout (Parallel Episodes)
# ────────────────────────────────────────────────

# Thread-local storage for per-thread HTTP sessions (requests.Session is not thread-safe)
_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    """Get or create a requests.Session for the current thread."""
    if not hasattr(_thread_local, 'session'):
        _thread_local.session = requests.Session()
        _thread_local.session.mount("http://", requests.adapters.HTTPAdapter(
            pool_maxsize=4, max_retries=2
        ))
        _thread_local.session.mount("https://", requests.adapters.HTTPAdapter(
            pool_maxsize=4, max_retries=2
        ))
    return _thread_local.session


def _threaded_reset(env_url: str, task_id: str, seed: int, mode: str) -> Dict[str, Any]:
    """Reset environment from a thread pool worker."""
    session = _get_thread_session()
    payload: Dict[str, Any] = {"task_id": task_id}
    if mode is not None:
        payload["mode"] = mode
    if seed is not None:
        payload["seed"] = seed
    resp = session.post(f"{env_url}/reset", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _threaded_step(env_url: str, action_type: str, target_node_id: str,
                   parameter: float) -> Dict[str, Any]:
    """Step environment from a thread pool worker."""
    session = _get_thread_session()
    payload = {
        "action": {
            "action_type": action_type,
            "target_node_id": target_node_id,
            "parameter": parameter,
        }
    }
    resp = session.post(f"{env_url}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def rollout_batch(
    env_url: str,
    model,
    tokenizer,
    task_ids: List[str],
    max_steps: int,
    cfg: Dict[str, Any],
    seeds: List[int],
) -> List[Episode]:
    """Run multiple episodes in parallel with batched generation.

    Instead of running 12 episodes sequentially (each step = 1 GPU forward pass),
    we run them in lockstep: at each step, all active episodes' observations are
    batched into a single forward pass, and env step HTTP calls are parallelized
    via ThreadPoolExecutor.

    This reduces 480 forward passes per iteration → 40, and 480 HTTP calls → 40
    parallel batches. ~10x speedup on generation, ~10x on env steps.
    """
    num_episodes = len(task_ids)
    env_mode = cfg.get("env_mode", "simulated")
    max_new_tokens = cfg.get("generation_max_new_tokens", 50)
    temperature = cfg.get("generation_temperature", 0.7)
    top_p = cfg.get("generation_top_p", 0.9)
    do_sample = cfg.get("generation_do_sample", True)

    env_url = env_url.rstrip("/")

    # ── Reset all episodes in parallel ──
    with ThreadPoolExecutor(max_workers=num_episodes) as pool:
        reset_futures = {
            pool.submit(_threaded_reset, env_url, task_ids[i], seeds[i], env_mode): i
            for i in range(num_episodes)
        }
        reset_results = [None] * num_episodes
        for future in as_completed(reset_futures):
            idx = reset_futures[future]
            try:
                reset_results[idx] = future.result()
            except Exception as e:
                print(f"  [batch] Episode {idx} reset failed: {e}")
                reset_results[idx] = None

    # Initialize episode state
    episodes = [Episode(task_id=task_ids[i]) for i in range(num_episodes)]
    obs_dicts: List[Dict] = [{}] * num_episodes
    episode_rewards = [0.0] * num_episodes
    sla_violations_list = [0] * num_episodes
    active = [True] * num_episodes

    for i in range(num_episodes):
        if reset_results[i] is not None:
            obs = reset_results[i].get("observation", reset_results[i])
            obs_dicts[i] = obs
            sla_violations_list[i] = obs.get("sla_violations", 0)
        else:
            active[i] = False

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    # ── Main loop: step all active episodes in lockstep ──
    for step in range(1, max_steps + 1):
        active_indices = [i for i in range(num_episodes) if active[i]]
        if not active_indices:
            break

        # ── Format observations and tokenize ──
        all_input_ids = []
        all_attention_masks = []
        all_obs_texts = []

        for i in active_indices:
            obs_text = format_observation(
                obs_dicts[i], task_ids[i], step, max_steps,
                episode_rewards[i], sla_violations_list[i]
            )
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": obs_text},
            ]
            input_text = render_no_think_chat(
                tokenizer, messages, add_generation_prompt=True
            )
            inputs = tokenize_text_only(tokenizer, input_text, model.device)

            all_input_ids.append(inputs["input_ids"].squeeze(0))
            all_attention_masks.append(inputs["attention_mask"].squeeze(0))
            all_obs_texts.append(obs_text)

        # ── Left-pad to same length for batch generation ──
        max_len = max(ids.shape[0] for ids in all_input_ids)
        padded_ids = []
        padded_masks = []
        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded_ids.append(torch.cat([
                    torch.full((pad_len,), pad_id, device=model.device), ids
                ]))
                padded_masks.append(torch.cat([
                    torch.zeros(pad_len, device=model.device, dtype=mask.dtype), mask
                ]))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)

        batch_input_ids = torch.stack(padded_ids)
        batch_attention_mask = torch.stack(padded_masks)
        input_lens = [ids.shape[0] for ids in all_input_ids]  # Before padding

        # ── Batch generate (single forward pass for all episodes) ──
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=pad_id,
            )

        # ── Parse actions ──
        # Free the generation output tensor immediately — it holds the full
        # KV-cache for all 36 layers on GPU.
        #
        # DECODING BUG FIX: slice at `max_len` (padded length), NOT `input_lens[idx]`.
        # All sequences are left-padded to max_len, so outputs[idx] has shape
        # [max_len + num_generated_tokens]. The generated tokens always start at
        # position max_len regardless of the original (unpadded) input length.
        #
        # Using input_lens[idx] (unpadded) causes shorter-input episodes to include
        # the trailing portion of their padded input tokens as "generated" text,
        # producing garbage that parse_action can't find valid JSON in.
        # This is why invalids only appear when num_episodes > 1.
        actions = []
        decoded_texts = []
        padded_len = batch_input_ids.shape[1]  # = max_len, same for all in batch
        for idx in range(len(active_indices)):
            generated_text = tokenizer.decode(
                outputs[idx][padded_len:], skip_special_tokens=True  # ← was input_lens[idx]
            )
            decoded_texts.append(generated_text)
        del outputs  # Free KV-cache before parsing
        torch.cuda.empty_cache()

        for idx, generated_text in enumerate(decoded_texts):
            generated_text = re.sub(
                '\x3cthink\x3e.*?\x3c/think\x3e', '',
                generated_text, flags=re.DOTALL
            ).strip()
            action = parse_action(generated_text)
            actions.append(action)

        # ── Step all active environments in parallel ──
        with ThreadPoolExecutor(max_workers=len(active_indices)) as pool:
            step_futures = {
                pool.submit(
                    _threaded_step, env_url,
                    actions[idx].action_type, actions[idx].target_node_id,
                    actions[idx].parameter
                ): idx
                for idx in range(len(active_indices))
            }
            step_results = [None] * len(active_indices)
            for future in as_completed(step_futures):
                idx = step_futures[future]
                try:
                    step_results[idx] = future.result()
                except Exception as e:
                    print(f"  E{active_indices[idx]} S{step:2d} | step failed: {e}")
                    step_results[idx] = None

        # ── Process results ──
        for idx, i in enumerate(active_indices):
            if step_results[idx] is None:
                active[i] = False
                continue

            result = step_results[idx]
            obs_dicts[i] = result.get("observation", result)
            step_reward = result.get("reward", 0.0)
            episode_rewards[i] = step_reward
            done = result.get("done", False)
            sla_violations_list[i] = obs_dicts[i].get(
                "sla_violations", sla_violations_list[i]
            )

            # Record transition
            transition = Transition(
                obs_text=all_obs_texts[idx],
                input_ids=all_input_ids[idx],
                attention_mask=all_attention_masks[idx],
                action=actions[idx],
                reward=step_reward,
            )
            episodes[i].transitions.append(transition)

            if not actions[idx].is_valid:
                episodes[i].num_invalid += 1

            # Log (compact: episode+step on one line)
            action_str = (f"{actions[idx].action_type:11s} "
                         f"{actions[idx].target_node_id} "
                         f"p={actions[idx].parameter:.2f}")
            notes = ("" if actions[idx].is_valid
                     else f"INVALID: {actions[idx].parse_error}")
            print(f"  E{i} S{step:2d} | {action_str:30s} | "
                  f"{step_reward:.4f}  | {notes}", flush=True)

            if done:
                episodes[i].done = True
                active[i] = False

    for ep in episodes:
        ep.finalize()
    return episodes

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
