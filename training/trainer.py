"""
AntiAtropos Training Loop.

Orchestrates episode collection, reward computation, and loss calculation
for training LLM-based SRE agents. Works with the local simulator in
pure-Python mode (no AWS/GPU needed for validation).

On Colab: Replace EpisodeCollector's "model" with a real QLoRA-backed
transformers model. The rest of the pipeline stays the same.
"""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Callable

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from simulator import ClusterSimulator, NodeStatus, COST_PER_CAPACITY_UNIT_PER_HOUR
from stability import (
    compute_lyapunov, compute_reward, compute_barrier,
    normalize_reward, smooth_sla_penalty,
)
from .losses import (
    compute_returns, compute_gae,
    reinforce_loss, reinforce_baseline_loss,
    grpo_loss, rloo_loss,
    normalize_rewards, compute_reward_stats,
)


# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════

LOSS_REINFORCE = "reinforce"
LOSS_REINFORCE_BASELINE = "reinforce_baseline"
LOSS_GRPO = "grpo"
LOSS_RLOO = "rloo"

VALID_LOSSES = {LOSS_REINFORCE, LOSS_REINFORCE_BASELINE, LOSS_GRPO, LOSS_RLOO}


@dataclass
class TrainingConfig:
    """Configuration for the SRE training loop."""

    # Episode settings
    n_nodes: int = 5
    max_steps: int = 100
    tasks: List[str] = field(default_factory=lambda: ["task-1", "task-2", "task-3"])

    # Loss function
    loss_fn: str = LOSS_REINFORCE_BASELINE  # Recommended starting point
    gamma: float = 0.99          # Discount factor
    gae_lambda: float = 0.95     # GAE lambda (only used with GAE advantages)

    # GRPO / RLOO settings
    n_samples_per_state: int = 4  # K rollouts per state for GRPO/RLOO

    # Reward normalization
    normalize_rewards: bool = True
    reward_ema_alpha: float = 0.01  # Exponential moving average update rate

    # Advantage normalization
    normalize_advantages: bool = True  # Standardize advantages (critical for SRE)

    # Logging
    log_every: int = 10  # Log every N episodes


# ════════════════════════════════════════════════════════════════════════════════
# Model Protocol (abstraction for real LLM or mock)
# ════════════════════════════════════════════════════════════════════════════════

class PolicyModel(Protocol):
    """Interface that both real LLMs and mock models must implement."""

    def get_log_prob(self, prompt: str, action_text: str) -> float:
        """Return log π(action_text | prompt) under the current policy."""
        ...

    def generate(self, prompt: str) -> str:
        """Sample an action from the current policy."""
        ...


class MockPolicyModel:
    """
    Random policy for local testing. Generates random valid actions
    and returns stochastic log probabilities.

    This is NOT for training — it's for validating the training pipeline
    (episode collection, reward computation, loss calculation) end-to-end
    before connecting a real model.

    Unlike a truly uniform policy (which would produce zero gradient with
    advantage normalization), this model returns varying log probs per
    action, simulating a real LLM that prefers some actions over others.
    This ensures the training pipeline produces non-trivial losses.
    """

    def __init__(self, n_nodes: int = 5, seed: int = 42):
        self._rng = random.Random(seed)
        self._n_nodes = n_nodes
        # Action-dependent log probs: different action types get different
        # log probs (simulating a real model that has preferences).
        # Base log prob ~ log(1/50) ≈ -3.9, with per-action noise.
        self._n_choices = 5 * n_nodes
        self._base_log_prob = math.log(1.0 / self._n_choices)

    def get_log_prob(self, prompt: str, action_text: str) -> float:
        """Return stochastic log probability (varies per action)."""
        # Add Gaussian noise to simulate a real model's varying confidence.
        # std=0.5 produces meaningful variation while staying in a plausible
        # range for LLM token log-probs.
        noise = self._rng.gauss(0, 0.5)
        return self._base_log_prob + noise

    def generate(self, prompt: str) -> str:
        """Generate a random valid action as JSON string."""
        import json
        action_types = ["SCALE_UP", "SCALE_DOWN", "REROUTE_TRAFFIC", "SHED_LOAD", "NO_OP"]
        node_id = f"node-{self._rng.randint(0, self._n_nodes - 1)}"
        action_type = self._rng.choice(action_types)
        parameter = round(self._rng.random(), 2)
        return json.dumps({
            "action_type": action_type,
            "target_node_id": node_id,
            "parameter": parameter,
        })


# ════════════════════════════════════════════════════════════════════════════════
# Observation formatting (mirrors inference.py logic)
# ════════════════════════════════════════════════════════════════════════════════

MAX_QUEUE_NORM = 200.0
MAX_LATENCY_NORM = 1000.0
MAX_REQUEST_RATE_NORM = 100.0
ALPHA, BETA, GAMMA, DELTA = 0.002, 0.01, 10.0, 0.005


def format_observation(nodes: List[dict], task_id: str, step: int, max_steps: int) -> str:
    """
    Format simulator state as a text prompt for the model.

    This mirrors inference.py's build_user_prompt and observation_for_model.
    """
    import json
    node_data = []
    for n in nodes:
        node_data.append({
            "node_id": n["node_id"],
            "status": n["status"] if isinstance(n["status"], str) else n["status"].value,
            "is_vip": n.get("is_vip", False),
            "queue_depth": min(1.0, max(0.0, n["queue_depth"] / MAX_QUEUE_NORM)),
            "latency_ms": min(1.0, max(0.0, n["latency_ms"] / MAX_LATENCY_NORM)),
            "cpu_utilization": min(1.0, max(0.0, n.get("cpu_utilization", 0.0))),
            "incoming_request_rate": min(1.0, max(0.0, n["incoming_request_rate"] / MAX_REQUEST_RATE_NORM)),
        })
    obs = {"task_id": task_id, "step": step, "max_steps": max_steps, "nodes": node_data}
    return json.dumps(obs, separators=(",", ":"))


def parse_action(action_text: str) -> dict:
    """Parse model output into an action dict."""
    import json
    try:
        data = json.loads(action_text)
        return {
            "action_type": str(data.get("action_type", "NO_OP")).upper(),
            "target_node_id": str(data.get("target_node_id", "node-0")),
            "parameter": float(data.get("parameter", 0.0)),
        }
    except (json.JSONDecodeError, ValueError):
        return {"action_type": "NO_OP", "target_node_id": "node-0", "parameter": 0.0}


# ════════════════════════════════════════════════════════════════════════════════
# Episode Collection
# ════════════════════════════════════════════════════════════════════════════════

@dataclass
class StepRecord:
    """A single step in an episode."""
    prompt: str           # Observation formatted as text
    action_text: str      # Model output (JSON string)
    log_prob: float       # log π(action | prompt)
    reward: float         # Raw reward for this step
    reward_normalized: float  # Normalized reward


@dataclass
class EpisodeRecord:
    """A complete episode trajectory."""
    task_id: str
    steps: List[StepRecord] = field(default_factory=list)
    total_reward: float = 0.0
    total_reward_normalized: float = 0.0
    avg_reward: float = 0.0
    sla_violations: int = 0
    final_lyapunov: float = 0.0


class EpisodeCollector:
    """
    Collects episodes by running the simulator with a policy model.

    This is the bridge between the simulator (physics) and the training
    pipeline (loss computation). It produces EpisodeRecords that feed
    directly into the loss functions.
    """

    def __init__(self, config: TrainingConfig):
        self._config = config
        self._sim = ClusterSimulator(n_nodes=config.n_nodes)

    def collect_episode(
        self,
        model: PolicyModel,
        task_id: str,
        seed: Optional[int] = None,
    ) -> EpisodeRecord:
        """Run one episode and collect step-level data."""
        cfg = self._config
        self._sim.reset(task_id=task_id, seed=seed)

        record = EpisodeRecord(task_id=task_id)
        prev_lyapunov = 0.0

        for step in range(1, cfg.max_steps + 1):
            # 1. Get observation
            nodes_true = self._sim.state(for_agent=False)
            nodes_obs = self._sim.state(for_agent=True)
            prompt = format_observation(nodes_obs, task_id, step, cfg.max_steps)

            # 2. Get action from model
            action_text = model.generate(prompt)
            log_prob = model.get_log_prob(prompt, action_text)

            # 3. Apply action
            action = parse_action(action_text)
            class _A:
                pass
            a = _A()
            a.action_type = action["action_type"]
            a.target_node_id = action["target_node_id"]
            a.parameter = action["parameter"]
            self._sim.apply_action(a)

            # 4. Tick
            self._sim.tick()

            # 5. Compute reward (mirrors environment.py)
            nodes_true = self._sim.state(for_agent=False)
            current_lyapunov = compute_lyapunov(nodes_true)

            # Importance-weighted average latency
            w_lat = 0.0
            w_sum = 0.0
            for n in nodes_true:
                w = n.get("importance_weight", 1.0)
                lat = MAX_LATENCY_NORM if n["status"] == NodeStatus.FAILED else n["latency_ms"]
                w_lat += w * lat
                w_sum += w
            avg_lat_norm = min(1.0, max(0.0, (w_lat / w_sum / MAX_LATENCY_NORM) if w_sum > 0 else 1.0))

            # Error rate
            total_in = sum(n.get("incoming_request_rate", 0) * n.get("importance_weight", 1.0) for n in nodes_true)
            total_drop = sum(n.get("dropped_requests", 0) * n.get("importance_weight", 1.0) for n in nodes_true)
            error_rate = min(1.0, total_drop / total_in) if total_in > 0 else 0.0

            sla_step = smooth_sla_penalty(avg_lat_norm, error_rate)
            if avg_lat_norm > 0.20 or error_rate > 0.05:
                record.sla_violations += 1

            # Cost
            total_cap = 0
            for n in nodes_true:
                if n["status"] != NodeStatus.FAILED:
                    total_cap += int(n.get("capacity_units", 0)) + int(n.get("pending_capacity_units", 0))
            cost = total_cap * COST_PER_CAPACITY_UNIT_PER_HOUR

            barrier = compute_barrier(nodes_true)
            raw_reward = compute_reward(
                prev_lyapunov, current_lyapunov, cost, sla_step,
                ALPHA, BETA, GAMMA, barrier, DELTA,
            )
            norm_reward = normalize_reward(raw_reward)

            record.steps.append(StepRecord(
                prompt=prompt,
                action_text=action_text,
                log_prob=log_prob,
                reward=raw_reward,
                reward_normalized=norm_reward,
            ))
            record.total_reward += raw_reward
            record.total_reward_normalized += norm_reward
            prev_lyapunov = current_lyapunov

        record.avg_reward = record.total_reward / max(1, len(record.steps))
        record.final_lyapunov = prev_lyapunov
        return record

    def collect_group(
        self,
        model: PolicyModel,
        task_id: str,
        k: int,
        seed: Optional[int] = None,
    ) -> List[EpisodeRecord]:
        """
        Collect K episodes from the same initial state (for GRPO/RLOO).

        Uses the same seed for all K episodes so they start from the same
        domain randomization, but different model samples produce different
        trajectories.
        """
        return [self.collect_episode(model, task_id, seed=seed) for _ in range(k)]


# ════════════════════════════════════════════════════════════════════════════════
# Trainer
# ════════════════════════════════════════════════════════════════════════════════

class SRETrainer:
    """
    Main training orchestrator for AntiAtropos SRE agents.

    Usage (local validation with MockPolicyModel):
        config = TrainingConfig(loss_fn="reinforce_baseline")
        trainer = SRETrainer(config)
        model = MockPolicyModel()
        metrics = trainer.train_step(model, task_id="task-1", seed=42)

    Usage (Colab with real model):
        config = TrainingConfig(loss_fn="grpo", n_samples_per_state=4)
        trainer = SRETrainer(config)
        model = QLoRAModel(...)  # Your transformers model
        for epoch in range(num_epochs):
            for task in config.tasks:
                metrics = trainer.train_step(model, task_id=task)
                model.update(metrics["loss"])  # Backprop
    """

    def __init__(self, config: TrainingConfig):
        assert config.loss_fn in VALID_LOSSES, f"Unknown loss: {config.loss_fn}"
        self._config = config
        self._collector = EpisodeCollector(config)
        self._running_reward_mean = 0.0
        self._running_reward_var = 1.0
        self._episode_count = 0

    def train_step(
        self,
        model: PolicyModel,
        task_id: str,
        seed: Optional[int] = None,
    ) -> dict:
        """
        Execute one training step: collect episode(s) → compute loss.

        Returns a metrics dict with:
            - loss: The computed loss value
            - avg_reward: Average raw reward across the episode
            - avg_norm_reward: Average normalized reward
            - episode_length: Number of steps
            - sla_violations: Number of SLA violations
            - final_lyapunov: Lyapunov energy at episode end
            - reward_mean/var: Running reward statistics
        """
        cfg = self._config

        if cfg.loss_fn in (LOSS_GRPO, LOSS_RLOO):
            return self._train_step_grouped(model, task_id, seed)
        else:
            return self._train_step_single(model, task_id, seed)

    def _train_step_single(
        self,
        model: PolicyModel,
        task_id: str,
        seed: Optional[int] = None,
    ) -> dict:
        """Train step for REINFORCE / REINFORCE+baseline."""
        cfg = self._config

        # 1. Collect episode
        episode = self._collector.collect_episode(model, task_id, seed=seed)

        # 2. Extract rewards and log probs
        rewards = [s.reward for s in episode.steps]
        log_probs = [s.log_prob for s in episode.steps]

        # 3. Update running reward stats
        ep_mean, ep_var = compute_reward_stats(rewards)
        self._running_reward_mean = (
            (1 - cfg.reward_ema_alpha) * self._running_reward_mean
            + cfg.reward_ema_alpha * ep_mean
        )
        self._running_reward_var = (
            (1 - cfg.reward_ema_alpha) * self._running_reward_var
            + cfg.reward_ema_alpha * ep_var
        )

        # 4. Optionally normalize rewards
        if cfg.normalize_rewards:
            rewards = normalize_rewards(
                rewards, self._running_reward_mean, self._running_reward_var
            )

        # 5. Compute returns
        returns = compute_returns(rewards, gamma=cfg.gamma)

        # 6. Compute loss
        if cfg.loss_fn == LOSS_REINFORCE:
            loss = reinforce_loss(log_probs, returns)
        elif cfg.loss_fn == LOSS_REINFORCE_BASELINE:
            # Use running mean as baseline
            baselines = [self._running_reward_mean] * len(returns)
            loss = reinforce_baseline_loss(
                log_probs, returns, baselines,
                normalize_advantage=cfg.normalize_advantages,
            )
        else:
            raise ValueError(f"Unexpected loss_fn: {cfg.loss_fn}")

        self._episode_count += 1

        return {
            "loss": loss,
            "avg_reward": episode.avg_reward,
            "avg_norm_reward": episode.total_reward_normalized / max(1, len(episode.steps)),
            "episode_length": len(episode.steps),
            "sla_violations": episode.sla_violations,
            "final_lyapunov": episode.final_lyapunov,
            "reward_mean": self._running_reward_mean,
            "reward_var": self._running_reward_var,
            "task_id": task_id,
            "episode": episode,
        }

    def _train_step_grouped(
        self,
        model: PolicyModel,
        task_id: str,
        seed: Optional[int] = None,
    ) -> dict:
        """Train step for GRPO / RLOO."""
        cfg = self._config
        k = cfg.n_samples_per_state

        # 1. Collect K episodes (same seed → same domain randomization)
        episodes = self._collector.collect_group(model, task_id, k=k, seed=seed)

        # 2. For each step position, form groups across episodes
        #    (assumes all episodes have same length)
        min_len = min(len(ep.steps) for ep in episodes)

        log_probs_groups = []
        rewards_groups = []

        for t in range(min_len):
            step_lps = []
            step_rs = []
            for ep in episodes:
                step_lps.append(ep.steps[t].log_prob)
                step_rs.append(ep.steps[t].reward)
            log_probs_groups.append(step_lps)
            rewards_groups.append(step_rs)

        # 3. Update running stats
        all_rewards = [s.reward for ep in episodes for s in ep.steps]
        ep_mean, ep_var = compute_reward_stats(all_rewards)
        self._running_reward_mean = (
            (1 - cfg.reward_ema_alpha) * self._running_reward_mean
            + cfg.reward_ema_alpha * ep_mean
        )
        self._running_reward_var = (
            (1 - cfg.reward_ema_alpha) * self._running_reward_var
            + cfg.reward_ema_alpha * ep_var
        )

        # 4. Normalize rewards
        if cfg.normalize_rewards:
            rewards_groups = [
                normalize_rewards(rs, self._running_reward_mean, self._running_reward_var)
                for rs in rewards_groups
            ]

        # 5. Compute loss
        if cfg.loss_fn == LOSS_GRPO:
            loss = grpo_loss(log_probs_groups, rewards_groups)
        elif cfg.loss_fn == LOSS_RLOO:
            loss = rloo_loss(log_probs_groups, rewards_groups)
        else:
            raise ValueError(f"Unexpected grouped loss_fn: {cfg.loss_fn}")

        # 6. Aggregate metrics across episodes
        avg_reward = sum(ep.avg_reward for ep in episodes) / len(episodes)
        avg_norm = sum(
            ep.total_reward_normalized / max(1, len(ep.steps)) for ep in episodes
        ) / len(episodes)
        total_sla = sum(ep.sla_violations for ep in episodes)
        avg_lyapunov = sum(ep.final_lyapunov for ep in episodes) / len(episodes)

        self._episode_count += k

        return {
            "loss": loss,
            "avg_reward": avg_reward,
            "avg_norm_reward": avg_norm,
            "episode_length": min_len,
            "sla_violations": total_sla,
            "final_lyapunov": avg_lyapunov,
            "reward_mean": self._running_reward_mean,
            "reward_var": self._running_reward_var,
            "task_id": task_id,
            "episodes": episodes,
        }

    def train_epoch(
        self,
        model: PolicyModel,
        seed: Optional[int] = None,
    ) -> List[dict]:
        """
        Run one training step per task in the curriculum.

        Returns a list of metrics dicts (one per task).
        """
        results = []
        for task_id in self._config.tasks:
            step_seed = seed + hash(task_id) % 1000 if seed is not None else None
            metrics = self.train_step(model, task_id, seed=step_seed)
            results.append(metrics)
            if self._episode_count % self._config.log_every == 0:
                self._log_metrics(metrics)
        return results

    def _log_metrics(self, metrics: dict) -> None:
        """Print training metrics."""
        print(
            f"[Episode {self._episode_count}] "
            f"task={metrics['task_id']} "
            f"loss={metrics['loss']:.4f} "
            f"avg_reward={metrics['avg_reward']:.4f} "
            f"avg_norm_reward={metrics['avg_norm_reward']:.4f} "
            f"sla_violations={metrics['sla_violations']} "
            f"lyapunov={metrics['final_lyapunov']:.1f} "
            f"reward_mean={metrics['reward_mean']:.4f} "
            f"reward_var={metrics['reward_var']:.4f}"
        )
