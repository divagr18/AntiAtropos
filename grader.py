# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AntiAtropos Grading Logic — Evaluation Engine.

Provides standard scoring for hackathon submissions across multiple dimensions:
1. Uptime: Fraction of ticks where cluster-wide SLA (latency/errors) was met.
2. Cost: Normalized efficiency score based on provisioned capacity.
3. Stability: Mean Lyapunov energy normalized to a target baseline.
4. Recovery Speed (task-2): Ticks from node failure to child queue recovery.
5. VIP Protection (task-3): Whether node-0 stayed healthy during surge.
6. Action Efficiency: Fraction of actions that had measurable effect.
"""

import math
from collections import Counter
from typing import Dict, Any, List, Optional

try:
    from .models import ClusterObservation
    from .simulator import CLUSTER_TOPOLOGY
    from .stability import lyapunov_variance as _lyapunov_variance
except ImportError:
    from models import ClusterObservation  # type: ignore
    from simulator import CLUSTER_TOPOLOGY  # type: ignore
    from stability import lyapunov_variance as _lyapunov_variance  # type: ignore

# ---------------------------------------------------------------------------
# SLA thresholds (must match environment.py)
# ---------------------------------------------------------------------------

SLA_LATENCY_MS: float = 0.20  # Normalized (200ms / 1000ms)
SLA_ERROR_RATE: float = 0.05

# ---------------------------------------------------------------------------
# Cost calibration
# ---------------------------------------------------------------------------

# Baseline cost = all 10 nodes at default capacity 3 with $0.05 / capacity-unit.
# 10 * 3 * 0.05 = $1.50 / hr.  This is what a perfectly provisioned agent pays.
BASELINE_COST_PER_HOUR: float = 1.50
MIN_COST_PER_HOUR: float = 0.05    # 1 active node at min capacity 1
MAX_COST_PER_HOUR: float = 25.00   # 10 nodes at ~50 capacity units (overprovisioned blow-out)
# Exponential cost penalty harshness — higher = steeper curve
COST_PENALTY_K: float = 3.0

# ---------------------------------------------------------------------------
# Stability normalisation
# ---------------------------------------------------------------------------

# Energy reference point for stability scoring (Task 1 baseline).
# This is a calibration midpoint, not a hard "full score" cutoff.
TARGET_ENERGY: float = 2000.0
# Curvature for stability scoring:
# score = 1 / (1 + (avg_energy / TARGET_ENERGY)^STABILITY_CURVE_POWER)
STABILITY_CURVE_POWER: float = 2.0

# ---------------------------------------------------------------------------
# Recovery speed (task-2) — normalized queue depth threshold [0, 1]
# ---------------------------------------------------------------------------

RECOVERY_QUEUE_CLEAR_THRESHOLD: float = 0.10
RECOVERY_SPEED_CAP: int = 10  # ticks; recovery_score = max(0, 1 - speed/CAP)

# ---------------------------------------------------------------------------
# VIP protection (task-3) — surge detection threshold
# ---------------------------------------------------------------------------

SURGE_INCOMING_THRESHOLD: float = 0.60  # normalized incoming_request_rate > 0.6

# ---------------------------------------------------------------------------
# Action record type
# ---------------------------------------------------------------------------


class ActionRecord:
    """A single action taken by the agent during an episode."""
    __slots__ = ("action_type", "target_node_id", "parameter", "had_effect")

    def __init__(self, action_type: str, target_node_id: str,
                 parameter: float, had_effect: bool):
        self.action_type = action_type
        self.target_node_id = target_node_id
        self.parameter = parameter
        self.had_effect = had_effect


# ---------------------------------------------------------------------------
# Grade with task-aware composite
# ---------------------------------------------------------------------------


class Grade:
    # Task-specific weight profiles for composite computation
    TASK_WEIGHTS: Dict[str, Dict[str, float]] = {
        "task-1": {"uptime": 0.4, "stability": 0.2, "cost": 0.4},
        "task-2": {"uptime": 0.25, "stability": 0.15, "cost": 0.25, "recovery": 0.35},
        "task-3": {"uptime": 0.35, "stability": 0.15, "cost_weighted": 0.35, "vip_protection": 0.15},
    }

    def __init__(self, task_id: str, scores: Dict[str, float]):
        self.task_id = task_id
        self.scores = scores

    @property
    def composite(self) -> float:
        """
        Weighted composite score using task-specific weight profiles.

        Task-1: 0.4*uptime + 0.2*stability + 0.4*cost
        Task-2: 0.25*uptime + 0.15*stability + 0.25*cost + 0.35*recovery
                (falls back to task-1 weights if recovery_speed is NaN)
        Task-3: 0.35*uptime + 0.15*stability + 0.35*cost_weighted + 0.15*vip_protection
                (cost_weighted = cost if uptime >= 0.5 else 0.0)

        Additional modifiers:
        - Invalid Action Penalty: -0.05 per forbidden command
        - Episode bonuses: +0.10 if zero VIP failures, +0.05 if <3 SLA violations,
          +0.05 if no invalid actions
        """
        uptime = self.scores["uptime"]
        stability = self.scores["stability"]
        cost = self.scores["cost"]
        invalid_penalty = self.scores.get("invalid_actions", 0) * 0.05

        # Episode-level prevention bonuses (NOT in step reward to avoid double-counting)
        bonus = 0.0
        if self.scores.get("vip_failure_count", 0) == 0:
            bonus += 0.10  # Zero VIP failures all episode
        if self.scores.get("violations", 0) < 3:
            bonus += 0.05  # Very few SLA violations all episode
        if self.scores.get("invalid_actions", 0) == 0:
            bonus += 0.05  # Clean actions all episode

        # Select weight profile by task
        weights = self.TASK_WEIGHTS.get(self.task_id, self.TASK_WEIGHTS["task-1"])

        if self.task_id == "task-2":
            recovery_speed = self.scores.get("recovery_speed")
            if recovery_speed is not None and not math.isnan(recovery_speed):
                recovery_score = max(0.0, 1.0 - recovery_speed / RECOVERY_SPEED_CAP)
                score = (
                    weights.get("uptime", 0.25) * uptime
                    + weights.get("stability", 0.15) * stability
                    + weights.get("cost", 0.25) * cost
                    + weights.get("recovery", 0.35) * recovery_score
                )
            else:
                # Fallback: no failure triggered this seed, use task-1 weights
                score = 0.4 * uptime + 0.2 * stability + 0.4 * cost

        elif self.task_id == "task-3":
            cost_weight = 1.0 if uptime >= 0.5 else 0.0
            cost_weighted = cost * cost_weight
            vip_protection = self.scores.get("vip_protection", 0.0)
            score = (
                weights.get("uptime", 0.35) * uptime
                + weights.get("stability", 0.15) * stability
                + weights.get("cost_weighted", 0.35) * cost_weighted
                + weights.get("vip_protection", 0.15) * vip_protection
            )
        else:
            score = (
                weights.get("uptime", 0.4) * uptime
                + weights.get("stability", 0.2) * stability
                + weights.get("cost", 0.4) * cost
            )

        return max(0.0, min(1.0, score - invalid_penalty + bonus))

    def summary(self) -> str:
        s = self.scores
        parts = [
            f"[{self.task_id}] composite={self.composite:.3f}",
            f"uptime={s['uptime']:.3f}",
            f"cost={s['cost']:.3f}",
            f"stability={s['stability']:.3f}",
            f"SLA_violations={int(s['violations'])}",
        ]
        if s.get("invalid_actions", 0) > 0:
            parts.append(f"INVALID={int(s['invalid_actions'])}")
        if "recovery_speed" in s and s["recovery_speed"] is not None and not math.isnan(s["recovery_speed"]):
            parts.append(f"recovery={s['recovery_speed']:.0f}ticks")
        if "vip_protection" in s:
            parts.append(f"vip_prot={s['vip_protection']:.1f}")
        if "action_efficiency" in s:
            parts.append(f"eff={s['action_efficiency']:.2f}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Episode Grader — collects observations + actions, computes Grade
# ---------------------------------------------------------------------------


class EpisodeGrader:
    """Consumes observations and actions from an environment episode to produce a grade."""

    def __init__(self, task_id: str = "task-1"):
        self.task_id = task_id
        self._records: List[Dict[str, Any]] = []
        self._action_records: List[ActionRecord] = []
        self._lyapunov_history: List[float] = []

    def record(self, observation: ClusterObservation) -> None:
        """Add a step's telemetry to the grading buffer."""
        obs_dict = observation.model_dump()
        self._records.append(obs_dict)
        lyap_val = obs_dict.get("lyapunov_energy", 0.0)
        self._lyapunov_history.append(float(lyap_val))

    def record_action(self, action_type: str, target_node_id: str,
                      parameter: float, had_effect: bool) -> None:
        """Record an action taken by the agent for efficiency analysis."""
        self._action_records.append(
            ActionRecord(action_type, target_node_id, parameter, had_effect)
        )

    # ── New metric computation methods ─────────────────────────────────────

    def _compute_action_efficiency(self) -> float:
        """Fraction of actions that had measurable effect. Range [0, 1]."""
        if not self._action_records:
            return 1.0  # No actions = trivially efficient
        effective = sum(1 for a in self._action_records if a.had_effect)
        return effective / len(self._action_records)

    def _compute_action_distribution(self) -> Dict[str, int]:
        """Count of each ActionType across the episode."""
        return dict(Counter(a.action_type for a in self._action_records))

    def _compute_node_heatmap(self) -> Dict[str, Dict[str, int]]:
        """Count of actions per node, grouped by action type."""
        heatmap: Dict[str, Dict[str, int]] = {}
        for a in self._action_records:
            by_type = heatmap.setdefault(a.action_type, {})
            by_type[a.target_node_id] = by_type.get(a.target_node_id, 0) + 1
        return heatmap

    def _compute_lyapunov_variance(self) -> float:
        """Variance of Lyapunov energy across the episode."""
        if len(self._lyapunov_history) < 2:
            return 0.0
        return _lyapunov_variance(self._lyapunov_history)

    def _compute_recovery_speed(self) -> Optional[float]:
        """
        Ticks from first FAILED node to when all its children have
        queue_depth < RECOVERY_QUEUE_CLEAR_THRESHOLD (normalized).
        Returns None if no failure occurred (NaN sentinel).
        """
        if self.task_id != "task-2":
            return None

        # Find first tick with a FAILED node
        t_fail: Optional[int] = None
        failed_node_id: Optional[str] = None
        for tick_idx, rec in enumerate(self._records):
            for node in rec.get("nodes", []):
                status = str(node.get("status", ""))
                if status == "FAILED":
                    t_fail = tick_idx
                    failed_node_id = node.get("node_id")
                    break
            if t_fail is not None:
                break

        if t_fail is None or failed_node_id is None:
            return float("nan")  # No failure in this seed

        # Get children of the failed node from DAG topology
        children = CLUSTER_TOPOLOGY.get(failed_node_id, [])
        if not children:
            # Leaf node failed — no children to starve, recovery = immediate
            return 0.0

        # Find first tick after failure where ALL children have cleared queues
        for tick_idx in range(t_fail, len(self._records)):
            rec = self._records[tick_idx]
            all_clear = True
            for node in rec.get("nodes", []):
                if node.get("node_id") in children:
                    if float(node.get("queue_depth", 1.0)) >= RECOVERY_QUEUE_CLEAR_THRESHOLD:
                        all_clear = False
                        break
            if all_clear:
                return float(tick_idx - t_fail)

        # Never recovered within the episode
        return float(len(self._records) - t_fail)

    def _compute_cost_trajectory(self) -> float:
        """
        Linear regression slope of cost over time.
        Negative = agent reduced cost (good). Positive = cost climbing.
        """
        costs = [r.get("current_cost_per_hour", 0.0) for r in self._records]
        n = len(costs)
        if n < 2:
            return 0.0
        mean_t = (n - 1) / 2.0
        mean_c = sum(costs) / n
        cov = sum((i - mean_t) * (costs[i] - mean_c) for i in range(n))
        var_t = sum((i - mean_t) ** 2 for i in range(n))
        if var_t == 0:
            return 0.0
        return cov / var_t

    def _compute_peak_queue_sum(self) -> float:
        """Maximum total_queue_backlog observed across the episode."""
        return max((r.get("total_queue_backlog", 0.0) for r in self._records), default=0.0)

    def _compute_vip_protection(self) -> float:
        """
        Task-3 only: 1.0 if node-0 never hit FAILED or DEGRADED during
        the surge window, else 0.0.

        The task-3 surge adds ~60 req/tick directly to node-1 and node-2
        via a side channel that bypasses node-0 (simulator direct_injections).
        Node-0's own incoming_request_rate stays ~0.30 — well below any
        threshold — so we detect the surge window from the nodes that
        actually receive it (node-1, node-2) instead.
        """
        if self.task_id != "task-3":
            return 0.0

        for rec in self._records:
            # Detect surge window: node-1 or node-2 has elevated incoming
            surge_active = False
            for node in rec.get("nodes", []):
                nid = node.get("node_id", "")
                incoming = float(node.get("incoming_request_rate", 0.0))
                if nid in ("node-1", "node-2") and incoming > SURGE_INCOMING_THRESHOLD:
                    surge_active = True
                    break

            if not surge_active:
                continue

            # During surge: check if node-0 is unhealthy
            for node in rec.get("nodes", []):
                if node.get("node_id") == "node-0":
                    status = str(node.get("status", ""))
                    if status in ("FAILED", "DEGRADED"):
                        return 0.0
                    break
        return 1.0

    # ── Main scoring method ────────────────────────────────────────────────

    def score(self) -> Grade:
        """Computes the final multi-dimensional performance grade."""
        if not self._records:
            return Grade(self.task_id, {
                "uptime": 0, "cost": 0, "stability": 0, "violations": 0
            })

        n = len(self._records)

        # ── 1. Uptime score ────────────────────────────────────────────────
        # Note: We exclude the t=0 state from uptime if n > 1.
        records_to_count = self._records[1:] if len(self._records) > 1 else self._records
        n_steps = len(records_to_count)

        sla_ok_steps = sum(
            1 for r in records_to_count
            if r.get("average_latency_ms", 0.0) <= SLA_LATENCY_MS
            and r.get("error_rate", 0.0) <= SLA_ERROR_RATE
        )
        uptime_score = sla_ok_steps / n_steps

        # Total cumulative SLA violations (use the last record's counter
        # since environment.py tracks this cumulatively)
        total_violations = self._records[-1].get("sla_violations", 0)

        # ── 2. Cost score ──────────────────────────────────────────────────
        # Computes efficiency relative to a 'perfectly provisioned' system.
        avg_cost = sum(r.get("current_cost_per_hour", 0.0) for r in self._records) / n

        # Exponential cost penalty: cost_score = exp(-k * over_provisioning_ratio)
        # over_provisioning_ratio = (avg_cost - BASELINE) / BASELINE
        # A perfectly provisioned agent (avg_cost == BASELINE) scores exp(0) = 1.0.
        # An agent that doubles the baseline (massive SCALE_UP spam) scores
        # exp(-3.0) ≈ 0.05 — nearly zero cost contribution.
        over_ratio = max(0.0, (avg_cost - BASELINE_COST_PER_HOUR) / BASELINE_COST_PER_HOUR)
        cost_score = max(0.0, min(1.0, math.exp(-COST_PENALTY_K * over_ratio)))

        # ── 3. Stability score ─────────────────────────────────────────────
        # Smooth inverse-energy score with no early saturation.
        # Avoids flattening diverse "good" policies into a perfect 1.0 bucket.
        avg_energy = sum(r.get("lyapunov_energy", 0.0) for r in self._records) / n
        if avg_energy <= 0:
            stability_score = 1.0
        else:
            ratio = avg_energy / TARGET_ENERGY
            stability_score = 1.0 / (1.0 + (ratio ** STABILITY_CURVE_POWER))

        # ── 4. Invalid Action tracking ──────────────────────────────────────
        total_invalid = self._records[-1].get("invalid_action_count", 0)
        total_vip_failures = self._records[-1].get("vip_failure_count", 0)

        # ── 5. New episode-level metrics ────────────────────────────────────
        recovery_speed = self._compute_recovery_speed()
        vip_protection = self._compute_vip_protection()
        action_efficiency = self._compute_action_efficiency()
        action_distribution = self._compute_action_distribution()
        node_heatmap = self._compute_node_heatmap()
        lyap_var = self._compute_lyapunov_variance()
        cost_trajectory = self._compute_cost_trajectory()
        peak_queue = self._compute_peak_queue_sum()

        scores: Dict[str, float] = {
            "uptime": uptime_score,
            "cost": cost_score,
            "stability": stability_score,
            "violations": total_violations,
            "invalid_actions": total_invalid,
            "vip_failure_count": total_vip_failures,
            "action_efficiency": action_efficiency,
            "lyapunov_variance": lyap_var,
            "cost_trajectory": cost_trajectory,
            "peak_queue_sum": peak_queue,
        }
        # Only include recovery_speed for task-2 (NaN-safe)
        if recovery_speed is not None:
            scores["recovery_speed"] = recovery_speed
        # Only include vip_protection for task-3
        if self.task_id == "task-3":
            scores["vip_protection"] = vip_protection
        # Action distribution and node heatmap as non-float metadata
        scores["action_distribution"] = action_distribution  # type: ignore[assignment]
        scores["node_heatmap"] = node_heatmap  # type: ignore[assignment]

        return Grade(self.task_id, scores)


def score_episode(task_id: str, observations: List[ClusterObservation]) -> Grade:
    """Helper for one-shot grading."""
    grader = EpisodeGrader(task_id)
    for obs in observations:
        grader.record(obs)
    return grader.score()
