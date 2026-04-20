# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AntiAtropos Grading Logic — Evaluation Engine.

Provides standard scoring for hackathon submissions across three dimensions:
1. Uptime: Fraction of ticks where cluster-wide SLA (latency/errors) was met.
2. Cost: Normalized efficiency score based on provisioned capacity.
3. Stability: Mean Lyapunov energy normalized to a target baseline.
"""

import math
from typing import Dict, Any, List
from .models import ClusterObservation

# ---------------------------------------------------------------------------
# SLA thresholds (must match environment.py)
# ---------------------------------------------------------------------------

SLA_LATENCY_MS: float = 0.20  # Normalized (200ms / 1000ms)
SLA_ERROR_RATE: float = 0.05

# ---------------------------------------------------------------------------
# Cost calibration
# ---------------------------------------------------------------------------

# Baseline cost = all 5 nodes at default capacity 3 with $0.05 / capacity-unit.
# 5 * 3 * 0.05 = $0.75 / hr.  This is what a perfectly provisioned agent pays.
BASELINE_COST_PER_HOUR: float = 0.75
MIN_COST_PER_HOUR: float = 0.05    # 1 active node at min capacity 1
MAX_COST_PER_HOUR: float = 1.25    # 5 nodes at capacity 5 (MAX_CAPACITY)
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


class Grade:
    def __init__(self, task_id: str, scores: Dict[str, float]):
        self.task_id = task_id
        self.scores = scores

    @property
    def composite(self) -> float:
        """
        Weighted composite score.

        Weights deliberately penalise cost heavily so that brute-force
        SCALE_UP spam cannot achieve a high composite even with perfect uptime.

        Hardening:
        - Task 3 coupling: Cost only rewards if Uptime is >= 50%. Stops 'Cheap-but-Dead'.
        - Invalid Action Penalty: -0.05 per forbidden command (SHED_LOAD on critical).
        - Episode bonuses: Prevention rewards that DON'T overlap with step-level
          reward signals (no double-counting). These are:
            +0.10 if zero VIP failures throughout the episode
            +0.05 if SLA violations < 3 for the whole episode
            +0.05 if no invalid actions
        These bonuses are small and additive, avoiding overlap with the
        step-level reward which already penalizes SLA violations and barrier
        breaches on each tick. The bonuses reward *sustained* prevention.
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

        if self.task_id == "task-3":
            # Coupling: If uptime < 0.5, the cost benefit is zeroed out.
            cost_weight = 1.0 if uptime >= 0.5 else 0.0
            score = (0.4 * uptime + 0.2 * stability + 0.4 * (cost * cost_weight))
        else:
            score = (0.4 * uptime + 0.2 * stability + 0.4 * cost)

        return max(0.0, min(1.0, score - invalid_penalty + bonus))

    def summary(self) -> str:
        s = self.scores
        summary = (
            f"[{self.task_id}] composite={self.composite:.3f} | "
            f"uptime={s['uptime']:.3f} | cost={s['cost']:.3f} | "
            f"stability={s['stability']:.3f} | SLA violations={int(s['violations'])}/101"
        )
        if s.get("invalid_actions", 0) > 0:
            summary += f" | INVALID ACTIONS={int(s['invalid_actions'])}"
        return summary


class EpisodeGrader:
    """Consumes observations from an environment episode to produce a grade."""

    def __init__(self, task_id: str = "task-1"):
        self.task_id = task_id
        self._records: List[Dict[str, Any]] = []

    def record(self, observation: ClusterObservation) -> None:
        """Add a step's telemetry to the grading buffer."""
        self._records.append(observation.model_dump())

    def score(self) -> Grade:
        """Computes the final multi-dimensional performance grade."""
        if not self._records:
            return Grade(self.task_id, {"uptime": 0, "cost": 0, "stability": 0, "violations": 0})

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

        return Grade(self.task_id, {
            "uptime": uptime_score,
            "cost": cost_score,
            "stability": stability_score,
            "violations": total_violations,
            "invalid_actions": total_invalid,
            "vip_failure_count": total_vip_failures,
        })


def score_episode(task_id: str, observations: List[ClusterObservation]) -> Grade:
    """Helper for one-shot grading."""
    grader = EpisodeGrader(task_id)
    for obs in observations:
        grader.record(obs)
    return grader.score()
