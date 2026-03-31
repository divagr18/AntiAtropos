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

# Baseline cost = all 5 nodes at initial capacity 3 with $0.05 / capacity-unit.
# 5 * 3 * 0.05 = $0.75 / hr.
BASELINE_COST_PER_HOUR: float = 0.75
MIN_COST_PER_HOUR: float = 0.05   # 1 active node at min capacity 1
MAX_COST_PER_HOUR: float = 2.50   # 5 nodes at max capacity 10

# ---------------------------------------------------------------------------
# Stability normalisation
# ---------------------------------------------------------------------------

# Estimated 'perfect' energy for a stable system (Task 1 baseline)
TARGET_ENERGY: float = 2000.0


class Grade:
    def __init__(self, task_id: str, scores: Dict[str, float]):
        self.task_id = task_id
        self.scores = scores

    @property
    def composite(self) -> float:
        """Weighted average of performance metrics."""
        return (
            0.5 * self.scores["uptime"] +
            0.3 * self.scores["stability"] +
            0.2 * self.scores["cost"]
        )

    def summary(self) -> str:
        s = self.scores
        return (
            f"[{self.task_id}] composite={self.composite:.3f} | "
            f"uptime={s['uptime']:.3f} | cost={s['cost']:.3f} | "
            f"stability={s['stability']:.3f} | SLA violations={int(s['violations'])}/101"
        )


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
        
        # normalized cost [0, 1] - Higher is better efficiency (closer to minimum)
        if avg_cost <= MIN_COST_PER_HOUR:
            cost_score = 1.0
        else:
            cost_score = max(0.0, 1.0 - (avg_cost - MIN_COST_PER_HOUR) / (MAX_COST_PER_HOUR - MIN_COST_PER_HOUR))

        # ── 3. Stability score ─────────────────────────────────────────────
        # Normalized inverse of average Lyapunov energy.
        avg_energy = sum(r.get("lyapunov_energy", 0.0) for r in self._records) / n
        stability_score = max(0.0, min(1.0, TARGET_ENERGY / avg_energy if avg_energy > 0 else 1.0))

        return Grade(self.task_id, {
            "uptime": uptime_score,
            "cost": cost_score,
            "stability": stability_score,
            "violations": total_violations
        })


def score_episode(task_id: str, observations: List[ClusterObservation]) -> Grade:
    """Helper for one-shot grading."""
    grader = EpisodeGrader(task_id)
    for obs in observations:
        grader.record(obs)
    return grader.score()
