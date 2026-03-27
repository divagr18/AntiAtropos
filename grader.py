"""
AntiAtropos Episode Grader — Phase 3.

Evaluates a completed episode and returns a deterministic scalar score
in [0.0, 1.0] that the hackathon judges (and the baseline agent) can use
to compare policies.

Each episode is scored on three sub-metrics, combined with task-specific
weights that reflect what each task is actually testing:

    Sub-metric          Measures
    ─────────────────────────────────────────────────────────────────────
    uptime_score        Fraction of steps that were SLA-compliant.
                        (latency < 200 ms AND error_rate < 5 %)

    cost_score          How close to the minimum viable provisioning the
                        agent ran.  Scaling up unnecessarily is penalised.

    stability_score     How well the agent kept Lyapunov energy V(s) low
                        and non-oscillating.  Based on the variance of
                        the V(s) trajectory, normalised to [0, 1].

    Task weights
    ─────────────────────────────────────────────────────────────────────
    task-1  Predictive Scaling:   uptime 0.40  cost 0.40  stability 0.20
    task-2  Fault Tolerance:      uptime 0.60  cost 0.20  stability 0.20
    task-3  Stability Under Surge:uptime 0.20  cost 0.20  stability 0.60

Usage
-----
    grader = EpisodeGrader(task_id="task-1")
    for obs in episode:          # obs is a plain dict from the API
        grader.record(obs)
    result = grader.score()
    print(result.composite_score)

Or with already-collected observations:
    result = grade_episode(observations, task_id="task-2")
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

try:
    from .stability import compute_lyapunov, lyapunov_variance
except ImportError:
    from stability import compute_lyapunov, lyapunov_variance  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# SLA thresholds (must match environment.py)
# ---------------------------------------------------------------------------

SLA_LATENCY_MS: float = 200.0
SLA_ERROR_RATE: float = 0.05

# ---------------------------------------------------------------------------
# Cost calibration
# ---------------------------------------------------------------------------

# Baseline cost = all N nodes active at initial capacity, no scaling.
# environment.py: cost = active_nodes × $0.10 / hr.
# With N=5 nodes and none failed:  $0.50 / hr is the neutral baseline.
BASELINE_COST_PER_HOUR: float = 0.50
MIN_COST_PER_HOUR: float = 0.10   # 1 node surviving
MAX_COST_PER_HOUR: float = 2.00   # all 5 nodes healthy (no capacity scaling in cost model)

# ---------------------------------------------------------------------------
# Stability normalisation
# ---------------------------------------------------------------------------

# Maximum "expected" Lyapunov variance for a fully uncontrolled episode.
# At OVERLOAD_THRESHOLD=80 with 5 nodes: V ~ 5×80² = 32 000.
# Variance across a 100-tick episode of rapid growth ≈ (32000)² / 4 ~ 2.56e8.
# We use a conservative ceiling so mild oscillations still score well.
MAX_LYAPUNOV_VARIANCE: float = 5.0e7

# ---------------------------------------------------------------------------
# Task weights
# ---------------------------------------------------------------------------

TASK_WEIGHTS: dict[str, dict[str, float]] = {
    "task-1": {"uptime": 0.40, "cost": 0.40, "stability": 0.20},
    "task-2": {"uptime": 0.60, "cost": 0.20, "stability": 0.20},
    "task-3": {"uptime": 0.20, "cost": 0.20, "stability": 0.60},
}
DEFAULT_WEIGHTS: dict[str, float] = {"uptime": 0.40, "cost": 0.30, "stability": 0.30}


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GradeResult:
    """
    Full grading breakdown for a single episode.

    Attributes
    ----------
    task_id:            Which task was graded.
    episode_length:     Total ticks recorded.
    sla_violations:     Cumulative SLA-breaching ticks (latency or error_rate).
    uptime_score:       [0,1]  Fraction of SLA-compliant ticks.
    cost_score:         [0,1]  Cost efficiency (1.0 = minimum viable spend).
    stability_score:    [0,1]  Lyapunov stability (1.0 = perfectly flat energy).
    composite_score:    [0,1]  Weighted combination, task-specific weights.
    avg_lyapunov:       Mean Lyapunov energy V(s) over the episode.
    lyapunov_var:       Variance of V(s) — the raw stability metric.
    avg_cost_per_hour:  Mean infrastructure cost over the episode.
    weights:            The task-specific weights used in aggregation.
    """

    task_id: str
    episode_length: int
    sla_violations: int

    uptime_score: float
    cost_score: float
    stability_score: float
    composite_score: float

    avg_lyapunov: float
    lyapunov_var: float
    avg_cost_per_hour: float

    weights: dict[str, float]

    def summary(self) -> str:
        """Human-readable one-liner."""
        return (
            f"[{self.task_id}] composite={self.composite_score:.3f} "
            f"| uptime={self.uptime_score:.3f} "
            f"| cost={self.cost_score:.3f} "
            f"| stability={self.stability_score:.3f} "
            f"| SLA violations={self.sla_violations}/{self.episode_length}"
        )


# ---------------------------------------------------------------------------
# EpisodeGrader
# ---------------------------------------------------------------------------

class EpisodeGrader:
    """
    Accumulates per-step observations and grades the episode at the end.

    Call ``record(obs)`` after every ``env.step()`` (and optionally after
    ``env.reset()`` for the t=0 baseline reading).

    ``obs`` can be:
      - A plain dict (as returned by the JSON API / client._parse_result).
      - Anything with attribute access that mirrors ClusterObservation fields.
    """

    def __init__(self, task_id: str = "task-1") -> None:
        self._task_id = task_id
        self._records: list[dict] = []

    # -----------------------------------------------------------------------
    # Data ingestion
    # -----------------------------------------------------------------------

    def record(self, obs) -> None:
        """
        Record one step's observation.

        Args:
            obs: ClusterObservation (Pydantic model) or equivalent plain dict.
        """
        if isinstance(obs, dict):
            self._records.append(obs)
        else:
            # Accept Pydantic models or any object with matching attributes
            self._records.append({
                "average_latency_ms":  getattr(obs, "average_latency_ms", 0.0),
                "error_rate":          getattr(obs, "error_rate", 0.0),
                "current_cost_per_hour": getattr(obs, "current_cost_per_hour", 0.0),
                "lyapunov_energy":     getattr(obs, "lyapunov_energy", 0.0),
                "sla_violations":      getattr(obs, "sla_violations", 0),
                "nodes":               getattr(obs, "nodes", []),
            })

    def reset(self, task_id: Optional[str] = None) -> None:
        """Clear all recorded observations (e.g. between episodes)."""
        self._records.clear()
        if task_id is not None:
            self._task_id = task_id

    # -----------------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------------

    def score(self) -> GradeResult:
        """
        Compute and return the episode grade.

        Returns:
            GradeResult with all sub-scores and the composite score.

        Raises:
            ValueError: If no observations have been recorded yet.
        """
        if not self._records:
            raise ValueError("No observations recorded. Call record() at least once.")

        n = len(self._records)

        # ── 1. Uptime score ────────────────────────────────────────────────
        sla_ok_steps = sum(
            1 for r in self._records
            if r.get("average_latency_ms", 0.0) < SLA_LATENCY_MS
            and r.get("error_rate", 0.0) < SLA_ERROR_RATE
        )
        uptime_score = sla_ok_steps / n

        # Total cumulative SLA violations (use the last record's counter
        # since environment.py tracks this cumulatively)
        sla_violations = self._records[-1].get("sla_violations", 0)

        # ── 2. Cost score ──────────────────────────────────────────────────
        costs = [r.get("current_cost_per_hour", BASELINE_COST_PER_HOUR)
                 for r in self._records]
        avg_cost = statistics.mean(costs)

        # Score = 1 when avg_cost == MIN_COST (best case: 1 surviving node).
        # Score = 0 when avg_cost == MAX_COST (fully over-provisioned).
        # Linear interpolation, clamped to [0, 1].
        if MAX_COST_PER_HOUR > MIN_COST_PER_HOUR:
            cost_score = 1.0 - (avg_cost - MIN_COST_PER_HOUR) / (
                MAX_COST_PER_HOUR - MIN_COST_PER_HOUR
            )
        else:
            cost_score = 1.0
        cost_score = max(0.0, min(1.0, cost_score))

        # ── 3. Stability score ─────────────────────────────────────────────
        v_history = [r.get("lyapunov_energy", 0.0) for r in self._records]
        avg_lyapunov = statistics.mean(v_history)
        v_var = lyapunov_variance(v_history)

        # Normalise variance: 0 variance → 1.0, MAX_LYAPUNOV_VARIANCE → 0.0
        # Use exponential decay so moderate variance still scores reasonably.
        stability_score = math.exp(-v_var / MAX_LYAPUNOV_VARIANCE)

        # ── 4. Composite score ─────────────────────────────────────────────
        weights = TASK_WEIGHTS.get(self._task_id, DEFAULT_WEIGHTS)
        composite_score = (
            weights["uptime"]    * uptime_score
            + weights["cost"]    * cost_score
            + weights["stability"] * stability_score
        )
        composite_score = max(0.0, min(1.0, composite_score))

        return GradeResult(
            task_id=self._task_id,
            episode_length=n,
            sla_violations=sla_violations,
            uptime_score=round(uptime_score, 4),
            cost_score=round(cost_score, 4),
            stability_score=round(stability_score, 4),
            composite_score=round(composite_score, 4),
            avg_lyapunov=round(avg_lyapunov, 2),
            lyapunov_var=round(v_var, 2),
            avg_cost_per_hour=round(avg_cost, 4),
            weights=weights,
        )


# ---------------------------------------------------------------------------
# Functional convenience wrapper
# ---------------------------------------------------------------------------

def grade_episode(
    observations: list,
    task_id: str = "task-1",
) -> GradeResult:
    """
    Grade a completed episode from a pre-collected list of observations.

    Args:
        observations: List of ClusterObservation dicts or model instances.
        task_id:      Task ID string ('task-1', 'task-2', 'task-3').

    Returns:
        GradeResult with the full scoring breakdown.
    """
    grader = EpisodeGrader(task_id=task_id)
    for obs in observations:
        grader.record(obs)
    return grader.score()
