"""
AntiAtropos Stability Layer — Phase 3.

This module is the mathematical core of the Lyapunov-inspired reward signal.
It is intentionally a stateless collection of pure functions so it can be
swapped, extended, or unit-tested independently of the environment.

Key concepts implemented
------------------------
1. Lyapunov Energy  V(s) = Σ Q_i²
   The "potential energy" of the cluster.  Zero means all queues are empty.
   A rising V means the cluster is destabilising.

2. Lyapunov Drift  ΔV(t) = V(s_t) − V(s_{t−1})
   The primary stabilising objective.  The reward penalises positive drift.
   Negative drift (energy decreasing) is "good" — the agent drove the system
   toward the equilibrium.

3. Control-Barrier Function  h_i(s) = max(0, Q_i − Q_max)²
   A soft safety constraint.  h_i > 0 only when node i has exceeded the
   hard-queue-depth safety ceiling Q_max.  Summing over all nodes gives the
   total barrier violation penalty.

4. Neely Drift-Plus-Penalty (optional, advanced)
   From Neely's Lyapunov optimisation framework:

       Δ(t) + V_weight · p(t)

   where:
       Δ(t)   = ΔV    — the one-step Lyapunov drift.
       p(t)   = cost  — the per-step penalty (infrastructure cost here).
       V_weight — trade-off parameter:  large V_weight → prioritise cost;
                  small V_weight → prioritise stability.

   Minimising this expression at every step produces a policy that is both
   stable (queue-stable in the mean) and cost-efficient.
"""

from __future__ import annotations

import math
import statistics
from typing import Sequence


# ---------------------------------------------------------------------------
# Safety ceiling used by the barrier function
# ---------------------------------------------------------------------------

Q_BARRIER_MAX: float = 150.0
"""Queue depth above which the barrier function fires (hard safety zone).
Set higher than OVERLOAD_THRESHOLD (80) to allow the agent time to react
before the barrier penalty kicks in."""

STABILITY_WINDOW: int = 10
"""Number of ticks to look back when judging whether the system is
trend-stable (V is on a decreasing trajectory)."""


# ---------------------------------------------------------------------------
# Core Lyapunov functions
# ---------------------------------------------------------------------------

def compute_lyapunov(nodes: list[dict]) -> float:
    """
    V(s) = Σ Q_i²

    Sum of squared queue depths across all nodes.  This is the cluster's
    Lyapunov energy.  Lower is more stable; zero means all queues are empty.

    Args:
        nodes: List of per-node state dicts (as returned by simulator.state()).
               Each dict must contain the key ``queue_depth``.

    Returns:
        Scalar Lyapunov energy  ≥ 0.
    """
    return float(
        sum(
            float(n.get("importance_weight", 1.0)) * (n["queue_depth"] ** 2)
            for n in nodes
        )
    )


def compute_drift(v_prev: float, v_curr: float) -> float:
    """
    ΔV(t) = V(s_t) − V(s_{t−1})

    One-step drift in Lyapunov energy.

    Negative drift → the agent moved the system toward a lower-energy state.
    Positive drift → the cluster is destabilising.

    Args:
        v_prev: Lyapunov energy at the *previous* tick.
        v_curr: Lyapunov energy at the *current* tick.

    Returns:
        Signed scalar drift value.
    """
    return v_curr - v_prev


def compute_barrier(nodes: list[dict], q_max: float = Q_BARRIER_MAX) -> float:
    """
    Control-Barrier Function (CBF) violation penalty.

        H(s) = Σ_i  max(0, Q_i − Q_max)²

    This is zero when no node exceeds the ceiling and grows quadratically as
    queues enter the "hard danger zone" above Q_max.  It can be added to the
    reward as an extra penalty for unsafe states.

    Args:
        nodes:  Per-node state dicts (must contain ``queue_depth``).
        q_max:  Safety ceiling for queue depth.  Default: Q_BARRIER_MAX.

    Returns:
        Scalar barrier violation energy  ≥ 0.
    """
    violation = 0.0
    for n in nodes:
        excess = n["queue_depth"] - q_max
        if excess > 0:
            violation += excess ** 2
    return violation


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def is_lyapunov_stable(
    v_history: Sequence[float],
    window: int = STABILITY_WINDOW,
) -> bool:
    """
    Return True if the Lyapunov energy has been on a non-increasing trend
    over the last ``window`` ticks.

    Uses a simple linear regression slope: if slope ≤ 0 the system is
    considered trend-stable.

    Args:
        v_history: Ordered sequence of Lyapunov energy values (oldest first).
        window:    How many recent values to consider.

    Returns:
        True if the system is trend-stable, False otherwise.
    """
    recent = list(v_history[-window:])
    if len(recent) < 2:
        return True   # not enough data — assume stable at episode start

    n = len(recent)
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = statistics.mean(recent)

    num = sum((xs[i] - mean_x) * (recent[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))

    if den == 0:
        return True

    slope = num / den
    return slope <= 0.0


def lyapunov_variance(v_history: Sequence[float]) -> float:
    """
    Variance of the Lyapunov energy trajectory over an episode.

    Used by the grader as the primary stability metric: a lower variance
    means the agent kept the cluster in a consistently stable state, rather
    than allowing wild oscillations.

    Args:
        v_history: All per-tick V(s) values for the episode.

    Returns:
        Population variance of the energy trajectory.
    """
    if len(v_history) < 2:
        return 0.0
    return statistics.variance(v_history)


# ---------------------------------------------------------------------------
# Neely Drift-Plus-Penalty (advanced reward signal)
# ---------------------------------------------------------------------------

def drift_plus_penalty(
    v_prev: float,
    v_curr: float,
    penalty_cost: float,
    V_weight: float = 1.0,
) -> float:
    """
    Neely's Drift-Plus-Penalty objective:

        DPP(t) = ΔV(t) + V_weight · p(t)

    where:
        ΔV(t)        = v_curr − v_prev  (Lyapunov drift)
        p(t)         = penalty_cost     (infrastructure cost this tick)
        V_weight     = trade-off coefficient:
                         large  → agent optimises cost more aggressively,
                         small  → agent focuses on stability.

    Minimising this at each step produces a queue-stable policy with bounded
    average cost — the theoretical guarantee from Neely's framework.

    This function can substitute for the simpler ΔV term in the reward
    when you want to make the cost trade-off explicit and theoretically
    grounded (rather than the ad-hoc β·Cost term).

    Args:
        v_prev:       Lyapunov energy at previous tick.
        v_curr:       Lyapunov energy at current tick.
        penalty_cost: Per-step cost to penalise (e.g. current_cost_per_hour).
        V_weight:     Trade-off weight  V  in Neely's framework.

    Returns:
        Scalar DPP value.  The reward should negate this:
        R_t = −DPP(t) − γ·SLA_violation_step
    """
    delta_v = compute_drift(v_prev, v_curr)
    return delta_v + V_weight * penalty_cost


# ---------------------------------------------------------------------------
# Convenience: full reward computation (matches environment.py formula)
# ---------------------------------------------------------------------------

def compute_reward(
    v_prev: float,
    v_curr: float,
    cost: float,
    sla_violation_step: int,
    alpha: float = 1.0,
    beta: float = 0.05,
    gamma: float = 2.0,
) -> float:
    """
    R_t = −(α·ΔV(s)  +  β·Cost  +  γ·SLA_violation_step)

    Convenience wrapper that mirrors the reward formula in environment.py.
    Can be used by the baseline agent to simulate rewards without calling
    the server, or by the grader to reconstruct reward trajectories.

    Args:
        v_prev:         Lyapunov energy at previous tick.
        v_curr:         Lyapunov energy at current tick.
        cost:           Infrastructure cost this tick (USD/hr).
        sla_violation_step: 1 if this step violated SLA, else 0.
        alpha:          Weight on Lyapunov drift.
        beta:           Weight on cost.
        gamma:          Weight on SLA violations.

    Returns:
        Scalar reward (higher is better, always ≤ 0 in a stable episode).
    """
    delta_v = compute_drift(v_prev, v_curr)
    return -(alpha * delta_v + beta * cost + gamma * sla_violation_step)
