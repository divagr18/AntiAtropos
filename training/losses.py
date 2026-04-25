"""
AntiAtropos RL Loss Functions.

Pure-Python implementations of policy gradient loss functions for LLM-based
SRE agents. These are mathematically identical to their PyTorch counterparts
and can be validated locally without GPU.

When porting to PyTorch (Colab), simply replace:
    - float ops with torch.tensor ops
    - sum() with torch.sum()
    - log() with torch.log()
    - The gradient flows through log_prob automatically

Loss function selection guide for the SRE domain:
─────────────────────────────────────────────────
┌───────────────────────┬────────────┬──────────────┬─────────────────────────┐
│ Method                │ Variance   │ Needs Value  │ Best for                │
├───────────────────────┼────────────┼──────────────┼─────────────────────────┤
│ REINFORCE             │ High       │ No           │ Quick baseline          │
│ REINFORCE + baseline  │ Medium     │ Optional     │ Most use cases          │
│ GRPO                  │ Low        │ No           │ Multi-sample rollouts   │
│ RLOO                  │ Lowest     │ No           │ Small groups (K=2-4)    │
└───────────────────────┴────────────┴──────────────┴─────────────────────────┘

Recommended starting point: REINFORCE + baseline (simplest, good variance,
no value head needed). If variance is still too high, switch to GRPO with K=4.

For the SRE domain specifically:
- Episodes are 100 steps long → significant credit assignment challenge
- Rewards are dense (computed every step) → advantage normalization is key
- Delayed effects (boot delay = 5 ticks) → GAE helps bridge the gap
"""

from __future__ import annotations

import math
from typing import List, Optional


# ════════════════════════════════════════════════════════════════════════════════
# Return / Advantage computation (shared across all loss functions)
# ════════════════════════════════════════════════════════════════════════════════

def compute_returns(
    rewards: List[float],
    gamma: float = 0.99,
) -> List[float]:
    """
    Compute discounted returns (Monte Carlo) for each timestep.

        G_t = r_t + γ * r_{t+1} + γ² * r_{t+2} + ... + γ^{T-t} * r_T

    Args:
        rewards: Per-step rewards [r_0, r_1, ..., r_{T-1}].
        gamma: Discount factor. 0.99 = far-sighted, 0.9 = myopic.

    Returns:
        List of returns [G_0, G_1, ..., G_{T-1}] same length as rewards.
    """
    returns: List[float] = []
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        returns.insert(0, g)
    return returns


def compute_gae(
    rewards: List[float],
    values: List[float],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> List[float]:
    """
    Generalized Advantage Estimation (GAE).

        Â_t = Σ_{l=0}^{T-t-1} (γλ)^l δ_{t+l}

    where δ_t = r_t + γ * V(s_{t+1}) - V(s_t) is the TD error.

    GAE provides a bias-variance trade-off controlled by λ:
        λ = 0  →  one-step TD (low variance, high bias)
        λ = 1  →  Monte Carlo returns (high variance, no bias)

    For SRE: λ=0.95 is a good default. The 5-tick boot delay means
    actions have delayed effects — GAE with λ close to 1 helps propagate
    credit across those gaps.

    Args:
        rewards: Per-step rewards [r_0, ..., r_{T-1}].
        values:  State value estimates [V(s_0), ..., V(s_{T-1})].
                 Pass a list of zeros for V=0 baseline (reduces to MC).
        gamma:   Discount factor.
        lam:     GAE lambda (trade-off parameter).

    Returns:
        List of GAE advantages [Â_0, ..., Â_{T-1}].
    """
    assert len(rewards) == len(values), f"len mismatch: rewards={len(rewards)} values={len(values)}"
    advantages: List[float] = []
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0  # Terminal state has V=0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages


# ════════════════════════════════════════════════════════════════════════════════
# Loss Functions
# ════════════════════════════════════════════════════════════════════════════════

def reinforce_loss(
    log_probs: List[float],
    returns: List[float],
) -> float:
    """
    Vanilla REINFORCE (Williams, 1992).

        L = -(1/T) Σ_t log π(a_t | s_t) · G_t

    The gradient of this loss is an unbiased estimator of the policy gradient:
        ∇J(θ) = E[Σ_t ∇log π(a_t|s_t) · G_t]

    Properties:
        - Unbiased but HIGH variance (no baseline)
        - Simplest possible policy gradient method
        - Good for initial prototyping, upgrade to baseline version ASAP

    Args:
        log_probs: log π(a_t | s_t) for each timestep.
        returns:   Discounted returns G_t for each timestep.

    Returns:
        Scalar loss (minimize to maximize expected return).
    """
    assert len(log_probs) == len(returns), f"len mismatch: log_probs={len(log_probs)} returns={len(returns)}"
    t = len(log_probs)
    if t == 0:
        return 0.0
    return -sum(lp * g for lp, g in zip(log_probs, returns)) / t


def reinforce_baseline_loss(
    log_probs: List[float],
    returns: List[float],
    baselines: Optional[List[float]] = None,
    normalize_advantage: bool = True,
    eps: float = 1e-8,
) -> float:
    """
    REINFORCE with baseline (variance reduction).

        L = -(1/T) Σ_t log π(a_t | s_t) · Â_t

    where Â_t = G_t - b_t is the advantage estimate.

    The baseline b_t does NOT introduce bias (only reduces variance)
    because E[∇log π(a|s) · b(s)] = 0 for any function b(s).

    Baseline options:
        - None (or zeros): Falls back to vanilla REINFORCE on returns
        - Running mean of returns: Simple, effective, no extra model needed
        - Learned value function: Most powerful, but needs value head

    For SRE agents on Colab: use running-mean baseline (pass baselines as
    the moving average of returns seen so far). No extra model needed.

    Advantage normalization: Standardizes Â to zero mean and unit variance.
    This is critical for SRE because raw returns can span orders of magnitude
    (0.001 vs 10.0) across episodes. Normalization keeps the learning rate
    well-conditioned.

    Args:
        log_probs:         log π(a_t | s_t) for each timestep.
        returns:           Discounted returns G_t for each timestep.
        baselines:         Baseline estimates b_t. If None, uses mean(returns).
        normalize_advantage: Whether to standardize advantages (recommended).
        eps:               Epsilon for numerical stability in normalization.

    Returns:
        Scalar loss.
    """
    assert len(log_probs) == len(returns), f"len mismatch: log_probs={len(log_probs)} returns={len(returns)}"
    t = len(log_probs)
    if t == 0:
        return 0.0

    # Compute advantages
    if baselines is None:
        # Default baseline = mean of returns (simple but effective)
        baseline_val = sum(returns) / len(returns)
        advantages = [g - baseline_val for g in returns]
    else:
        assert len(baselines) == len(returns), f"len mismatch: baselines={len(baselines)} returns={len(returns)}"
        advantages = [g - b for g, b in zip(returns, baselines)]

    # Normalize advantages (critical for SRE reward scale)
    if normalize_advantage and len(advantages) > 1:
        mean_adv = sum(advantages) / len(advantages)
        var_adv = sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)
        std_adv = math.sqrt(var_adv) + eps
        advantages = [(a - mean_adv) / std_adv for a in advantages]

    return -sum(lp * adv for lp, adv in zip(log_probs, advantages)) / t


def grpo_loss(
    log_probs_groups: List[List[float]],
    rewards_groups: List[List[float]],
    eps: float = 1e-8,
) -> float:
    """
    Group Relative Policy Optimization (GRPO).

    For each state s, generate K sample actions and compute group-relative
    advantages without needing a value function:

        Â_k = (r_k - μ_group) / σ_group

        L = -(1/N) Σ_i (1/K_i) Σ_k log π(a_{i,k} | s_i) · Â_{i,k}

    where μ_group and σ_group are the mean and std of rewards within group i.

    This is the method used in DeepSeek-R1. It eliminates the need for a
    value head entirely — the group statistics serve as the baseline.

    Pros:
        - No value function needed
        - Low variance (group statistics absorb reward scale)
        - Natural normalization

    Cons:
        - Requires K >= 2 rollouts per state (K * more compute)
        - For K=1, falls back to REINFORCE (no baseline)

    For SRE on Colab: Use K=4 with QLoRA on a T4. Each "group" is 4
    different actions sampled for the same cluster state.

    Args:
        log_probs_groups: List of groups, each group is log π(a_k|s) for K samples.
        rewards_groups:   List of groups, each group is reward_k for K samples.
        eps:              Epsilon for std normalization.

    Returns:
        Scalar loss.
    """
    assert len(log_probs_groups) == len(rewards_groups), "group count mismatch"
    if not log_probs_groups:
        return 0.0

    total_loss = 0.0
    n_groups = 0

    for log_probs, rewards in zip(log_probs_groups, rewards_groups):
        assert len(log_probs) == len(rewards), f"group size mismatch: {len(log_probs)} vs {len(rewards)}"
        k = len(log_probs)
        if k == 0:
            continue

        # Group statistics
        mean_r = sum(rewards) / k
        var_r = sum((r - mean_r) ** 2 for r in rewards) / k
        std_r = math.sqrt(var_r) + eps

        # Normalized advantages
        advantages = [(r - mean_r) / std_r for r in rewards]

        # Policy gradient for this group
        group_loss = -sum(lp * adv for lp, adv in zip(log_probs, advantages)) / k
        total_loss += group_loss
        n_groups += 1

    return total_loss / max(1, n_groups)


def rloo_loss(
    log_probs_groups: List[List[float]],
    rewards_groups: List[List[float]],
) -> float:
    """
    REINFORCE Leave-One-Out (RLOO).

    Similar to GRPO but uses a leave-one-out baseline instead of group
    statistics. For each sample k in a group of K:

        b_k = (1/(K-1)) Σ_{j≠k} r_j    (leave-one-out mean)

        Â_k = r_k - b_k

        L = -(1/N) Σ_i (1/K_i) Σ_k log π(a_{i,k} | s_i) · Â_{i,k}

    RLOO has lower variance than GRPO for small group sizes (K=2-4) because
    the baseline is computed from the actual other samples rather than a
    statistical estimate. For K >= 8, GRPO and RLOO converge.

    For SRE on Colab: Best choice when you can only afford K=2-4 rollouts.
    The leave-one-out baseline is surprisingly effective.

    Args:
        log_probs_groups: List of groups, each group is log π(a_k|s) for K samples.
        rewards_groups:   List of groups, each group is reward_k for K samples.

    Returns:
        Scalar loss.
    """
    assert len(log_probs_groups) == len(rewards_groups), "group count mismatch"
    if not log_probs_groups:
        return 0.0

    total_loss = 0.0
    n_groups = 0

    for log_probs, rewards in zip(log_probs_groups, rewards_groups):
        assert len(log_probs) == len(rewards), f"group size mismatch: {len(log_probs)} vs {len(rewards)}"
        k = len(log_probs)
        if k == 0:
            continue

        if k == 1:
            # Single sample: no baseline possible, fall back to REINFORCE
            total_loss += -log_probs[0] * rewards[0]
            n_groups += 1
            continue

        # Leave-one-out baselines
        sum_r = sum(rewards)
        baselines = [(sum_r - r) / (k - 1) for r in rewards]
        advantages = [r - b for r, b in zip(rewards, baselines)]

        # Policy gradient
        group_loss = -sum(lp * adv for lp, adv in zip(log_probs, advantages)) / k
        total_loss += group_loss
        n_groups += 1

    return total_loss / max(1, n_groups)


# ════════════════════════════════════════════════════════════════════════════════
# Reward normalization utilities
# ════════════════════════════════════════════════════════════════════════════════

def normalize_rewards(
    rewards: List[float],
    running_mean: float = 0.0,
    running_var: float = 1.0,
    eps: float = 1e-8,
) -> List[float]:
    """
    Normalize rewards using running statistics.

    For SRE: Raw rewards are always negative (they're penalties). This function
    shifts them to be centered around zero with unit variance, which is
    essential for stable policy gradient updates.

    On Colab: Maintain a running mean/var across episodes and pass them here.
    Initialize with mean=0, var=1 and update with exponential moving average.

        running_mean = 0.99 * running_mean + 0.01 * batch_mean
        running_var  = 0.99 * running_var  + 0.01 * batch_var

    Args:
        rewards:       Raw rewards to normalize.
        running_mean:  Running mean estimate across episodes.
        running_var:   Running variance estimate across episodes.
        eps:           Numerical stability constant.

    Returns:
        Normalized rewards (zero mean, unit variance relative to running stats).
    """
    std = math.sqrt(running_var) + eps
    return [(r - running_mean) / std for r in rewards]


def compute_reward_stats(rewards: List[float]) -> tuple[float, float]:
    """
    Compute mean and variance of a reward list.

    Returns:
        (mean, variance) tuple.
    """
    if not rewards:
        return 0.0, 1.0
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    return mean, var
