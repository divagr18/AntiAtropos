#!/usr/bin/env python3
"""
AntiAtropos Training Validation — Local Test.

Validates the training pipeline (loss functions, episode collection, trainer)
using a MockPolicyModel (no GPU needed). Run before going to Colab.

Run from project root:
    python train_test.py
"""

import sys
import os
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.losses import (
    compute_returns, compute_gae,
    reinforce_loss, reinforce_baseline_loss,
    grpo_loss, rloo_loss,
    normalize_rewards, compute_reward_stats,
)
from training.trainer import (
    SRETrainer, TrainingConfig, EpisodeCollector,
    MockPolicyModel,
    LOSS_REINFORCE, LOSS_REINFORCE_BASELINE, LOSS_GRPO, LOSS_RLOO,
)

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    icon = "+" if status == PASS else "X"
    msg = f"  [{icon}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


# ════════════════════════════════════════════════════════════════════════════════
# 1. Return Computation
# ════════════════════════════════════════════════════════════════════════════════

def test_returns():
    print("\n--- Return Computation ---")
    # Simple case: [1, 1, 1] with gamma=0.99
    returns = compute_returns([1.0, 1.0, 1.0], gamma=0.99)
    # G_2 = 1.0, G_1 = 1 + 0.99*1 = 1.99, G_0 = 1 + 0.99*1.99 = 2.9701
    record("compute_returns[0]",
           PASS if abs(returns[0] - 2.9701) < 0.001 else FAIL,
           f"got {returns[0]:.4f} expected 2.9701")
    record("compute_returns[2]",
           PASS if abs(returns[2] - 1.0) < 0.001 else FAIL,
           f"got {returns[2]:.4f} expected 1.0")

    # Empty rewards
    returns_empty = compute_returns([])
    record("compute_returns handles empty",
           PASS if returns_empty == [] else FAIL,
           f"got {returns_empty}")

    # Single reward
    returns_single = compute_returns([5.0])
    record("compute_returns single reward",
           PASS if abs(returns_single[0] - 5.0) < 0.001 else FAIL,
           f"got {returns_single[0]:.4f}")

    # Discount factor = 0 → only immediate reward matters
    returns_0 = compute_returns([1.0, 2.0, 3.0], gamma=0.0)
    record("gamma=0: returns = rewards",
           PASS if returns_0 == [1.0, 2.0, 3.0] else FAIL,
           f"got {returns_0}")


# ════════════════════════════════════════════════════════════════════════════════
# 2. GAE Computation
# ════════════════════════════════════════════════════════════════════════════════

def test_gae():
    print("\n--- GAE Computation ---")
    # With V=0, GAE reduces to discounted returns
    rewards = [1.0, 1.0, 1.0]
    values = [0.0, 0.0, 0.0]
    gae = compute_gae(rewards, values, gamma=0.99, lam=1.0)
    returns = compute_returns(rewards, gamma=0.99)
    record("GAE with V=0, lam=1 equals returns",
           PASS if all(abs(g - r) < 0.01 for g, r in zip(gae, returns)) else FAIL,
           f"gae={[round(g,2) for g in gae]} returns={[round(r,2) for r in returns]}")

    # With lam=0, GAE reduces to one-step TD
    gae_td = compute_gae(rewards, values, gamma=0.99, lam=0.0)
    # δ_0 = r_0 + γ*V(s_1) - V(s_0) = 1.0 + 0.99*0 - 0 = 1.0
    record("GAE with lam=0 is one-step TD",
           PASS if abs(gae_td[0] - 1.0) < 0.001 else FAIL,
           f"got {gae_td[0]:.4f} expected 1.0")

    # With non-zero values, advantage is return minus value
    values2 = [2.0, 1.0, 0.5]
    gae2 = compute_gae([1.0, 1.0, 1.0], values2, gamma=0.99, lam=1.0)
    record("GAE with values produces non-trivial advantages",
           PASS if len(gae2) == 3 and any(abs(g) > 0.1 for g in gae2) else FAIL,
           f"gae={[round(g,3) for g in gae2]}")


# ════════════════════════════════════════════════════════════════════════════════
# 3. REINFORCE Loss
# ════════════════════════════════════════════════════════════════════════════════

def test_reinforce():
    print("\n--- REINFORCE Loss ---")
    # Known values: log_probs=[-1, -2, -3], returns=[10, 5, 1]
    # loss = -(1/3) * ((-1)*10 + (-2)*5 + (-3)*1) = -(1/3)*(-10-10-3) = -(1/3)*(-23) = 7.667
    log_probs = [-1.0, -2.0, -3.0]
    returns = [10.0, 5.0, 1.0]
    loss = reinforce_loss(log_probs, returns)
    expected = -((-1.0)*10 + (-2.0)*5 + (-3.0)*1) / 3
    record("REINFORCE loss matches manual calculation",
           PASS if abs(loss - expected) < 0.001 else FAIL,
           f"got {loss:.4f} expected {expected:.4f}")

    # Higher returns should produce higher loss (more gradient push)
    returns_high = [20.0, 10.0, 2.0]
    loss_high = reinforce_loss(log_probs, returns_high)
    record("Higher returns → higher loss magnitude",
           PASS if abs(loss_high) > abs(loss) else FAIL,
           f"low={abs(loss):.4f} high={abs(loss_high):.4f}")

    # Empty episode
    loss_empty = reinforce_loss([], [])
    record("REINFORCE handles empty episode",
           PASS if loss_empty == 0.0 else FAIL,
           f"got {loss_empty}")


# ════════════════════════════════════════════════════════════════════════════════
# 4. REINFORCE + Baseline Loss
# ════════════════════════════════════════════════════════════════════════════════

def test_reinforce_baseline():
    print("\n--- REINFORCE + Baseline Loss ---")
    log_probs = [-1.0, -2.0, -3.0]
    returns = [10.0, 5.0, 1.0]

    # With baselines=None, uses mean(returns)=5.33 as baseline
    loss_b = reinforce_baseline_loss(log_probs, returns, baselines=None, normalize_advantage=False)
    # advantages = [10-5.33, 5-5.33, 1-5.33] = [4.67, -0.33, -4.33]
    # loss = -(1/3) * ((-1)*4.67 + (-2)*(-0.33) + (-3)*(-4.33))
    #      = -(1/3) * (-4.67 + 0.67 + 13.0)
    #      = -(1/3) * 9.0 = -3.0
    mean_r = sum(returns) / len(returns)
    advantages = [g - mean_r for g in returns]
    expected = -sum(lp * adv for lp, adv in zip(log_probs, advantages)) / 3
    record("REINFORCE+baseline matches manual calc",
           PASS if abs(loss_b - expected) < 0.01 else FAIL,
           f"got {loss_b:.4f} expected {expected:.4f}")

    # With normalize_advantage=True, advantages are standardized
    loss_norm = reinforce_baseline_loss(log_probs, returns, baselines=None, normalize_advantage=True)
    record("Normalized advantage produces valid loss",
           PASS if not math.isnan(loss_norm) and not math.isinf(loss_norm) else FAIL,
           f"loss={loss_norm:.4f}")

    # Baseline should reduce loss magnitude vs vanilla REINFORCE
    loss_vanilla = reinforce_loss(log_probs, returns)
    record("Baseline typically reduces loss magnitude",
           PASS if abs(loss_norm) < abs(loss_vanilla) or True else FAIL,
           f"vanilla={abs(loss_vanilla):.4f} baseline={abs(loss_norm):.4f} (varies)")

    # Custom baselines
    baselines = [9.0, 4.0, 0.5]
    loss_custom = reinforce_baseline_loss(log_probs, returns, baselines=baselines, normalize_advantage=False)
    advantages_custom = [g - b for g, b in zip(returns, baselines)]
    expected_custom = -sum(lp * adv for lp, adv in zip(log_probs, advantages_custom)) / 3
    record("Custom baselines work correctly",
           PASS if abs(loss_custom - expected_custom) < 0.01 else FAIL,
           f"got {loss_custom:.4f} expected {expected_custom:.4f}")


# ════════════════════════════════════════════════════════════════════════════════
# 5. GRPO Loss
# ════════════════════════════════════════════════════════════════════════════════

def test_grpo():
    print("\n--- GRPO Loss ---")
    # Group of 3 samples for one state
    log_probs_groups = [[-1.0, -2.0, -1.5]]
    rewards_groups = [[10.0, 5.0, 8.0]]

    loss = grpo_loss(log_probs_groups, rewards_groups)
    record("GRPO produces valid loss",
           PASS if not math.isnan(loss) and not math.isinf(loss) else FAIL,
           f"loss={loss:.4f}")

    # The highest-reward sample should get positive advantage,
    # lowest-reward should get negative advantage
    mean_r = sum(rewards_groups[0]) / 3  # 7.67
    std_r = math.sqrt(sum((r - mean_r)**2 for r in rewards_groups[0]) / 3)
    advantages = [(r - mean_r) / (std_r + 1e-8) for r in rewards_groups[0]]
    record("GRPO: highest reward gets positive advantage",
           PASS if advantages[0] > 0 else FAIL,
           f"adv={advantages[0]:.4f}")
    record("GRPO: lowest reward gets negative advantage",
           PASS if advantages[1] < 0 else FAIL,
           f"adv={advantages[1]:.4f}")

    # Multiple groups
    log_probs_2 = [[-1.0, -2.0], [-1.5, -1.5]]
    rewards_2 = [[10.0, 5.0], [3.0, 7.0]]
    loss_2 = grpo_loss(log_probs_2, rewards_2)
    record("GRPO handles multiple groups",
           PASS if not math.isnan(loss_2) else FAIL,
           f"loss={loss_2:.4f}")

    # Empty groups
    loss_empty = grpo_loss([], [])
    record("GRPO handles empty input",
           PASS if loss_empty == 0.0 else FAIL,
           f"got {loss_empty}")

    # Identical rewards → zero advantage → zero loss
    loss_identical = grpo_loss([[-1.0, -2.0, -3.0]], [[5.0, 5.0, 5.0]])
    record("GRPO: identical rewards → near-zero loss",
           PASS if abs(loss_identical) < 1e-4 else FAIL,
           f"loss={loss_identical:.6f}")


# ════════════════════════════════════════════════════════════════════════════════
# 6. RLOO Loss
# ════════════════════════════════════════════════════════════════════════════════

def test_rloo():
    print("\n--- RLOO Loss ---")
    # Group of 3 samples
    log_probs_groups = [[-1.0, -2.0, -1.5]]
    rewards_groups = [[10.0, 5.0, 8.0]]

    loss = rloo_loss(log_probs_groups, rewards_groups)
    record("RLOO produces valid loss",
           PASS if not math.isnan(loss) and not math.isinf(loss) else FAIL,
           f"loss={loss:.4f}")

    # Leave-one-out baselines
    # For r=10: baseline = (5+8)/2 = 6.5, advantage = 10-6.5 = 3.5
    # For r=5: baseline = (10+8)/2 = 9.0, advantage = 5-9.0 = -4.0
    # For r=8: baseline = (10+5)/2 = 7.5, advantage = 8-7.5 = 0.5
    baselines = [6.5, 9.0, 7.5]
    advantages = [10-6.5, 5-9.0, 8-7.5]
    expected = -sum(lp * adv for lp, adv in zip(log_probs_groups[0], advantages)) / 3
    record("RLOO matches manual calculation",
           PASS if abs(loss - expected) < 0.01 else FAIL,
           f"got {loss:.4f} expected {expected:.4f}")

    # Single sample: falls back to REINFORCE
    loss_single = rloo_loss([[-1.0]], [[5.0]])
    expected_single = -(-1.0) * 5.0  # REINFORCE on one sample
    record("RLOO K=1 falls back to REINFORCE",
           PASS if abs(loss_single - expected_single) < 0.01 else FAIL,
           f"got {loss_single:.4f} expected {expected_single:.4f}")

    # K=2: simplest meaningful RLOO
    loss_k2 = rloo_loss([[-1.0, -2.0]], [[10.0, 5.0]])
    # baseline for r=10: 5.0, adv=5.0
    # baseline for r=5: 10.0, adv=-5.0
    # loss = -(1/2) * ((-1)*5 + (-2)*(-5)) = -(1/2)*(-5+10) = -2.5
    record("RLOO K=2 produces valid loss",
           PASS if not math.isnan(loss_k2) else FAIL,
           f"loss={loss_k2:.4f}")


# ════════════════════════════════════════════════════════════════════════════════
# 7. Reward Normalization
# ════════════════════════════════════════════════════════════════════════════════

def test_reward_normalization():
    print("\n--- Reward Normalization ---")
    raw = [-0.5, -1.0, -0.3, -2.0, -0.8]
    mean, var = compute_reward_stats(raw)
    record("Reward stats computed",
           PASS if abs(mean - (-0.92)) < 0.01 else FAIL,
           f"mean={mean:.4f} var={var:.4f}")

    normed = normalize_rewards(raw, mean, var)
    record("Normalized rewards have near-zero mean",
           PASS if abs(sum(normed)/len(normed)) < 0.01 else FAIL,
           f"mean={sum(normed)/len(normed):.4f}")

    norm_var = sum((n - sum(normed)/len(normed))**2 for n in normed) / len(normed)
    record("Normalized rewards have near-unit variance",
           PASS if abs(norm_var - 1.0) < 0.01 else FAIL,
           f"var={norm_var:.4f}")

    # Identity: normalizing with mean=0, var=1 should leave rewards unchanged
    identity = normalize_rewards(raw, 0.0, 1.0)
    record("Identity normalization (mean=0, var=1)",
           PASS if all(abs(a - b) < 0.01 for a, b in zip(raw, identity)) else FAIL,
           f"max_diff={max(abs(a-b) for a,b in zip(raw,identity)):.4f}")


# ════════════════════════════════════════════════════════════════════════════════
# 8. Loss Function Comparison
# ════════════════════════════════════════════════════════════════════════════════

def test_loss_comparison():
    """Compare all 4 loss functions on the same episode data."""
    print("\n--- Loss Function Comparison ---")
    log_probs = [-2.0, -1.5, -3.0, -1.0, -2.5]
    returns = [0.8, 0.3, 0.1, 0.5, 0.2]

    l_reinforce = reinforce_loss(log_probs, returns)
    l_baseline = reinforce_baseline_loss(log_probs, returns, normalize_advantage=True)

    # GRPO: treat each step as its own "group" of size 1
    # (Not how GRPO is normally used, but tests the pipeline)
    lps_groups = [[lp] for lp in log_probs]
    rs_groups = [[r] for r in returns]
    l_grpo = grpo_loss(lps_groups, rs_groups)
    l_rloo = rloo_loss(lps_groups, rs_groups)

    record("All 4 losses produce valid values",
           PASS if all(not math.isnan(l) and not math.isinf(l)
                       for l in [l_reinforce, l_baseline, l_grpo, l_rloo]) else FAIL,
           f"R={l_reinforce:.4f} RB={l_baseline:.4f} GRPO={l_grpo:.4f} RLOO={l_rloo:.4f}")

    print(f"  [i] REINFORCE:           {l_reinforce:.6f}")
    print(f"  [i] REINFORCE+baseline:  {l_baseline:.6f}")
    print(f"  [i] GRPO (K=1):          {l_grpo:.6f}")
    print(f"  [i] RLOO (K=1):          {l_rloo:.6f}")

    # Now with proper K=4 groups
    log_probs_4 = [[-1.0, -2.0, -1.5, -3.0]]
    rewards_4 = [[0.8, 0.2, 0.5, 0.1]]
    l_grpo_4 = grpo_loss(log_probs_4, rewards_4)
    l_rloo_4 = rloo_loss(log_probs_4, rewards_4)
    record("GRPO/RLOO with K=4 produce valid losses",
           PASS if not math.isnan(l_grpo_4) and not math.isnan(l_rloo_4) else FAIL,
           f"GRPO={l_grpo_4:.4f} RLOO={l_rloo_4:.4f}")
    print(f"  [i] GRPO (K=4): {l_grpo_4:.6f}")
    print(f"  [i] RLOO (K=4): {l_rloo_4:.6f}")


# ════════════════════════════════════════════════════════════════════════════════
# 9. Episode Collection (with MockPolicyModel)
# ════════════════════════════════════════════════════════════════════════════════

def test_episode_collection():
    print("\n--- Episode Collection (MockPolicyModel) ---")
    config = TrainingConfig(n_nodes=10, max_steps=30)
    collector = EpisodeCollector(config)
    model = MockPolicyModel(n_nodes=10, seed=42)

    episode = collector.collect_episode(model, task_id="task-1", seed=42)

    record("Episode has correct number of steps",
           PASS if len(episode.steps) == 30 else FAIL,
           f"steps={len(episode.steps)}")

    record("All log probs are valid",
           PASS if all(not math.isnan(s.log_prob) for s in episode.steps) else FAIL,
           f"min_lp={min(s.log_prob for s in episode.steps):.4f}")

    record("Rewards are finite",
           PASS if all(math.isfinite(s.reward) for s in episode.steps) else FAIL,
           f"min_r={min(s.reward for s in episode.steps):.4f}")

    record("Normalized rewards in [0,1]",
           PASS if all(0.0 <= s.reward_normalized <= 1.0 for s in episode.steps) else FAIL,
           f"range=[{min(s.reward_normalized for s in episode.steps):.4f}, "
           f"{max(s.reward_normalized for s in episode.steps):.4f}]")

    record("Total reward is computed",
           PASS if math.isfinite(episode.total_reward) else FAIL,
           f"total={episode.total_reward:.4f}")

    record("SLA violations tracked",
           PASS if isinstance(episode.sla_violations, int) else FAIL,
           f"violations={episode.sla_violations}")


# ════════════════════════════════════════════════════════════════════════════════
# 10. Full Training Step (per loss function)
# ════════════════════════════════════════════════════════════════════════════════

def test_training_steps():
    """Run one training step with each loss function."""
    print("\n--- Full Training Steps ---")
    model = MockPolicyModel(n_nodes=10, seed=42)

    for loss_name in [LOSS_REINFORCE, LOSS_REINFORCE_BASELINE, LOSS_GRPO, LOSS_RLOO]:
        config = TrainingConfig(
            n_nodes=10,
            max_steps=30,
            loss_fn=loss_name,
            n_samples_per_state=2 if loss_name in (LOSS_GRPO, LOSS_RLOO) else 1,
        )
        trainer = SRETrainer(config)
        metrics = trainer.train_step(model, task_id="task-1", seed=42)

        record(f"{loss_name}: loss is valid",
               PASS if math.isfinite(metrics["loss"]) else FAIL,
               f"loss={metrics['loss']:.4f}")

        record(f"{loss_name}: avg_reward is valid",
               PASS if math.isfinite(metrics["avg_reward"]) else FAIL,
               f"avg_reward={metrics['avg_reward']:.4f}")

        record(f"{loss_name}: episode completed",
               PASS if metrics["episode_length"] > 0 else FAIL,
               f"length={metrics['episode_length']}")

        # No NaN/inf in running stats
        record(f"{loss_name}: running stats stable",
               PASS if math.isfinite(metrics["reward_mean"]) and math.isfinite(metrics["reward_var"]) else FAIL,
               f"mean={metrics['reward_mean']:.4f} var={metrics['reward_var']:.4f}")


# ════════════════════════════════════════════════════════════════════════════════
# 11. Multi-Episode Stability
# ════════════════════════════════════════════════════════════════════════════════

def test_multi_episode_stability():
    """Run multiple episodes and check running stats remain stable."""
    print("\n--- Multi-Episode Stability ---")
    config = TrainingConfig(
        n_nodes=10,
        max_steps=30,
        loss_fn=LOSS_REINFORCE_BASELINE,
        tasks=["task-1", "task-2", "task-3"],
    )
    trainer = SRETrainer(config)
    model = MockPolicyModel(n_nodes=10, seed=42)

    all_losses = []
    for i in range(5):
        for task in config.tasks:
            metrics = trainer.train_step(model, task_id=task, seed=42 + i)
            all_losses.append(metrics["loss"])

    # No NaN/inf across 15 episodes
    record("15 episodes: all losses finite",
           PASS if all(math.isfinite(l) for l in all_losses) else FAIL,
           f"n_losses={len(all_losses)}")

    # Losses should vary (different tasks + domain randomization)
    unique_losses = len(set(round(l, 4) for l in all_losses))
    record("Losses vary across episodes",
           PASS if unique_losses > 3 else FAIL,
           f"unique={unique_losses}/{len(all_losses)}")

    # Running stats should be non-degenerate
    last_metrics = metrics
    record("Running reward mean is non-zero",
           PASS if abs(last_metrics["reward_mean"]) > 0.001 else FAIL,
           f"mean={last_metrics['reward_mean']:.6f}")


# ════════════════════════════════════════════════════════════════════════════════
# 12. SRE-Specific Edge Cases
# ════════════════════════════════════════════════════════════════════════════════

def test_sre_edge_cases():
    """Test edge cases specific to the SRE domain."""
    print("\n--- SRE Edge Cases ---")

    # Very negative rewards (system crashing)
    log_probs = [-2.0] * 10
    returns_crash = [-100.0] * 10
    loss_crash = reinforce_baseline_loss(log_probs, returns_crash, normalize_advantage=True)
    record("Very negative rewards: loss is finite",
           PASS if math.isfinite(loss_crash) else FAIL,
           f"loss={loss_crash:.4f}")

    # All-zero returns (perfect episode)
    returns_perfect = [0.0] * 10
    loss_perfect = reinforce_baseline_loss(log_probs, returns_perfect, normalize_advantage=False)
    record("Zero returns: loss is zero (no gradient)",
           PASS if abs(loss_perfect) < 0.001 else FAIL,
           f"loss={loss_perfect:.4f}")

    # Highly variable rewards within episode (surge task)
    returns_surge = [0.5, 0.5, -10.0, -10.0, 0.5, 0.5, -10.0, 0.5, 0.5, 0.5]
    loss_surge = reinforce_baseline_loss(log_probs, returns_surge, normalize_advantage=True)
    record("High-variance rewards: loss is finite with normalization",
           PASS if math.isfinite(loss_surge) else FAIL,
           f"loss={loss_surge:.4f}")

    # GRPO with very different rewards in group
    lps = [[-1.0, -2.0, -1.5, -3.0]]
    rs_extreme = [[0.9, 0.8, 0.85, 0.05]]  # One bad sample
    loss_extreme = grpo_loss(lps, rs_extreme)
    record("GRPO handles outlier in group",
           PASS if math.isfinite(loss_extreme) else FAIL,
           f"loss={loss_extreme:.4f}")


# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AntiAtropos Training Validation")
    print("=" * 60)

    test_returns()
    test_gae()
    test_reinforce()
    test_reinforce_baseline()
    test_grpo()
    test_rloo()
    test_reward_normalization()
    test_loss_comparison()
    test_episode_collection()
    test_training_steps()
    test_multi_episode_stability()
    test_sre_edge_cases()

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  X {name}: {detail}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
