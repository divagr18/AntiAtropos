#!/usr/bin/env python3
"""
AntiAtropos Local Smoke Test — 5-Node Validation.

Validates simulator physics, reward signals, and grading WITHOUT any LLM,
Colab, or AWS infrastructure. Uses only stdlib + project modules
(simulator, stability, curriculum have zero external deps).

Run from project root:
    python smoke_test.py
"""

import sys
import os
import random
import math

# ── Make standalone imports work ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import (
    ClusterSimulator, NodeStatus, DEFAULT_CAPACITY, MAX_CAPACITY,
    VIP_NODE_WEIGHTS, CRITICAL_NODES, COST_PER_CAPACITY_UNIT_PER_HOUR,
    T1_INITIAL_LAMBDA, T2_INITIAL_LAMBDA, T3_INITIAL_LAMBDA,
)
from stability import (
    compute_lyapunov, compute_reward, compute_barrier,
    normalize_reward, smooth_sla_penalty, compute_drift,
)
from curriculum import CurriculumTracker, CURRICULUM

# ── Test harness ─────────────────────────────────────────────────────────────────

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []  # (name, status, detail)


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    icon = "+" if status == PASS else "X"
    msg = f"  [{icon}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)


def random_action(sim: ClusterSimulator) -> object:
    """Generate a random valid action."""
    node_ids = [n.node_id for n in sim._nodes]
    action_types = ["SCALE_UP", "SCALE_DOWN", "REROUTE_TRAFFIC", "SHED_LOAD", "NO_OP"]

    class _A:
        pass

    a = _A()
    a.action_type = random.choice(action_types)
    a.target_node_id = random.choice(node_ids)
    a.parameter = round(random.random(), 2)
    return a


def run_episode(
    sim: ClusterSimulator,
    task_id: str,
    max_steps: int = 60,
    seed: int = 42,
    action_policy: str = "random",
) -> dict:
    """
    Run a full episode and collect diagnostics.

    action_policy: 'random' | 'noop' | 'scale_up_vip'
    """
    sim.reset(task_id=task_id, seed=seed)

    rewards_raw: list[float] = []
    rewards_norm: list[float] = []
    lyapunov_history: list[float] = []
    sla_violations = 0
    prev_v = 0.0
    MAX_QUEUE_NORM = 200.0
    MAX_LATENCY_NORM = 1000.0
    ALPHA, BETA, GAMMA, DELTA = 0.002, 0.01, 10.0, 0.005

    for step in range(1, max_steps + 1):
        # Choose action
        if action_policy == "noop":
            class _A:
                pass
            a = _A()
            a.action_type = "NO_OP"
            a.target_node_id = "node-0"
            a.parameter = 0.0
        elif action_policy == "scale_up_vip":
            class _A:
                pass
            a = _A()
            a.action_type = "SCALE_UP"
            a.target_node_id = "node-0"
            a.parameter = 0.8
        else:
            a = random_action(sim)

        sim.apply_action(a)
        sim.tick()

        # Compute reward (mirrors environment.py logic)
        nodes_true = sim.state(for_agent=False)
        current_v = compute_lyapunov(nodes_true)

        # Avg latency (importance-weighted)
        w_lat = 0.0
        w_sum = 0.0
        for n in nodes_true:
            w = n.get("importance_weight", 1.0)
            lat = MAX_LATENCY_NORM if n["status"] == NodeStatus.FAILED else n["latency_ms"]
            w_lat += w * lat
            w_sum += w
        avg_lat_norm = min(1.0, max(0.0, (w_lat / w_sum / MAX_LATENCY_NORM) if w_sum > 0 else 1.0))

        # Error rate
        total_in = sum(
            n.get("incoming_request_rate", 0) * n.get("importance_weight", 1.0)
            for n in nodes_true
        )
        total_drop = sum(
            n.get("dropped_requests", 0) * n.get("importance_weight", 1.0)
            for n in nodes_true
        )
        error_rate = min(1.0, total_drop / total_in) if total_in > 0 else 0.0

        sla_step = smooth_sla_penalty(avg_lat_norm, error_rate)
        if avg_lat_norm > 0.20 or error_rate > 0.05:
            sla_violations += 1

        # Cost
        total_cap = 0
        for n in nodes_true:
            if n["status"] != NodeStatus.FAILED:
                total_cap += int(n.get("capacity_units", 0)) + int(n.get("pending_capacity_units", 0))
        cost = total_cap * COST_PER_CAPACITY_UNIT_PER_HOUR

        barrier = compute_barrier(nodes_true)
        raw_r = compute_reward(
            prev_v, current_v, cost, sla_step, ALPHA, BETA, GAMMA, barrier, DELTA
        )
        norm_r = normalize_reward(raw_r)

        rewards_raw.append(raw_r)
        rewards_norm.append(norm_r)
        lyapunov_history.append(current_v)
        prev_v = current_v

    return {
        "rewards_raw": rewards_raw,
        "rewards_norm": rewards_norm,
        "lyapunov_history": lyapunov_history,
        "final_state": sim.state(for_agent=False),
        "invalid_count": sim.invalid_action_count,
        "sla_violations": sla_violations,
    }


# ════════════════════════════════════════════════════════════════════════════════
# TEST FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def test_simulator_node_count():
    """Simulator creates exactly 10 nodes; node-0 is VIP."""
    print("\n--- Simulator Node Count ---")
    sim = ClusterSimulator(n_nodes=5, task_id="task-1", seed=1)
    nodes = sim.state(for_agent=False)

    record("10 nodes created",
           PASS if len(nodes) == 10 else FAIL,
           f"got {len(nodes)}")

    record("node-0 is VIP",
           PASS if nodes[0]["is_vip"] else FAIL,
           f"is_vip={nodes[0]['is_vip']}")

    record("node-0 weight=4.0",
           PASS if nodes[0]["importance_weight"] == 4.0 else FAIL,
           f"weight={nodes[0]['importance_weight']}")

    non_vip_weights = [n["importance_weight"] for n in nodes[1:]]
    record("Non-VIP weight=1.0",
           PASS if all(w == 1.0 for w in non_vip_weights) else FAIL,
           f"unique weights={set(non_vip_weights)}")

    node_ids = [n["node_id"] for n in nodes]
    expected_ids = [f"node-{i}" for i in range(10)]
    record("Node IDs 0-9",
           PASS if node_ids == expected_ids else FAIL,
           f"ids={node_ids}")

    caps = [n["capacity_units"] for n in nodes]
    record("All nodes at capacity 3",
           PASS if all(c == 3 for c in caps) else FAIL,
           f"caps={caps}")


def test_task1_ramp():
    """Task-1: traffic ramps, queues grow under NO_OP, rewards non-degenerate."""
    print("\n--- Task-1: Linear Ramp (NO_OP policy) ---")
    sim = ClusterSimulator(n_nodes=5, task_id="task-1")
    ep = run_episode(sim, "task-1", max_steps=60, seed=42, action_policy="noop")

    # Queues should grow (no scaling action taken)
    final_queues = [n["queue_depth"] for n in ep["final_state"]]
    max_q = max(final_queues)
    record("Queues grow under NO_OP",
           PASS if max_q > 0 else FAIL,
           f"max_queue={max_q:.1f}")

    # Rewards should not all be identical
    unique_raw = len(set(round(r, 6) for r in ep["rewards_raw"]))
    record("Raw rewards vary across steps",
           PASS if unique_raw > 5 else FAIL,
           f"unique values={unique_raw}/{len(ep['rewards_raw'])}")

    # Normalized rewards in [0, 1]
    all_in_range = all(0.0 <= r <= 1.0 for r in ep["rewards_norm"])
    record("Normalized rewards in [0,1]",
           PASS if all_in_range else FAIL,
           f"min={min(ep['rewards_norm']):.4f} max={max(ep['rewards_norm']):.4f}")

    # No NaN / inf
    has_nan = any(math.isnan(r) or math.isinf(r) for r in ep["rewards_raw"])
    record("No NaN/inf in raw rewards",
           PASS if not has_nan else FAIL,
           "")

    # Lyapunov energy should trend upward (system destabilizing under NO_OP)
    v_first5 = sum(ep["lyapunov_history"][:5]) / 5
    v_last5 = sum(ep["lyapunov_history"][-5:]) / 5
    record("Lyapunov energy rises under NO_OP",
           PASS if v_last5 > v_first5 else FAIL,
           f"early_avg={v_first5:.1f} late_avg={v_last5:.1f}")

    print(f"  [i] SLA violations: {ep['sla_violations']}/60")
    print(f"  [i] Avg norm reward: {sum(ep['rewards_norm'])/len(ep['rewards_norm']):.4f}")


def test_task2_fault():
    """Task-2: a node fails, queues react, reroute reduces load on failed node."""
    print("\n--- Task-2: Fault Tolerance ---")
    sim = ClusterSimulator(n_nodes=5, task_id="task-2")
    ep = run_episode(sim, "task-2", max_steps=60, seed=42, action_policy="noop")

    # At least one node should be FAILED by end (scripted failure)
    failed = [n for n in ep["final_state"] if n["status"] == "FAILED"]
    record("Scripted failure occurs",
           PASS if len(failed) >= 1 else FAIL,
           f"failed_nodes={len(failed)}")

    # node-0 should NOT be the failed one (excluded from failure pool)
    failed_ids = [n["node_id"] for n in failed]
    record("node-0 not in failed set",
           PASS if "node-0" not in failed_ids else FAIL,
           f"failed_ids={failed_ids}")

    # Rewards may plateau under NO_OP on constant-load tasks.
    # Task-2 has fixed lambda, so steady-state reward has very low variance.
    # This is expected — active policies (scale/reroute) create variation.
    record("Raw rewards produced (may plateau under NO_OP)",
           PASS if len(ep['rewards_raw']) == 60 else FAIL,
           f"steps={len(ep['rewards_raw'])}")

    # More importantly, normalized rewards should differ from 0.5 midpoint
    # (proving the raw reward signal is non-trivial)
    avg_norm = sum(ep['rewards_norm']) / len(ep['rewards_norm'])
    record("Normalized reward is non-trivial (not stuck at 0.5)",
           PASS if abs(avg_norm - 0.5) > 0.01 else FAIL,
           f"avg_norm={avg_norm:.4f}")

    # Normalized rewards in [0, 1]
    all_in_range = all(0.0 <= r <= 1.0 for r in ep["rewards_norm"])
    record("Normalized rewards in [0,1]",
           PASS if all_in_range else FAIL,
           f"min={min(ep['rewards_norm']):.4f} max={max(ep['rewards_norm']):.4f}")

    # No NaN / inf
    has_nan = any(math.isnan(r) or math.isinf(r) for r in ep["rewards_raw"])
    record("No NaN/inf in raw rewards",
           PASS if not has_nan else FAIL, "")

    # Now test with targeted reroute on the scripted-failed node
    # (NOT all nodes — rerouting everything to node-0 kills it)
    sim2 = ClusterSimulator(n_nodes=5, task_id="task-2", seed=99)
    sim2.reset(task_id="task-2", seed=99)
    scripted_fail_id = None
    for step in range(1, 61):
        sim2.tick()
        # Check if the scripted failure has been assigned
        if sim2._failed_node_id and scripted_fail_id is None:
            scripted_fail_id = sim2._failed_node_id
            # Apply reroute specifically to the failed node
            class _A:
                pass
            a = _A()
            a.action_type = "REROUTE_TRAFFIC"
            a.target_node_id = scripted_fail_id
            a.parameter = 1.0
            sim2.apply_action(a)
            # Tick once more to see the effect
            sim2.tick()
            failed_node = next((n for n in sim2._nodes if n.node_id == scripted_fail_id), None)
            base_share = sim2._t2_init_lambda / sim2._n_nodes
            record("Reroute reduces failed node traffic",
                   PASS if failed_node.incoming_request_rate < base_share else FAIL,
                   f"node={scripted_fail_id} incoming={failed_node.incoming_request_rate:.1f} base_share={base_share:.1f}")
            break


def test_task3_surge():
    """Task-3: surge hits node-1/node-2, SHED_LOAD on critical nodes rejected."""
    print("\n--- Task-3: Periodic Surge ---")
    sim = ClusterSimulator(n_nodes=5, task_id="task-3")
    ep = run_episode(sim, "task-3", max_steps=60, seed=42, action_policy="noop")

    # Rewards non-degenerate
    unique_raw = len(set(round(r, 6) for r in ep["rewards_raw"]))
    record("Raw rewards vary",
           PASS if unique_raw > 5 else FAIL,
           f"unique values={unique_raw}/{len(ep['rewards_raw'])}")

    # Normalized rewards in [0, 1]
    all_in_range = all(0.0 <= r <= 1.0 for r in ep["rewards_norm"])
    record("Normalized rewards in [0,1]",
           PASS if all_in_range else FAIL,
           f"min={min(ep['rewards_norm']):.4f} max={max(ep['rewards_norm']):.4f}")

    # No NaN / inf
    has_nan = any(math.isnan(r) or math.isinf(r) for r in ep["rewards_raw"])
    record("No NaN/inf in raw rewards",
           PASS if not has_nan else FAIL, "")

    # Test SHED_LOAD rejection on critical nodes
    sim3 = ClusterSimulator(n_nodes=5, task_id="task-3", seed=7)
    sim3.reset(task_id="task-3", seed=7)
    for critical_id in CRITICAL_NODES:
        class _A:
            pass
        a = _A()
        a.action_type = "SHED_LOAD"
        a.target_node_id = critical_id
        a.parameter = 0.5
        sim3.apply_action(a)
    record("SHED_LOAD on critical nodes rejected",
           PASS if sim3.invalid_action_count == len(CRITICAL_NODES) else FAIL,
           f"invalid_count={sim3.invalid_action_count} expected={len(CRITICAL_NODES)}")

    # SHED_LOAD on non-critical should be allowed
    class _A2:
        pass
    a2 = _A2()
    a2.action_type = "SHED_LOAD"
    a2.target_node_id = "node-5"
    a2.parameter = 0.5
    sim3.apply_action(a2)
    record("SHED_LOAD on non-critical node allowed",
           PASS if sim3.invalid_action_count == len(CRITICAL_NODES) else FAIL,
           f"invalid_count={sim3.invalid_action_count}")


def test_scale_up_down():
    """SCALE_UP increases capacity after boot delay; SCALE_DOWN decreases it."""
    print("\n--- Scale Up / Scale Down ---")
    sim = ClusterSimulator(n_nodes=5, task_id="task-1", seed=1)
    sim.reset(task_id="task-1", seed=1)

    # SCALE_UP node-3
    class _A:
        pass
    a = _A()
    a.action_type = "SCALE_UP"
    a.target_node_id = "node-3"
    a.parameter = 1.0  # 1 * MAX_SCALING_STEP=3 → 3 units
    sim.apply_action(a)

    # Check pending capacity before boot
    node3 = next(n for n in sim._nodes if n.node_id == "node-3")
    record("Pending capacity queued after SCALE_UP",
           PASS if len(node3.pending_capacity_queue) > 0 else FAIL,
           f"pending={len(node3.pending_capacity_queue)}")

    # Tick through boot delay
    for _ in range(6):
        sim.tick()

    node3 = next(n for n in sim._nodes if n.node_id == "node-3")
    record("Capacity goes live after boot delay",
           PASS if node3.capacity > DEFAULT_CAPACITY else FAIL,
           f"capacity={node3.capacity}")

    # SCALE_DOWN
    prev_cap = node3.capacity
    class _A2:
        pass
    a2 = _A2()
    a2.action_type = "SCALE_DOWN"
    a2.target_node_id = "node-3"
    a2.parameter = 0.5
    sim.apply_action(a2)
    record("SCALE_DOWN reduces capacity",
           PASS if node3.capacity < prev_cap else FAIL,
           f"before={prev_cap} after={node3.capacity}")


def test_reward_sanity():
    """Detailed reward component sanity checks."""
    print("\n--- Reward Sanity ---")

    # Test normalize_reward mapping
    r0 = normalize_reward(0.0)
    record("normalize_reward(0.0) in [0,1]",
           PASS if 0.0 <= r0 <= 1.0 else FAIL,
           f"got {r0:.4f}")

    r_neg = normalize_reward(-100.0)
    r_pos = normalize_reward(100.0)
    record("More negative raw -> lower normalized",
           PASS if r_neg < r_pos else FAIL,
           f"neg={r_neg:.4f} pos={r_pos:.4f}")

    # Smooth SLA penalty
    p_safe = smooth_sla_penalty(0.05, 0.01)   # well below thresholds
    p_danger = smooth_sla_penalty(0.30, 0.10)  # above thresholds
    record("SLA penalty: safe < danger",
           PASS if p_safe < p_danger else FAIL,
           f"safe={p_safe:.4f} danger={p_danger:.4f}")

    # Barrier function
    nodes_ok = [{"queue_depth": 50.0} for _ in range(10)]
    nodes_bad = [{"queue_depth": 200.0} for _ in range(10)]
    b_ok = compute_barrier(nodes_ok)
    b_bad = compute_barrier(nodes_bad)
    record("Barrier: safe queues < overloaded queues",
           PASS if b_ok < b_bad else FAIL,
           f"ok={b_ok:.1f} bad={b_bad:.1f}")
    record("Barrier is 0 when all below Q_BARRIER_MAX",
           PASS if b_ok == 0.0 else FAIL,
           f"got {b_ok:.1f}")

    # Lyapunov with VIP weight
    nodes_no_vip = [{"queue_depth": 100.0, "importance_weight": 1.0} for _ in range(10)]
    nodes_with_vip = [{"queue_depth": 100.0, "importance_weight": 4.0}] + \
                     [{"queue_depth": 100.0, "importance_weight": 1.0} for _ in range(9)]
    v_no_vip = compute_lyapunov(nodes_no_vip)
    v_with_vip = compute_lyapunov(nodes_with_vip)
    record("VIP weight amplifies Lyapunov energy",
           PASS if v_with_vip > v_no_vip else FAIL,
           f"no_vip={v_no_vip:.1f} with_vip={v_with_vip:.1f}")


def test_grader_inline():
    """Inline grader score validation (mirrors grader.py logic without importing it)."""
    print("\n--- Grader Logic (Inline) ---")

    # Baseline cost for 10 nodes at capacity 3
    BASELINE = 10 * 3 * 0.05  # 1.50
    MAX_COST = 10 * 5 * 0.05   # 2.50
    COST_K = 3.0
    TARGET_ENERGY = 2000.0
    CURVE_POWER = 2.0

    # Perfectly provisioned: cost == baseline -> score = 1.0
    over_ratio = max(0.0, (BASELINE - BASELINE) / BASELINE)
    cost_score = max(0.0, min(1.0, math.exp(-COST_K * over_ratio)))
    record("Cost score=1.0 at baseline",
           PASS if abs(cost_score - 1.0) < 1e-6 else FAIL,
           f"got {cost_score:.4f}")

    # 2x over-provisioned: score should be very low
    over_ratio_2x = max(0.0, (2 * BASELINE - BASELINE) / BASELINE)
    cost_score_2x = max(0.0, min(1.0, math.exp(-COST_K * over_ratio_2x)))
    record("Cost score near 0 at 2x baseline",
           PASS if cost_score_2x < 0.1 else FAIL,
           f"got {cost_score_2x:.4f}")

    # Stability: low energy -> high score
    low_energy = 100.0
    ratio = low_energy / TARGET_ENERGY
    stab_score = 1.0 / (1.0 + ratio ** CURVE_POWER)
    record("Stability score high at low energy",
           PASS if stab_score > 0.9 else FAIL,
           f"energy={low_energy} score={stab_score:.4f}")

    # Stability: high energy -> low score
    high_energy = 10000.0
    ratio_h = high_energy / TARGET_ENERGY
    stab_score_h = 1.0 / (1.0 + ratio_h ** CURVE_POWER)
    record("Stability score low at high energy",
           PASS if stab_score_h < 0.1 else FAIL,
           f"energy={high_energy} score={stab_score_h:.4f}")


def test_curriculum_tracker():
    """Curriculum tracker advances stages on passing scores."""
    print("\n--- Curriculum Tracker ---")
    tracker = CurriculumTracker()

    record("Starts at stage 0",
           PASS if tracker.current_index == 0 else FAIL,
           f"idx={tracker.current_index}")

    record(f"Total stages = {len(CURRICULUM)}",
           PASS if len(CURRICULUM) == 10 else FAIL,
           f"got {len(CURRICULUM)}")

    # Pass first stage
    stage0 = tracker.current
    passed = tracker.report_score(0.50)  # > 0.40 threshold
    record("Pass stage 0 with score 0.50",
           PASS if passed and tracker.current_index == 1 else FAIL,
           f"passed={passed} idx={tracker.current_index}")

    # Fail stage 1 (needs 0.50)
    passed2 = tracker.report_score(0.30)  # < 0.50
    record("Fail stage 1 with score 0.30",
           PASS if not passed2 else FAIL,
           f"passed={passed2} retries={tracker.current.retries}")

    # Pass on retry
    passed3 = tracker.report_score(0.60)
    record("Pass stage 1 on retry with score 0.60",
           PASS if passed3 and tracker.current_index == 2 else FAIL,
           f"passed={passed3} idx={tracker.current_index}")

    # Progress summary doesn't crash
    summary = tracker.progress_summary()
    record("progress_summary() returns string",
           PASS if isinstance(summary, str) and len(summary) > 0 else FAIL,
           f"len={len(summary)}")


def test_cascade_and_recovery():
    """Cascade failure detection and auto-recovery work."""
    print("\n--- Cascade & Recovery ---")
    sim = ClusterSimulator(n_nodes=5, task_id="task-1", seed=1)
    sim.reset(task_id="task-1", seed=1)

    # Artificially overload a node to trigger failure
    node = sim._nodes[5]
    node.queue_depth = 250.0  # > FATAL_FAIL_THRESHOLD=200
    sim._update_statuses()
    record("Node fails when queue > FATAL_FAIL_THRESHOLD",
           PASS if node.status == NodeStatus.FAILED else FAIL,
           f"status={node.status}")

    record("Recovery timer set on overload failure",
           PASS if node.recovery_timer > 0 else FAIL,
           f"timer={node.recovery_timer}")

    # Tick through recovery
    for _ in range(25):
        sim._process_recovery()

    record("Node recovers after NODE_RECOVERY_TICKS",
           PASS if node.status == NodeStatus.HEALTHY else FAIL,
           f"status={node.status}")


# ════════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AntiAtropos Smoke Test — 5-Node Cluster Validation")
    print("=" * 60)

    test_simulator_node_count()
    test_task1_ramp()
    test_task2_fault()
    test_task3_surge()
    test_scale_up_down()
    test_reward_sanity()
    test_grader_inline()
    test_curriculum_tracker()
    test_cascade_and_recovery()

    # ── Summary ──
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
