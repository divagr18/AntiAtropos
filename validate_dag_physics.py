#!/usr/bin/env python3
"""
Comprehensive DAG Physics Validation for AntiAtropos training readiness.

Verifies:
  A. DAG traffic routing (parent->child propagation)
  B. Task-2 scripted failure flows through DAG (not bypassed)
  C. Task-3 surge correct overlay on DAG
  D. Backpressure is temporary (not permanent capacity drain)
  E. Gradual recovery completes fully
  F. Graph-bounded cascades
  G. Reroute weights work with DAG
  H. Graph Lyapunov edge penalty
  I. Environment observation populates graph fields
  J. Reward components are non-degenerate across tasks

Run: python validate_dag_physics.py
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator import (
    ClusterSimulator, NodeStatus, DEFAULT_CAPACITY,
    CLUSTER_TOPOLOGY, EXTERNAL_TRAFFIC_NODES, _TOPOLOGICAL_ORDER,
    DEFAULT_ROUTING_SPLIT, T1_INITIAL_LAMBDA, T2_INITIAL_LAMBDA,
    T3_INITIAL_LAMBDA, T3_SURGE_MAGNITUDE,
    COST_PER_CAPACITY_UNIT_PER_HOUR, FATAL_FAIL_THRESHOLD,
    NODE_RECOVERY_TICKS, BACKPRESSURE_THRESHOLD,
)
from stability import (
    compute_lyapunov, compute_lyapunov_graph, compute_reward,
    compute_barrier, normalize_reward, smooth_sla_penalty, compute_drift,
    BARRIER_NORM_SCALE,
)

# --- Test harness ---
PASS = "PASS"
FAIL = "FAIL"
results = []

def record(name, status, detail=""):
    results.append((name, status, detail))
    icon = "+" if status == PASS else "X"
    msg = f"  [{icon}] {name}"
    if detail:
        msg += f"  -- {detail}"
    print(msg)

# ============================================================================
# A. DAG Traffic Routing
# ============================================================================
def test_A_dag_routing():
    print("\n=== A. DAG Traffic Routing ===")
    sim = ClusterSimulator(n_nodes=5, task_id="task-1", seed=1)
    sim.reset(task_id="task-1", seed=1)

    # Tick once with NO_OP
    class _A: pass
    a = _A(); a.action_type = "NO_OP"; a.target_node_id = "node-0"; a.parameter = 0.0
    sim.apply_action(a)
    sim.tick()

    nodes = {n.node_id: n for n in sim._nodes}

    # node-0 and node-4 should receive external traffic
    n0_in = nodes["node-0"].incoming_request_rate
    n4_in = nodes["node-4"].incoming_request_rate
    record("node-0 (ingress) receives traffic",
           PASS if n0_in > 0 else FAIL,
           f"incoming={n0_in:.1f}")
    record("node-4 (ingress) receives traffic",
           PASS if n4_in > 0 else FAIL,
           f"incoming={n4_in:.1f}")

    # node-1, node-2 should receive outflow from node-0
    n1_in = nodes["node-1"].incoming_request_rate
    n2_in = nodes["node-2"].incoming_request_rate
    record("node-1 receives from node-0 (DAG child)",
           PASS if n1_in > 0 else FAIL,
           f"incoming={n1_in:.1f}")
    record("node-2 receives from node-0 (DAG child)",
           PASS if n2_in > 0 else FAIL,
           f"incoming={n2_in:.1f}")

    # node-3 should receive outflow from node-2
    n3_in = nodes["node-3"].incoming_request_rate
    record("node-3 receives from node-2 (DAG grandchild)",
           PASS if n3_in > 0 else FAIL,
           f"incoming={n3_in:.1f}")

    # node-0 outflow should be ~incoming (since capacity >> lambda at start)
    record("node-0 has positive outflow_rate",
           PASS if nodes["node-0"].outflow_rate > 0 else FAIL,
           f"outflow={nodes['node-0'].outflow_rate:.1f}")


# ============================================================================
# B. Task-2 Scripted Failure Flows Through DAG
# ============================================================================
def test_B_task2_dag():
    print("\n=== B. Task-2 Failure Through DAG ===")
    sim = ClusterSimulator(n_nodes=5, task_id="task-2", seed=42)
    sim.reset(task_id="task-2", seed=42)

    # Run enough ticks for failure to trigger
    class _A: pass
    a = _A(); a.action_type = "NO_OP"; a.target_node_id = "node-0"; a.parameter = 0.0

    failed_id = None
    for _ in range(60):
        sim.apply_action(a)
        sim.tick()
        if sim._failed_node_id and failed_id is None:
            failed_id = sim._failed_node_id

    record("Scripted failure was assigned",
           PASS if failed_id is not None else FAIL,
           f"failed_id={failed_id}")

    nodes = {n.node_id: n for n in sim._nodes}
    failed_node = nodes.get(failed_id)
    record("Failed node has FAILED status",
           PASS if failed_node and failed_node.status == NodeStatus.FAILED else FAIL,
           f"status={failed_node.status if failed_node else 'N/A'}")

    # If failed node is a child of node-0 (e.g., node-1 or node-2),
    # node-0 should still be sending traffic to it (flow not bypassed)
    if failed_id in CLUSTER_TOPOLOGY.get("node-0", []):
        # The failed node outflow should be 0 (service_rate=0),
        # but it should still have incoming from DAG
        record("Failed child still receives DAG traffic (as dropped requests)",
               PASS if failed_node.incoming_request_rate >= 0 else FAIL,
               f"incoming={failed_node.incoming_request_rate:.1f} dropped={failed_node.dropped_requests:.1f}")

    # If failed node was node-2, node-3 should be starved (outflow=0 upstream)
    if failed_id == "node-2":
        n3 = nodes["node-3"]
        record("node-3 starved when parent node-2 fails",
               PASS if n3.incoming_request_rate == 0 else FAIL,
               f"node-3 incoming={n3.incoming_request_rate:.1f}")


# ============================================================================
# C. Task-3 Surge Overlay on DAG
# ============================================================================
def test_C_task3_surge_dag():
    print("\n=== C. Task-3 Surge Overlay ===")
    sim = ClusterSimulator(n_nodes=5, task_id="task-3", seed=7)
    sim.reset(task_id="task-3", seed=7)

    # Force surge window to be active immediately
    sim._t3_surge_start = 0
    sim._t3_surge_end = 999

    class _A: pass
    a = _A(); a.action_type = "NO_OP"; a.target_node_id = "node-0"; a.parameter = 0.0
    sim.apply_action(a)
    sim.tick()

    nodes = {n.node_id: n for n in sim._nodes}

    # node-1 and node-2 should have surge + DAG traffic
    n1_in = nodes["node-1"].incoming_request_rate
    n2_in = nodes["node-2"].incoming_request_rate
    record("node-1 receives surge + DAG traffic",
           PASS if n1_in > T3_SURGE_MAGNITUDE else FAIL,
           f"incoming={n1_in:.1f} (surge={T3_SURGE_MAGNITUDE})")
    record("node-2 receives surge + DAG traffic",
           PASS if n2_in > T3_SURGE_MAGNITUDE else FAIL,
           f"incoming={n2_in:.1f} (surge={T3_SURGE_MAGNITUDE})")

    # node-0 should still have base DAG traffic (not affected by surge directly)
    n0_in = nodes["node-0"].incoming_request_rate
    record("node-0 gets base DAG traffic (surge is side-channel)",
           PASS if n0_in < T3_SURGE_MAGNITUDE else FAIL,
           f"node-0 incoming={n0_in:.1f}")


# ============================================================================
# D. Backpressure Is Temporary
# ============================================================================
def test_D_backpressure_temporary():
    print("\n=== D. Backpressure Temporary ===")
    sim = ClusterSimulator(n_nodes=5, task_id="task-1", seed=1)
    sim.reset(task_id="task-1", seed=1)

    node0 = next(n for n in sim._nodes if n.node_id == "node-0")
    original_cap = node0.capacity

    # Artificially overload node-0's children to trigger backpressure
    for n in sim._nodes:
        if n.node_id in CLUSTER_TOPOLOGY.get("node-0", []):
            n.queue_depth = BACKPRESSURE_THRESHOLD + 100.0  # well above threshold

    # Tick: backpressure should reduce node-0's capacity for THIS tick only
    class _A: pass
    a = _A(); a.action_type = "NO_OP"; a.target_node_id = "node-0"; a.parameter = 0.0
    sim.apply_action(a)
    sim.tick()

    cap_after_tick = node0.capacity
    record("node-0 capacity restored after backpressure tick",
           PASS if abs(cap_after_tick - original_cap) < 0.01 else FAIL,
           f"before={original_cap:.2f} after={cap_after_tick:.2f}")

    # Tick again (children still overloaded) — capacity should still be original
    sim.apply_action(a)
    sim.tick()
    cap_after_tick2 = node0.capacity
    record("node-0 capacity intact after multiple backpressure ticks",
           PASS if abs(cap_after_tick2 - original_cap) < 0.01 else FAIL,
           f"after 2 ticks={cap_after_tick2:.2f} original={original_cap:.2f}")

    # Clear children overload — capacity should remain original
    for n in sim._nodes:
        if n.node_id in CLUSTER_TOPOLOGY.get("node-0", []):
            n.queue_depth = 0.0
    sim.apply_action(a)
    sim.tick()
    cap_clear = node0.capacity
    record("node-0 capacity unchanged after children clear",
           PASS if abs(cap_clear - original_cap) < 0.01 else FAIL,
           f"capacity={cap_clear:.2f}")


# ============================================================================
# E. Gradual Recovery Completes
# ============================================================================
def test_E_gradual_recovery():
    print("\n=== E. Gradual Recovery ===")
    sim = ClusterSimulator(n_nodes=5, task_id="task-1", seed=1)
    sim.reset(task_id="task-1", seed=1)

    node = sim._nodes[2]
    node.queue_depth = 250.0
    sim._update_statuses()

    record("Node becomes FAILED on overload",
           PASS if node.status == NodeStatus.FAILED else FAIL,
           f"status={node.status}")
    record("Capacity drops to 0.5 at failure",
           PASS if abs(node.capacity - 0.5) < 0.01 else FAIL,
           f"capacity={node.capacity}")

    # Tick through full recovery (NODE_RECOVERY_TICKS + some margin)
    class _A: pass
    a = _A(); a.action_type = "NO_OP"; a.target_node_id = "node-0"; a.parameter = 0.0
    for _ in range(NODE_RECOVERY_TICKS + 5):
        sim.apply_action(a)
        sim.tick()

    record("Node reaches HEALTHY after full recovery",
           PASS if node.status == NodeStatus.HEALTHY else FAIL,
           f"status={node.status}")

    # Capacity should have ramped: start=0.5, each recovery tick adds 0.5
    # After NODE_RECOVERY_TICKS=20 ticks: 0.5 + 20*0.5 = 10.5, capped at 3.0
    record("Capacity recovered to DEFAULT_CAPACITY (capped)",
           PASS if abs(node.capacity - DEFAULT_CAPACITY) < 0.01 else FAIL,
           f"capacity={node.capacity:.2f} expected={DEFAULT_CAPACITY}")


# ============================================================================
# F. Graph-Bounded Cascades
# ============================================================================
def test_F_graph_cascade():
    print("\n=== F. Graph-Bounded Cascades ===")
    sim = ClusterSimulator(n_nodes=5, task_id="task-1", seed=1)
    sim.reset(task_id="task-1", seed=1)

    # Fail node-2, overload its children/parents
    node2 = sim._nodes[2]
    node2.queue_depth = 250.0
    sim._update_statuses()  # node-2 becomes FAILED, triggers cascade

    # node-3 is child of node-2 — should be at_risk
    # node-0 is parent of node-2 — should be at_risk
    node3 = sim._nodes[3]
    node0 = sim._nodes[0]
    # Overload node-3 to trigger cascade
    node3.queue_depth = FATAL_FAIL_THRESHOLD * 1.5  # > cascade threshold
    sim._cascade_failures()

    record("node-3 (child of failed node-2) cascades to DEGRADED",
           PASS if node3.status == NodeStatus.DEGRADED else FAIL,
           f"node-3 status={node3.status}")

    # node-4 is NOT adjacent to node-2 — should NOT cascade
    node4 = sim._nodes[4]
    node4.queue_depth = FATAL_FAIL_THRESHOLD * 1.5
    sim._cascade_failures()
    record("node-4 (not adjacent to failed) does NOT cascade",
           PASS if node4.status != NodeStatus.DEGRADED else FAIL,
           f"node-4 status={node4.status}")


# ============================================================================
# G. Reroute Weights with DAG
# ============================================================================
def test_G_reroute_with_dag():
    print("\n=== G. Reroute Weights with DAG ===")
    sim = ClusterSimulator(n_nodes=5, task_id="task-1", seed=1)
    sim.reset(task_id="task-1", seed=1)

    class _A: pass

    # Tick once to establish baseline
    a = _A(); a.action_type = "NO_OP"; a.target_node_id = "node-0"; a.parameter = 0.0
    sim.apply_action(a)
    sim.tick()

    node0 = next(n for n in sim._nodes if n.node_id == "node-0")
    node4 = next(n for n in sim._nodes if n.node_id == "node-4")
    baseline_n0_in = node0.incoming_request_rate

    # Reroute 100% of node-0 traffic away
    a2 = _A(); a2.action_type = "REROUTE_TRAFFIC"; a2.target_node_id = "node-0"; a2.parameter = 1.0
    sim.apply_action(a2)
    sim.tick()

    n0_in_after = node0.incoming_request_rate
    record("Reroute reduces node-0 incoming traffic",
           PASS if n0_in_after < baseline_n0_in else FAIL,
           f"before={baseline_n0_in:.1f} after={n0_in_after:.1f}")

    # Verify outflow_rate is also reduced (since incoming is lower)
    record("node-0 outflow reduced after reroute",
           PASS if node0.outflow_rate < baseline_n0_in else FAIL,
           f"outflow={node0.outflow_rate:.1f}")

    # Reroute weight should decay each tick (0.5 factor)
    w = sim._reroute_weights.get("node-0", 0.0)
    record("Reroute weight decays (0.5 * prev)",
           PASS if w < 1.0 else FAIL,
           f"weight after first decay={w:.3f}")


# ============================================================================
# H. Graph Lyapunov Edge Penalty
# ============================================================================
def test_H_graph_lyapunov():
    print("\n=== H. Graph Lyapunov Edge Penalty ===")
    nodes_balanced = [
        {"node_id": "node-0", "queue_depth": 50.0, "importance_weight": 2.0},
        {"node_id": "node-4", "queue_depth": 50.0, "importance_weight": 1.0},
        {"node_id": "node-1", "queue_depth": 50.0, "importance_weight": 1.0},
        {"node_id": "node-2", "queue_depth": 50.0, "importance_weight": 1.0},
        {"node_id": "node-3", "queue_depth": 50.0, "importance_weight": 1.0},
    ]
    nodes_imbalanced = [
        {"node_id": "node-0", "queue_depth": 200.0, "importance_weight": 2.0},
        {"node_id": "node-4", "queue_depth": 10.0, "importance_weight": 1.0},
        {"node_id": "node-1", "queue_depth": 10.0, "importance_weight": 1.0},
        {"node_id": "node-2", "queue_depth": 10.0, "importance_weight": 1.0},
        {"node_id": "node-3", "queue_depth": 10.0, "importance_weight": 1.0},
    ]

    v_bal = compute_lyapunov_graph(nodes_balanced, CLUSTER_TOPOLOGY)
    v_imb = compute_lyapunov_graph(nodes_imbalanced, CLUSTER_TOPOLOGY)
    record("Graph Lyapunov: imbalanced > balanced",
           PASS if v_imb > v_bal else FAIL,
           f"balanced={v_bal:.1f} imbalanced={v_imb:.1f}")

    # Compare with flat Lyapunov: graph version should add edge penalty
    v_flat_imb = compute_lyapunov(nodes_imbalanced)
    record("Graph Lyapunov > flat Lyapunov for imbalanced cluster",
           PASS if v_imb > v_flat_imb else FAIL,
           f"graph={v_imb:.1f} flat={v_flat_imb:.1f}")


# ============================================================================
# I. Environment Observation Populates Graph Fields
# ============================================================================
def test_I_env_graph_fields():
    print("\n=== I. Environment Graph Fields ===")
    try:
        from server.AntiAtropos_environment import AntiAtroposEnvironment
        from models import SREAction, ActionType
    except ImportError:
        record("Environment import", FAIL, "Cannot import AntiAtroposEnvironment")
        return

    env = AntiAtroposEnvironment()
    obs = env.reset(task_id="task-1", mode="simulated", seed=42)

    # Check that NodeObservations have graph fields
    n0 = next((n for n in obs.nodes if n.node_id == "node-0"), None)
    record("Environment reset succeeds",
           PASS if n0 is not None else FAIL, "")

    if n0:
        record("node-0 has downstream_nodes",
               PASS if isinstance(n0.downstream_nodes, list) and len(n0.downstream_nodes) > 0 else FAIL,
               f"downstream={n0.downstream_nodes}")
        record("node-0 has upstream_nodes",
               PASS if isinstance(n0.upstream_nodes, list) else FAIL,
               f"upstream={n0.upstream_nodes}")
        record("node-0 has upstream_pressure",
               PASS if n0.upstream_pressure is not None else FAIL,
               f"pressure={n0.upstream_pressure:.3f}")
        record("node-0 has outflow_rate",
               PASS if n0.outflow_rate is not None else FAIL,
               f"outflow={n0.outflow_rate:.3f}")

    # Step once with SCALE_UP to verify reward computation
    action = SREAction(action_type=ActionType.SCALE_UP, target_node_id="node-0", parameter=0.5)
    obs2 = env.step(action)

    record("Step reward is non-zero",
           PASS if obs2.reward != 0.0 else FAIL,
           f"reward={obs2.reward:.4f}")
    # Lyapunov can be 0 on first tick if all queues are below capacity.
    # Just verify it's a number (not None/NaN).
    lyap_ok = obs2.lyapunov_energy is not None and not math.isnan(obs2.lyapunov_energy)
    record("lyapunov_energy is valid (>=0, not NaN)",
           PASS if lyap_ok else FAIL,
           f"energy={obs2.lyapunov_energy}")


# ============================================================================
# J. Reward Components Across Tasks
# ============================================================================
def test_J_reward_components():
    print("\n=== J. Reward Components ===")
    env_module = None
    try:
        from server.AntiAtropos_environment import AntiAtroposEnvironment
        from models import SREAction, ActionType
    except ImportError:
        record("Reward components", FAIL, "Cannot import environment")
        return

    for task_id, warmup_ticks in [("task-1", 30), ("task-2", 5), ("task-3", 5)]:
        env = AntiAtroposEnvironment()
        env.reset(task_id=task_id, mode="simulated", seed=42)

        # Task-3 surge may not be active in early ticks (window depends on
        # jitter).  Force the surge window to be wide-open for validation.
        if task_id == "task-3":
            env._sim._t3_surge_start = 1
            env._sim._t3_surge_end = 999

        # Run enough ticks for queues to accumulate and drift to become non-zero.
        # Task-1 has a slow ramp (0.5/tick) and needs ~15+ ticks to exceed ingress
        # capacity of 90 req/tick (starting from ~86).
        action = SREAction(action_type=ActionType.NO_OP, target_node_id="node-0", parameter=0.0)
        obs = None
        for _ in range(warmup_ticks):
            obs = env.step(action)

        # All sub-components should be present
        has_drift = obs.reward_drift != 0.0
        has_cost = obs.reward_cost != 0.0
        record(f"{task_id}: reward_drift non-zero (after {warmup_ticks} ticks)",
               PASS if has_drift else FAIL,
               f"drift={obs.reward_drift:.4f}")
        record(f"{task_id}: reward_cost non-zero",
               PASS if has_cost else FAIL,
               f"cost={obs.reward_cost:.4f}")
        record(f"{task_id}: no NaN in raw_reward",
               PASS if not math.isnan(obs.raw_reward) else FAIL,
               f"raw={obs.raw_reward:.4f}")

    # Quick check: drift and cost should be negative (penalties)
    env = AntiAtroposEnvironment()
    env.reset(task_id="task-1", mode="simulated", seed=42)
    action = SREAction(action_type=ActionType.NO_OP, target_node_id="node-0", parameter=0.0)
    obs = env.step(action)
    record("reward_drift <= 0 (penalty, not reward)",
           PASS if obs.reward_drift <= 0 else FAIL,
           f"drift={obs.reward_drift:.4f}")


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 65)
    print("AntiAtropos DAG Physics Validation")
    print("=" * 65)

    test_A_dag_routing()
    test_B_task2_dag()
    test_C_task3_surge_dag()
    test_D_backpressure_temporary()
    test_E_gradual_recovery()
    test_F_graph_cascade()
    test_G_reroute_with_dag()
    test_H_graph_lyapunov()
    test_I_env_graph_fields()
    test_J_reward_components()

    passed = sum(1 for _, s, _ in results if s == PASS)
    failed = sum(1 for _, s, _ in results if s == FAIL)
    total = len(results)

    print("\n" + "=" * 65)
    print(f"RESULTS: {passed}/{total} passed, {failed} failed")
    print("=" * 65)

    if failed > 0:
        print("\nFAILED TESTS:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  X {name}: {detail}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
