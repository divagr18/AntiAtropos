import importlib.util
import sys
from pathlib import Path


def _load_simulator_module():
    sim_path = Path(__file__).resolve().parents[1] / "AntiAtropos" / "simulator.py"
    spec = importlib.util.spec_from_file_location("antiatropos_simulator", sim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load simulator module from {sim_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


simulator = _load_simulator_module()
ClusterSimulator = simulator.ClusterSimulator


def _step(sim, action_type: str, target_node_id: str = "node-0", parameter: float = 0.0):
    sim.apply_action(
        {
            "action_type": action_type,
            "target_node_id": target_node_id,
            "parameter": parameter,
        }
    )
    sim.tick()


def _incoming(sim, node_id: str) -> float:
    return next(n.incoming_request_rate for n in sim._nodes if n.node_id == node_id)


def _total_incoming(sim) -> float:
    return sum(n.incoming_request_rate for n in sim._nodes)


def _error_rate_from_drops(sim) -> float:
    total_incoming = _total_incoming(sim)
    if total_incoming <= 0:
        return 0.0
    total_drops = sum(n.dropped_requests for n in sim._nodes)
    return total_drops / total_incoming


def test_task2_reroute_can_offload_failed_node():
    sim = ClusterSimulator(task_id="task-2", seed=42)
    sim._t2_fail_tick = 1

    _step(sim, "NO_OP")
    failed = sim._failed_node_id
    assert failed is not None

    failed_before = _incoming(sim, failed)
    total_before = _total_incoming(sim)
    assert failed_before > 0.0

    _step(sim, "REROUTE_TRAFFIC", target_node_id=failed, parameter=1.0)

    failed_after = _incoming(sim, failed)
    total_after = _total_incoming(sim)

    assert failed_after == 0.0
    assert abs(total_after - total_before) < 1e-9


def test_reroute_on_healthy_node_still_offloads_as_before():
    sim = ClusterSimulator(task_id="task-2", seed=42)
    sim._t2_fail_tick = 999  # keep all nodes healthy during this test

    _step(sim, "NO_OP")
    target = "node-0"
    target_before = _incoming(sim, target)
    total_before = _total_incoming(sim)
    assert target_before > 0.0

    _step(sim, "REROUTE_TRAFFIC", target_node_id=target, parameter=1.0)

    target_after = _incoming(sim, target)
    total_after = _total_incoming(sim)

    assert target_after == 0.0
    assert abs(total_after - total_before) < 1e-9


def test_task2_failed_node_reroute_reduces_error_floor():
    sim = ClusterSimulator(task_id="task-2", seed=42)
    sim._t2_fail_tick = 1

    _step(sim, "NO_OP")
    failed = sim._failed_node_id
    assert failed is not None
    err_before = _error_rate_from_drops(sim)

    _step(sim, "REROUTE_TRAFFIC", target_node_id=failed, parameter=1.0)
    err_after = _error_rate_from_drops(sim)

    assert err_before > 0.0
    assert err_after < err_before
    assert err_after <= 0.05
