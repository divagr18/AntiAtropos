import importlib.util
import random
import sys
from pathlib import Path


def _load_simulator_module():
    sim_path = Path(__file__).resolve().parents[1] / "AntiAtropos" / "simulator.py"
    spec = importlib.util.spec_from_file_location("antiatropos_simulator_seed", sim_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load simulator module from {sim_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


simulator = _load_simulator_module()
ClusterSimulator = simulator.ClusterSimulator


def _snapshot_randomized_profile(sim) -> tuple:
    return (
        sim._t1_ramp_slope,
        sim._t1_init_lambda,
        sim._t2_fail_tick,
        sim._t2_init_lambda,
        sim._t3_surge_start,
        sim._t3_surge_end,
    )


def test_default_seed_uses_none(monkeypatch):
    seen_seeds = []
    real_random_cls = random.Random

    class SpyRandom(real_random_cls):
        def __init__(self, seed=None):
            seen_seeds.append(seed)
            super().__init__(seed)

    monkeypatch.setattr(simulator.random, "Random", SpyRandom)
    ClusterSimulator(task_id="task-1")

    assert seen_seeds, "Expected ClusterSimulator to initialize RNG"
    assert seen_seeds[0] is None


def test_explicit_seed_replays_identical_randomization():
    sim_a = ClusterSimulator(task_id="task-1", seed=42)
    sim_b = ClusterSimulator(task_id="task-1", seed=42)

    assert _snapshot_randomized_profile(sim_a) == _snapshot_randomized_profile(sim_b)
