import importlib.util
import math
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "AntiAtropos"


def _ensure_package() -> None:
    if "AntiAtropos" in sys.modules:
        return
    pkg = types.ModuleType("AntiAtropos")
    pkg.__path__ = [str(PKG_DIR)]  # type: ignore[attr-defined]
    sys.modules["AntiAtropos"] = pkg


def _load_pkg_module(module_name: str, relative_file: str):
    _ensure_package()
    fq_name = f"AntiAtropos.{module_name}"
    if fq_name in sys.modules:
        return sys.modules[fq_name]

    path = PKG_DIR / relative_file
    spec = importlib.util.spec_from_file_location(fq_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {fq_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[fq_name] = module
    spec.loader.exec_module(module)
    return module


grader_mod = _load_pkg_module("grader", "grader.py")
stability_mod = _load_pkg_module("stability", "stability.py")


class _Obs:
    def __init__(self, payload: dict):
        self._payload = payload

    def model_dump(self) -> dict:
        return dict(self._payload)


def _grade_stability_for_energy(energy: float) -> float:
    grader = grader_mod.EpisodeGrader(task_id="task-1")
    obs = _Obs(
        {
            "average_latency_ms": 0.0,
            "error_rate": 0.0,
            "current_cost_per_hour": 0.75,
            "lyapunov_energy": energy,
            "sla_violations": 0,
        }
    )
    grader.record(obs)
    return grader.score().scores["stability"]


def test_stability_score_does_not_saturate_below_target_energy():
    low = _grade_stability_for_energy(250.0)
    medium = _grade_stability_for_energy(1000.0)
    near_target = _grade_stability_for_energy(1900.0)
    at_target = _grade_stability_for_energy(2000.0)

    assert 0.0 < at_target < near_target < medium < low < 1.0


def test_compute_reward_penalizes_only_step_violation():
    params = dict(v_prev=100.0, v_curr=120.0, cost=1.5, alpha=1.0, beta=1.0, gamma=2.5)
    r_ok = stability_mod.compute_reward(sla_violation_step=0, **params)
    r_bad = stability_mod.compute_reward(sla_violation_step=1, **params)

    assert math.isclose(r_bad, r_ok - 2.5, rel_tol=0.0, abs_tol=1e-12)


def test_weighted_lyapunov_penalizes_vip_more():
    vip_energy = stability_mod.compute_lyapunov(
        [
            {"queue_depth": 0.5, "importance_weight": 4.0},
            {"queue_depth": 0.5, "importance_weight": 1.0},
        ]
    )
    normal_energy = stability_mod.compute_lyapunov(
        [
            {"queue_depth": 0.5, "importance_weight": 1.0},
            {"queue_depth": 0.5, "importance_weight": 1.0},
        ]
    )

    assert vip_energy > normal_energy
