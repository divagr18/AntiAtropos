import importlib.util
import sys
from pathlib import Path


def _load_stability_module():
    path = Path(__file__).resolve().parents[1] / "AntiAtropos" / "stability.py"
    spec = importlib.util.spec_from_file_location("antiatropos_stability_norm", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load stability module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stability = _load_stability_module()


def test_normalize_reward_monotonic():
    r1 = stability.normalize_reward(-100.0)
    r2 = stability.normalize_reward(0.0)
    r3 = stability.normalize_reward(100.0)
    assert r1 < r2 < r3


def test_normalize_reward_bounds():
    for val in [-1e9, -1000.0, 0.0, 1000.0, 1e9]:
        out = stability.normalize_reward(val)
        assert 0.0 <= out <= 1.0


def test_normalize_reward_numerical_stability_with_tiny_temperature():
    out_neg = stability.normalize_reward(-1e6, midpoint=0.0, temperature=1e-12, eps=1e-8)
    out_pos = stability.normalize_reward(1e6, midpoint=0.0, temperature=1e-12, eps=1e-8)
    assert 0.0 <= out_neg <= 1.0
    assert 0.0 <= out_pos <= 1.0
    assert out_neg < 0.01
    assert out_pos > 0.99
