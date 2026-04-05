import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "AntiAtropos"


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


def _install_openenv_stubs() -> None:
    if "openenv.core.env_server.interfaces" in sys.modules:
        return

    openenv_mod = types.ModuleType("openenv")
    core_mod = types.ModuleType("openenv.core")
    env_server_mod = types.ModuleType("openenv.core.env_server")
    interfaces_mod = types.ModuleType("openenv.core.env_server.interfaces")
    types_mod = types.ModuleType("openenv.core.env_server.types")

    class Environment:
        pass

    class State:
        def __init__(self, episode_id: str, step_count: int):
            self.episode_id = episode_id
            self.step_count = step_count

    interfaces_mod.Environment = Environment
    types_mod.State = State

    sys.modules["openenv"] = openenv_mod
    sys.modules["openenv.core"] = core_mod
    sys.modules["openenv.core.env_server"] = env_server_mod
    sys.modules["openenv.core.env_server.interfaces"] = interfaces_mod
    sys.modules["openenv.core.env_server.types"] = types_mod


def _load_pkg_module(module_name: str, relative_file: str):
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


def _bootstrap_environment_module():
    _install_openenv_stubs()
    _ensure_package("AntiAtropos", PKG_DIR)
    _ensure_package("AntiAtropos.control", PKG_DIR / "control")
    _ensure_package("AntiAtropos.server", PKG_DIR / "server")
    _ensure_package("AntiAtropos.telemetry", PKG_DIR / "telemetry")

    _load_pkg_module("models", "models.py")
    _load_pkg_module("simulator", "simulator.py")
    _load_pkg_module("stability", "stability.py")
    _load_pkg_module("telemetry.mapping", "telemetry/mapping.py")
    telemetry_prom_mod = _load_pkg_module("telemetry.prometheus_client", "telemetry/prometheus_client.py")
    telemetry_obs_mod = _load_pkg_module("telemetry.observability", "telemetry/observability.py")
    validation_mod = _load_pkg_module("control.validation", "control/validation.py")
    kube_mod = _load_pkg_module("control.kubernetes_executor", "control/kubernetes_executor.py")

    telemetry_pkg = sys.modules["AntiAtropos.telemetry"]
    telemetry_pkg.PrometheusClient = telemetry_prom_mod.PrometheusClient
    telemetry_pkg.TelemetryRecord = telemetry_prom_mod.TelemetryRecord
    telemetry_pkg.get_observability_tracker = telemetry_obs_mod.get_observability_tracker
    telemetry_pkg.render_prometheus_metrics = telemetry_obs_mod.render_prometheus_metrics

    control_pkg = sys.modules["AntiAtropos.control"]
    control_pkg.ActionValidator = validation_mod.ActionValidator
    control_pkg.KubernetesExecutor = kube_mod.KubernetesExecutor

    models_mod = _load_pkg_module("models", "models.py")
    env_mod = _load_pkg_module("server.AntiAtropos_environment", "server/AntiAtropos_environment.py")
    return env_mod, models_mod


def test_validator_enforces_task_bounds():
    _ensure_package("AntiAtropos", PKG_DIR)
    _ensure_package("AntiAtropos.control", PKG_DIR / "control")
    validation_mod = _load_pkg_module("control.validation", "control/validation.py")
    validator = validation_mod.ActionValidator()

    valid, _ = validator.validate("REROUTE_TRAFFIC", "node-3", 0.4, valid_targets=["node-3"])
    assert valid

    valid, error = validator.validate("REROUTE_TRAFFIC", "node-3", 1.2, valid_targets=["node-3"])
    assert not valid
    assert "parameter must be in [0.0, 1.0]" in error

    valid, error = validator.validate("NO_OP", "node-3", 0.5, valid_targets=["node-3"])
    assert not valid
    assert "NO_OP requires parameter=0.0" in error


def test_live_rejected_actions_do_not_mutate_simulator():
    env_mod, models_mod = _bootstrap_environment_module()

    env = env_mod.AntiAtroposEnvironment()
    env.reset(mode="live")
    env._executor.is_mock = False
    env._executor.execute = lambda *args, **kwargs: "Rejected: action blocked"

    action = models_mod.SREAction(action_type=models_mod.ActionType.SCALE_UP, target_node_id="node-0", parameter=1.0)
    env.step(action)

    node0 = next(n for n in env._sim._nodes if n.node_id == "node-0")
    assert len(node0.pending_capacity_queue) == 0


def test_live_mode_rejects_actions_without_real_executor():
    env_mod, models_mod = _bootstrap_environment_module()

    env = env_mod.AntiAtroposEnvironment()
    env.reset(mode="live")

    action = models_mod.SREAction(action_type=models_mod.ActionType.SCALE_UP, target_node_id="node-0", parameter=1.0)
    obs = env.step(action)

    assert "no real Kubernetes executor is configured" in env._action_ack_status
    assert obs.executor_error_code == "MODE_UNSUPPORTED"
    assert obs.action_id
    assert obs.executor_latency_ms == 0.0
    node0 = next(n for n in env._sim._nodes if n.node_id == "node-0")
    assert len(node0.pending_capacity_queue) == 0


def test_executor_requires_explicit_workload_mapping(monkeypatch):
    _ensure_package("AntiAtropos", PKG_DIR)
    _ensure_package("AntiAtropos.control", PKG_DIR / "control")
    kube_mod = _load_pkg_module("control.kubernetes_executor", "control/kubernetes_executor.py")

    monkeypatch.delenv("ANTIATROPOS_WORKLOAD_MAP", raising=False)
    monkeypatch.delenv("ANTIATROPOS_NODE_DEPLOYMENT_MAP", raising=False)

    executor = kube_mod.KubernetesExecutor(kubeconfig="C:/tmp/kubeconfig")
    with pytest.raises(ValueError, match="Missing workload mapping"):
        executor._resolve_workload_target("node-0")


def test_executor_loads_explicit_workload_mapping(monkeypatch):
    _ensure_package("AntiAtropos", PKG_DIR)
    _ensure_package("AntiAtropos.control", PKG_DIR / "control")
    kube_mod = _load_pkg_module("control.kubernetes_executor", "control/kubernetes_executor.py")

    monkeypatch.setenv(
        "ANTIATROPOS_WORKLOAD_MAP",
        '{"node-0":{"deployment":"payments-api","namespace":"sre-sandbox"}}',
    )
    monkeypatch.delenv("ANTIATROPOS_NODE_DEPLOYMENT_MAP", raising=False)

    executor = kube_mod.KubernetesExecutor(kubeconfig="C:/tmp/kubeconfig")
    namespace, deployment = executor._resolve_workload_target("node-0")

    assert namespace == "sre-sandbox"
    assert deployment == "payments-api"


def test_live_capability_accepts_enum_action_type(monkeypatch):
    _ensure_package("AntiAtropos", PKG_DIR)
    _ensure_package("AntiAtropos.control", PKG_DIR / "control")
    models_mod = _load_pkg_module("models", "models.py")
    kube_mod = _load_pkg_module("control.kubernetes_executor", "control/kubernetes_executor.py")

    monkeypatch.setenv("KUBECONFIG", "C:/tmp/kubeconfig")
    monkeypatch.setenv(
        "ANTIATROPOS_WORKLOAD_MAP",
        '{"node-0":{"deployment":"payments-api","namespace":"sre-sandbox"}}',
    )
    monkeypatch.delenv("ANTIATROPOS_NODE_DEPLOYMENT_MAP", raising=False)

    executor = kube_mod.KubernetesExecutor()
    err = executor.live_capability_error(models_mod.ActionType.SCALE_UP)
    assert err is None
