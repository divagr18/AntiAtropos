import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = ROOT / "AntiAtropos"


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = pkg


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


def test_metric_mapper_aggregates_pod_labels_to_node():
    _ensure_package("AntiAtropos", PKG_DIR)
    _ensure_package("AntiAtropos.telemetry", PKG_DIR / "telemetry")
    mapping_mod = _load_pkg_module("telemetry.mapping", "telemetry/mapping.py")

    mapper = mapping_mod.MetricMapper(mapping_strategy="sum")
    mapper.node_mapping = {"payments-pod-a": "node-0", "payments-pod-b": "node-0"}

    aggregated = mapper.aggregate_node_metrics(
        [
            {"metric_name": "request_rate", "labels": {"pod": "payments-pod-a"}, "value": 10.0},
            {"metric_name": "request_rate", "labels": {"pod": "payments-pod-b"}, "value": 15.0},
            {"metric_name": "cpu_utilization", "labels": {"pod": "payments-pod-a"}, "value": 0.3},
        ]
    )

    assert aggregated["node-0"]["request_rate"] == 25.0
    assert aggregated["node-0"]["cpu_utilization"] == 0.3


def test_prometheus_client_vector_queries_are_aggregated_by_mapper():
    _ensure_package("AntiAtropos", PKG_DIR)
    _ensure_package("AntiAtropos.telemetry", PKG_DIR / "telemetry")
    _load_pkg_module("telemetry.mapping", "telemetry/mapping.py")
    prom_mod = _load_pkg_module("telemetry.prometheus_client", "telemetry/prometheus_client.py")

    client = prom_mod.PrometheusClient(prometheus_url="http://prometheus.local")
    client.metric_mapper.node_mapping = {"payments-pod-a": "node-0"}

    def fake_query_vector(promql: str):
        if "http_requests_total" in promql and 'status=~"5.."' not in promql:
            return [{"metric": {"pod": "payments-pod-a"}, "value": [0, "12.0"]}]
        if "http_request_duration_seconds_bucket" in promql:
            return [{"metric": {"pod": "payments-pod-a"}, "value": [0, "210.0"]}]
        if 'status=~"5.."' in promql:
            return [{"metric": {"pod": "payments-pod-a"}, "value": [0, "0.03"]}]
        if "container_cpu_usage_seconds_total" in promql:
            return [{"metric": {"pod": "payments-pod-a"}, "value": [0, "0.7"]}]
        if "queue_depth" in promql:
            return [{"metric": {"pod": "payments-pod-a"}, "value": [0, "5.0"]}]
        return []

    client._query_vector = fake_query_vector

    out = client._fetch_real_metrics(["node-0"])
    rec = out["node-0"]

    assert rec.request_rate == 12.0
    assert rec.latency_ms == 210.0
    assert rec.error_rate == 0.03
    assert rec.cpu_utilization == 0.7
    assert rec.queue_depth == 5.0
