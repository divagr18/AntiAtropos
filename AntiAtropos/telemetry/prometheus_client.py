import os
import random
from typing import Dict, List, Optional
import requests
from pydantic import BaseModel

class TelemetryRecord(BaseModel):
    node_id: str
    latency_ms: float
    request_rate: float
    error_rate: float
    cpu_utilization: float
    queue_depth: float

class PrometheusClient:
    """
    Adapter to fetch and normalize metrics from Prometheus.
    Supports a mock mode for local development without a live cluster.
    """
    def __init__(self, prometheus_url: Optional[str] = None):
        # Use provided URL or env var, defaulting to mock if neither is found
        self.url = prometheus_url or os.getenv("PROMETHEUS_URL")
        self.is_mock = not self.url or self.url.lower() == "mock"
        self.timeout_s = float(os.getenv("ANTIATROPOS_PROM_TIMEOUT_S", "2.5"))
        self.strict_real = os.getenv("ANTIATROPOS_STRICT_REAL", "false").lower() == "true"

        self.request_rate_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_REQUEST_RATE",
            'sum(rate(http_requests_total{node_id="{node_id}"}[1m]))'
        )
        self.latency_ms_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_LATENCY_MS",
            'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{node_id="{node_id}"}[5m])) by (le)) * 1000'
        )
        self.error_rate_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_ERROR_RATE",
            'sum(rate(http_requests_total{node_id="{node_id}",status=~"5.."}[1m])) / clamp_min(sum(rate(http_requests_total{node_id="{node_id}"}[1m])), 1)'
        )
        self.cpu_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_CPU",
            'avg(rate(container_cpu_usage_seconds_total{pod=~".*{node_id}.*"}[1m]))'
        )
        self.queue_depth_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_QUEUE_DEPTH",
            'sum(queue_depth{node_id="{node_id}"})'
        )
        
    def fetch_latest_metrics(self, node_ids: List[str]) -> Dict[str, TelemetryRecord]:
        """
        Query Prometheus for the latest metrics for the given nodes.
        Returns a mapping from node_id to TelemetryRecord.
        """
        if self.is_mock:
            return self._generate_mock_metrics(node_ids)
        
        # Real implementation using queries
        try:
            return self._fetch_real_metrics(node_ids)
        except Exception:
            if self.strict_real:
                raise
            return self._generate_mock_metrics(node_ids)

    def _fetch_real_metrics(self, node_ids: List[str]) -> Dict[str, TelemetryRecord]:
        """Fetches node telemetry from Prometheus instant queries."""
        metrics: Dict[str, TelemetryRecord] = {}
        saw_any_real_signal = False

        for node_id in node_ids:
            req_rate = self._query_scalar(self.request_rate_query.format(node_id=node_id))
            lat_ms = self._query_scalar(self.latency_ms_query.format(node_id=node_id))
            err_rate = self._query_scalar(self.error_rate_query.format(node_id=node_id))
            cpu = self._query_scalar(self.cpu_query.format(node_id=node_id))
            q_depth = self._query_scalar(self.queue_depth_query.format(node_id=node_id))

            if any(v is not None for v in [req_rate, lat_ms, err_rate, cpu, q_depth]):
                saw_any_real_signal = True

            metrics[node_id] = TelemetryRecord(
                node_id=node_id,
                latency_ms=float(lat_ms if lat_ms is not None else 20.0),
                request_rate=float(req_rate if req_rate is not None else 0.0),
                error_rate=max(0.0, min(1.0, float(err_rate if err_rate is not None else 0.0))),
                cpu_utilization=max(0.0, min(1.0, float(cpu if cpu is not None else 0.0))),
                queue_depth=max(0.0, float(q_depth if q_depth is not None else 0.0)),
            )

        if self.strict_real and not saw_any_real_signal:
            raise RuntimeError("Prometheus returned no usable real telemetry for requested node IDs.")

        return metrics

    def _query_scalar(self, promql: str) -> Optional[float]:
        """Runs a scalar/vector Prometheus instant query and returns the first value."""
        if not self.url:
            return None

        response = requests.get(
            f"{self.url.rstrip('/')}/api/v1/query",
            params={"query": promql},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "success":
            return None

        result = payload.get("data", {}).get("result", [])
        if not result:
            return None

        value = result[0].get("value")
        if not value or len(value) < 2:
            return None

        try:
            return float(value[1])
        except (TypeError, ValueError):
            return None

    def _generate_mock_metrics(self, node_ids: List[str]) -> Dict[str, TelemetryRecord]:
        """Generates realistic-looking mock telemetry."""
        metrics = {}
        for nid in node_ids:
            metrics[nid] = TelemetryRecord(
                node_id=nid,
                latency_ms=random.uniform(20.0, 150.0),
                request_rate=random.uniform(10.0, 50.0),
                error_rate=random.uniform(0.0, 0.05),
                cpu_utilization=random.uniform(0.1, 0.8),
                queue_depth=random.uniform(0.0, 50.0)
            )
        return metrics
