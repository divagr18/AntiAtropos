import os
import random
import logging
from typing import Any, Dict, List, Optional
import requests
from pydantic import BaseModel
from .mapping import MetricMapper

logger = logging.getLogger("antiatropos.telemetry")

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
        self.metric_mapper = MetricMapper(
            mapping_strategy=os.getenv("ANTIATROPOS_METRIC_AGGREGATION", "sum")
        )

        self.request_rate_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_REQUEST_RATE",
            'sum(rate(http_requests_total[1m])) by (pod)'
        )
        self.latency_ms_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_LATENCY_MS",
            'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (pod, le)) * 1000'
        )
        self.error_rate_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_ERROR_RATE",
            'sum(rate(http_requests_total{status=~"5.."}[1m])) by (pod) / clamp_min(sum(rate(http_requests_total[1m])) by (pod), 1)'
        )
        self.cpu_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_CPU",
            'avg(rate(container_cpu_usage_seconds_total[1m])) by (pod)'
        )
        self.queue_depth_query = os.getenv(
            "ANTIATROPOS_PROM_QUERY_QUEUE_DEPTH",
            'sum(queue_depth) by (pod)'
        )
        
    def fetch_latest_metrics(self, node_ids: List[str]) -> Dict[str, Any]:
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

    def _fetch_real_metrics(self, node_ids: List[str]) -> Dict[str, Any]:
        """Fetches node telemetry from Prometheus instant queries."""
        metrics: Dict[str, Any] = {}
        saw_any_real_signal = False

        req_by_node = self._collect_metric_values("request_rate", self.request_rate_query, node_ids)
        lat_by_node = self._collect_metric_values("latency_ms", self.latency_ms_query, node_ids)
        err_by_node = self._collect_metric_values("error_rate", self.error_rate_query, node_ids)
        cpu_by_node = self._collect_metric_values("cpu_utilization", self.cpu_query, node_ids)
        q_by_node = self._collect_metric_values("queue_depth", self.queue_depth_query, node_ids)

        for node_id in node_ids:
            req_rate = req_by_node.get(node_id)
            lat_ms = lat_by_node.get(node_id)
            err_rate = err_by_node.get(node_id)
            cpu = cpu_by_node.get(node_id)
            q_depth = q_by_node.get(node_id)

            if any(v is not None for v in (req_rate, lat_ms, err_rate, cpu, q_depth)):
                saw_any_real_signal = True
            else:
                # No usable sample for this node this cycle; skip reconciliation
                # so simulator dynamics are preserved instead of being collapsed
                # toward zero by synthetic defaults.
                continue

            node_payload: Dict[str, float] = {}
            if lat_ms is not None:
                node_payload["latency_ms"] = float(lat_ms)
            if req_rate is not None:
                node_payload["request_rate"] = float(req_rate)
            if err_rate is not None:
                node_payload["error_rate"] = max(0.0, min(1.0, float(err_rate)))
            if cpu is not None:
                node_payload["cpu_utilization"] = max(0.0, min(1.0, float(cpu)))
            if q_depth is not None:
                node_payload["queue_depth"] = max(0.0, float(q_depth))
            if node_payload:
                metrics[node_id] = node_payload

        if self.strict_real and not saw_any_real_signal:
            raise RuntimeError("Prometheus returned no usable real telemetry for requested node IDs.")

        if not saw_any_real_signal:
            logger.warning(
                "No per-node Prometheus samples found for configured queries; "
                "skipping telemetry reconciliation for this step."
            )

        return metrics

    def _collect_metric_values(
        self,
        metric_name: str,
        query: str,
        node_ids: List[str],
    ) -> Dict[str, Optional[float]]:
        """
        Collect node values for one logical metric.

        If query contains "{node_id}", execute per-node scalar queries.
        Otherwise run one vector query and aggregate labels via MetricMapper.
        """
        out: Dict[str, Optional[float]] = {node_id: None for node_id in node_ids}

        if "{node_id}" in query:
            for node_id in node_ids:
                out[node_id] = self._query_scalar(query.format(node_id=node_id))
            return out

        samples = self._query_vector(query)
        raw_metrics: List[Dict[str, Any]] = []
        for sample in samples:
            labels = sample.get("metric")
            value = sample.get("value")
            if not isinstance(labels, dict):
                continue
            if not value or len(value) < 2:
                continue
            raw_metrics.append(
                {
                    "metric_name": metric_name,
                    "labels": labels,
                    "value": value[1],
                }
            )

        by_node = self.metric_mapper.aggregate_node_metrics(raw_metrics)
        for node_id in node_ids:
            metric_map = by_node.get(node_id, {})
            value = metric_map.get(metric_name)
            out[node_id] = value if value is not None else None

        return out

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

    def _query_vector(self, promql: str) -> List[Dict[str, Any]]:
        """Runs a Prometheus instant query and returns the full vector result list."""
        if not self.url:
            return []

        response = requests.get(
            f"{self.url.rstrip('/')}/api/v1/query",
            params={"query": promql},
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "success":
            return []
        result = payload.get("data", {}).get("result", [])
        return result if isinstance(result, list) else []

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
