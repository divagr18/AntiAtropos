from typing import Dict, List, Any, Optional
import os
import json

class MetricMapper:
    """
    Utility for mapping Prometheus label sets into internal node IDs.
    In environments with many pods, we need to decide which pods to 
    aggregate for a given node_id.
    """
    def __init__(self, mapping_strategy: str = "sum"):
        self.strategy = mapping_strategy.lower()
        self.node_mapping = self._load_node_mapping()

    def _load_node_mapping(self) -> Dict[str, str]:
        """Loads label-value -> node_id mapping from env config."""
        raw = os.getenv("ANTIATROPOS_LABEL_NODE_MAP", "")
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except json.JSONDecodeError:
                pass

        # Safe default mapping for local demos.
        return {
            "payments": "node-0",
            "checkout": "node-1",
            "catalog": "node-2",
            "cart": "node-3",
            "auth": "node-4",
        }

    def _resolve_node_id(self, labels: Dict[str, Any]) -> Optional[str]:
        """Resolve internal node_id from a Prometheus sample labelset."""
        explicit = labels.get("node_id")
        if explicit:
            return str(explicit)

        for key in ("pod", "service", "app", "workload", "deployment", "instance"):
            label_value = labels.get(key)
            if label_value and str(label_value) in self.node_mapping:
                return self.node_mapping[str(label_value)]

        return None

    def _reduce(self, values: List[float]) -> float:
        if not values:
            return 0.0
        if self.strategy == "max":
            return max(values)
        if self.strategy == "mean":
            return sum(values) / len(values)
        # default: sum
        return sum(values)

    def aggregate_node_metrics(self, raw_metrics: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregates labeled metric samples into node-level telemetry.

        Expected sample shape:
        {
          "metric_name": "request_rate",
          "labels": {"pod": "web-node-1"},
          "value": 42.0
        }
        """
        bucket: Dict[str, Dict[str, List[float]]] = {}

        for sample in raw_metrics:
            metric_name = str(sample.get("metric_name", ""))
            labels = sample.get("labels") or {}
            value = sample.get("value")
            if not metric_name or not isinstance(labels, dict) or value is None:
                continue

            node_id = self._resolve_node_id(labels)
            if not node_id:
                continue

            try:
                val = float(value)
            except (TypeError, ValueError):
                continue

            bucket.setdefault(node_id, {}).setdefault(metric_name, []).append(val)

        aggregated: Dict[str, Dict[str, float]] = {}
        for node_id, metric_map in bucket.items():
            aggregated[node_id] = {
                metric_name: self._reduce(values)
                for metric_name, values in metric_map.items()
            }

        return aggregated
