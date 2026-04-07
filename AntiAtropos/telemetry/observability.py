import threading
from typing import Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest
except ImportError:  # pragma: no cover
    Counter = Gauge = Histogram = None

    def generate_latest() -> bytes:  # type: ignore[override]
        return b"# prometheus_client not installed\n"


def _enabled() -> bool:
    return Counter is not None and Gauge is not None and Histogram is not None


class ObservabilityTracker:
    """Prometheus metrics for action/reward/health monitoring."""

    def __init__(self):
        self._lock = threading.Lock()
        self._is_enabled = _enabled()
        if not self._is_enabled:
            return

        self.steps_total = Counter(
            "antiatropos_steps_total",
            "Total environment steps",
            ["task_id", "mode"],
        )
        self.actions_total = Counter(
            "antiatropos_actions_total",
            "Actions executed by type/target/status",
            ["task_id", "mode", "action_type", "target_node_id", "ack_class"],
        )
        self.executor_errors_total = Counter(
            "antiatropos_executor_errors_total",
            "Executor errors by code",
            ["mode", "error_code"],
        )
        self.executor_latency_ms = Histogram(
            "antiatropos_executor_latency_ms",
            "Executor latency in milliseconds",
            ["mode"],
            buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000),
        )

        self.reward_gauge = Gauge(
            "antiatropos_reward",
            "Latest output reward value (depends on reward output mode)",
            ["task_id", "mode"],
        )
        self.reward_raw_gauge = Gauge(
            "antiatropos_reward_raw",
            "Latest raw reward value before normalization",
            ["task_id", "mode"],
        )
        self.reward_normalized_gauge = Gauge(
            "antiatropos_reward_normalized",
            "Latest normalized reward value in [0,1]",
            ["task_id", "mode"],
        )
        self.lyapunov_gauge = Gauge(
            "antiatropos_lyapunov_energy",
            "Latest Lyapunov energy",
            ["task_id", "mode"],
        )
        self.queue_gauge = Gauge(
            "antiatropos_total_queue_backlog",
            "Latest normalized total queue backlog",
            ["task_id", "mode"],
        )
        self.latency_gauge = Gauge(
            "antiatropos_average_latency_norm",
            "Latest normalized average latency",
            ["task_id", "mode"],
        )

    def record_step(
        self,
        task_id: str,
        mode: str,
        action_type: str,
        target_node_id: str,
        ack_status: str,
        reward_output: float,
        reward_raw: float,
        reward_normalized: float,
        lyapunov_energy: float,
        total_queue_backlog: float,
        average_latency_ms: float,
        executor_latency_ms: float,
        executor_error_code: str,
    ) -> None:
        if not self._is_enabled:
            return

        ack_class = self._classify_ack(ack_status)
        with self._lock:
            self.steps_total.labels(task_id=task_id, mode=mode).inc()
            self.actions_total.labels(
                task_id=task_id,
                mode=mode,
                action_type=action_type,
                target_node_id=target_node_id,
                ack_class=ack_class,
            ).inc()
            self.reward_gauge.labels(task_id=task_id, mode=mode).set(float(reward_output))
            self.reward_raw_gauge.labels(task_id=task_id, mode=mode).set(float(reward_raw))
            self.reward_normalized_gauge.labels(task_id=task_id, mode=mode).set(float(reward_normalized))
            self.lyapunov_gauge.labels(task_id=task_id, mode=mode).set(float(lyapunov_energy))
            self.queue_gauge.labels(task_id=task_id, mode=mode).set(float(total_queue_backlog))
            self.latency_gauge.labels(task_id=task_id, mode=mode).set(float(average_latency_ms))
            self.executor_latency_ms.labels(mode=mode).observe(max(0.0, float(executor_latency_ms)))
            if executor_error_code:
                self.executor_errors_total.labels(mode=mode, error_code=executor_error_code).inc()

    @staticmethod
    def _classify_ack(ack_status: str) -> str:
        status = str(ack_status)
        if status.startswith("Ack:") or status.startswith("success"):
            return "ack"
        if status.startswith("Rejected:"):
            return "rejected"
        if status.startswith("Error:"):
            return "error"
        return "unknown"


_TRACKER: Optional[ObservabilityTracker] = None


def get_observability_tracker() -> ObservabilityTracker:
    global _TRACKER
    if _TRACKER is None:
        _TRACKER = ObservabilityTracker()
    return _TRACKER


def render_prometheus_metrics() -> bytes:
    return generate_latest()
