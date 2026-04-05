import time
import json
import logging
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SREAction, ClusterObservation, NodeObservation, NodeStatus, EnvironmentMode
    from ..simulator import ClusterSimulator, COST_PER_CAPACITY_UNIT_PER_HOUR
    from ..stability import compute_lyapunov, compute_reward
    from ..telemetry import PrometheusClient, get_observability_tracker
    from ..control import KubernetesExecutor, ActionValidator
except ImportError:
    from models import SREAction, ClusterObservation, NodeObservation, NodeStatus, EnvironmentMode  # type: ignore[no-redef]
    from simulator import ClusterSimulator, COST_PER_CAPACITY_UNIT_PER_HOUR  # type: ignore[no-redef]
    from stability import compute_lyapunov, compute_reward  # type: ignore[no-redef]
    from telemetry import PrometheusClient, get_observability_tracker  # type: ignore[no-redef]
    from control import KubernetesExecutor, ActionValidator  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Reward hyper-parameters (synchronized with stability.py constants)
# ---------------------------------------------------------------------------

ALPHA: float = 1e-5   # Massively scaled down Weight on Lyapunov energy drift ΔV(s)
BETA:  float = 1.0    # Weight on infrastructure cost
GAMMA: float = 1.0    # Weight on per-step SLA violation indicator

MAX_QUEUE_NORM = 200.0
MAX_LATENCY_NORM = 1000.0
MAX_REQUEST_RATE_NORM = 100.0

MAX_STEPS: int = 100      # Episode length
N_NODES:   int = 5        # Cluster size


class AntiAtroposEnvironment(Environment):
    """
    Autonomous SRE simulation environment.

    The agent observes a microservice cluster and issues management commands
    (SCALE_UP, SCALE_DOWN, REROUTE_TRAFFIC, SHED_LOAD, NO_OP) each step.
    The environment advances one discrete time-tick per step, computes the
    Lyapunov reward, and returns the updated ClusterObservation.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialise environment metadata and the simulation core."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: str = "task-1"
        self._mode: EnvironmentMode = EnvironmentMode.SIMULATED
        
        # Core components
        self._sim: ClusterSimulator = ClusterSimulator(n_nodes=N_NODES, task_id="task-1")
        self._telemetry = PrometheusClient()
        self._executor = KubernetesExecutor()
        self._validator = ActionValidator()
        self._observability = get_observability_tracker()
        self._logger = logging.getLogger("antiatropos.env")
        
        self._nodes_true: list[dict] = []
        self._nodes_obs: list[dict] = []
        self._prev_lyapunov: float = 0.0
        self._sla_violations: int = 0
        self._action_ack_status: str = "success"
        self._last_action_id: str = ""
        self._last_executor_latency_ms: float = 0.0
        self._last_executor_error_code: str = ""
        self._last_metric_time: float = 0.0

    def reset(self, task_id: str = "task-1", mode: str = "simulated") -> ClusterObservation:
        """
        Start a fresh episode with a specific task profile and mode.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = task_id
        try:
            self._mode = EnvironmentMode(mode)
        except ValueError:
            self._mode = EnvironmentMode.SIMULATED
            
        self._sla_violations = 0
        self._action_ack_status = "success"
        self._last_action_id = ""
        self._last_executor_latency_ms = 0.0
        self._last_executor_error_code = ""
        
        # Only set baseline metric time for hybrid/live to prevent misleading freshness in SIM
        if self._mode in [EnvironmentMode.HYBRID, EnvironmentMode.LIVE]:
            self._last_metric_time = time.time()
        else:
            self._last_metric_time = 0.0

        # Initialize core components based on mode
        if self._mode != EnvironmentMode.SIMULATED:
            # In Hybrid/Live mode, we might want to connect to real endpoints
            # self._telemetry = PrometheusClient(url=os.getenv("PROMETHEUS_URL"))
            pass

        self._sim.reset(task_id=task_id)
        
        # If in hybrid mode, immediately pull a baseline
        if self._mode in [EnvironmentMode.HYBRID, EnvironmentMode.LIVE]:
            node_ids = [n["node_id"] for n in self._sim.state(for_agent=False)]
            metrics = self._telemetry.fetch_latest_metrics(node_ids)
            self._sim.reconcile_state(metrics)

        self._nodes_true = self._sim.state(for_agent=False)
        self._nodes_obs  = self._sim.state(for_agent=True)
        self._prev_lyapunov = compute_lyapunov(self._nodes_true)

        return self._build_observation()

    def step(self, action: SREAction) -> ClusterObservation:  # type: ignore[override]
        """
        Advance the simulation by one discrete time tick.
        """
        self._state.step_count += 1
        self._last_action_id = str(uuid4())
        self._last_executor_latency_ms = 0.0
        self._last_executor_error_code = ""
        
        # 1. Action Validation & Execution
        valid_targets = [n["node_id"] for n in self._nodes_true]
        is_enabled, mode_error = self._is_action_enabled_for_mode(action.action_type)
        if not is_enabled:
            self._action_ack_status = f"Rejected: {mode_error}"
            self._last_executor_error_code = "MODE_UNSUPPORTED"
            is_valid = False
            error = mode_error
        else:
            is_valid, error = self._validator.validate(
            action.action_type, 
            action.target_node_id, 
            action.parameter,
            valid_targets=valid_targets
            )
        
        apply_to_simulator = False

        if not is_valid:
            self._action_ack_status = f"Rejected: {error}"
            if not self._last_executor_error_code:
                self._last_executor_error_code = "VALIDATION_FAILED"
            # Increment invalid action count on the simulator so it's consistent
            self._sim.invalid_action_count += 1
            # Still advance time but the action didn't happen
        else:
            if self._mode == EnvironmentMode.LIVE:
                # In LIVE mode, we actually hit the cluster
                exec_result = self._executor.execute_with_metadata(
                    action.action_type,
                    action.target_node_id,
                    action.parameter,
                )
                self._last_action_id = exec_result.get("action_id", self._last_action_id)
                self._action_ack_status = exec_result.get("ack_status", "Error: missing ack status")
                self._last_executor_latency_ms = float(exec_result.get("executor_latency_ms", 0.0))
                self._last_executor_error_code = str(exec_result.get("executor_error_code", ""))
                apply_to_simulator = self._action_ack_status.startswith("Ack:")
            else:
                self._action_ack_status = "success (simulated)"
                apply_to_simulator = True

        # Keep simulator aligned with control-plane ack semantics.
        if apply_to_simulator:
            self._sim.apply_action(action)

        # 2. Advance Physics
        self._sim.tick()
        
        # 3. Telemetry Ingestion (Hybrid / Live)
        if self._mode in [EnvironmentMode.HYBRID, EnvironmentMode.LIVE]:
            node_ids = [n["node_id"] for n in self._nodes_true]
            metrics = self._telemetry.fetch_latest_metrics(node_ids)
            self._sim.reconcile_state(metrics)
            self._last_metric_time = time.time()
        
        # 4. Extract states (Ground Truth for reward; Observation for agent)
        self._nodes_true = self._sim.state(for_agent=False)
        self._nodes_obs  = self._sim.state(for_agent=True)

        # 5. SLA Check
        avg_latency = self._avg_latency(self._nodes_true)
        error_rate  = self._error_rate(self._nodes_true)
        sla_violation_step = 1 if (avg_latency > 200.0 or error_rate > 0.05) else 0
        if sla_violation_step:
            self._sla_violations += 1

        # 6. Compute Lyapunov stability metrics from Ground Truth
        current_lyapunov = compute_lyapunov(self._nodes_true)
        
        # 7. Compute scalar reward
        cost = self._compute_cost(self._nodes_true)
        reward = compute_reward(
            v_prev=self._prev_lyapunov,
            v_curr=current_lyapunov,
            cost=cost,
            sla_violation_step=sla_violation_step,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA
        )
        
        self._prev_lyapunov = current_lyapunov

        # 8. Termination check
        done = (
            self._state.step_count >= MAX_STEPS
            or all(n["status"] == NodeStatus.FAILED for n in self._nodes_true)
        )

        # 9. Package Observation
        obs = self._build_observation()
        obs.done   = done
        obs.reward = reward

        self._observability.record_step(
            task_id=self._task_id,
            mode=str(self._mode.value),
            action_type=str(action.action_type.value),
            target_node_id=str(action.target_node_id),
            ack_status=self._action_ack_status,
            reward=reward,
            lyapunov_energy=obs.lyapunov_energy,
            total_queue_backlog=obs.total_queue_backlog,
            average_latency_ms=obs.average_latency_ms,
            executor_latency_ms=self._last_executor_latency_ms,
            executor_error_code=self._last_executor_error_code,
        )

        self._logger.info(
            json.dumps(
                {
                    "event": "antiatropos_step",
                    "episode_id": self._state.episode_id,
                    "task_id": self._task_id,
                    "mode": self._mode.value,
                    "step": self._state.step_count,
                    "action_type": action.action_type.value,
                    "target_node_id": action.target_node_id,
                    "parameter": float(action.parameter),
                    "action_id": self._last_action_id,
                    "action_ack_status": self._action_ack_status,
                    "executor_latency_ms": self._last_executor_latency_ms,
                    "executor_error_code": self._last_executor_error_code,
                    "reward": reward,
                    "lyapunov_energy": obs.lyapunov_energy,
                    "average_latency_ms_norm": obs.average_latency_ms,
                    "total_queue_backlog_norm": obs.total_queue_backlog,
                    "error_rate": obs.error_rate,
                    "done": done,
                }
            )
        )

        return obs

    @property
    def state(self) -> State:
        return self._state

    # -----------------------------------------------------------------------
    # Logic Helpers
    # -----------------------------------------------------------------------
    def _is_action_enabled_for_mode(self, action_type: str) -> tuple[bool, str]:
        if hasattr(action_type, "value"):
            action = str(action_type.value)
        else:
            action = str(action_type)
        if self._mode in [EnvironmentMode.SIMULATED, EnvironmentMode.HYBRID]:
            return True, "Enabled"

        if self._mode == EnvironmentMode.LIVE:
            capability_error = self._executor.live_capability_error(action)
            if capability_error:
                return False, capability_error
            return True, "Enabled"

        return False, f"Unsupported environment mode: {self._mode}"

    def _compute_cost(self, nodes_true: list[dict]) -> float:
        """Calculates current running infra cost using provisioned capacity units."""
        total_capacity_units = 0
        for node in nodes_true:
            if node["status"] == NodeStatus.FAILED:
                continue
            total_capacity_units += int(node.get("capacity_units", 0))
            total_capacity_units += int(node.get("pending_capacity_units", 0))
        return total_capacity_units * COST_PER_CAPACITY_UNIT_PER_HOUR

    def _avg_latency(self, nodes: list[dict]) -> float:
        """Computes importance-weighted mean latency across the cluster."""
        if not nodes:
            return float("inf")

        weighted_latency = 0.0
        total_weight = 0.0
        for n in nodes:
            weight = float(n.get("importance_weight", 1.0))
            latency = MAX_LATENCY_NORM if n["status"] == NodeStatus.FAILED else float(n["latency_ms"])
            weighted_latency += weight * latency
            total_weight += weight

        if total_weight <= 0:
            return float("inf")
        return weighted_latency / total_weight

    def _error_rate(self, nodes: list[dict]) -> float:
        """Calculates an importance-weighted fraction of dropped or lost requests."""
        total_incoming = sum(float(n.get("incoming_request_rate", 0.0)) * float(n.get("importance_weight", 1.0)) for n in nodes)
        if total_incoming <= 0:
            return 0.0
        total_drops = sum(float(n.get("dropped_requests", 0.0)) * float(n.get("importance_weight", 1.0)) for n in nodes)
        return min(1.0, total_drops / total_incoming)

    def _vip_failure_count(self, nodes: list[dict]) -> int:
        """Counts failed VIP nodes for reporting and diagnostics."""
        return sum(1 for n in nodes if n.get("is_vip") and n["status"] == NodeStatus.FAILED)

    def _build_observation(self) -> ClusterObservation:
        """Assembles the ClusterObservation from the current observed simulator state."""
        node_obs = [
            NodeObservation(
                node_id=n["node_id"],
                status=n["status"],
                queue_depth=min(1.0, max(0.0, float(n["queue_depth"]) / MAX_QUEUE_NORM)),
                latency_ms=min(1.0, max(0.0, float(n["latency_ms"]) / MAX_LATENCY_NORM)),
                incoming_request_rate=min(1.0, max(0.0, float(n["incoming_request_rate"]) / MAX_REQUEST_RATE_NORM)),
                cpu_utilization=min(1.0, max(0.0, float(n["cpu_utilization"]))),
                is_vip=bool(n.get("is_vip", False)),
                importance_weight=float(n.get("importance_weight", 1.0)),
                done=False,
                reward=0.0,
            )
            for n in self._nodes_obs
        ]

        freshness = int((time.time() - self._last_metric_time) * 1000) if self._last_metric_time > 0 else 0

        return ClusterObservation(
            cluster_id=self._state.episode_id,
            task_id=self._task_id,
            mode=self._mode,
            active_nodes=sum(1 for n in self._nodes_true if n["status"] != NodeStatus.FAILED),
            average_latency_ms=min(1.0, max(0.0, self._avg_latency(self._nodes_true) / MAX_LATENCY_NORM)),
            error_rate=self._error_rate(self._nodes_true),
            total_queue_backlog=min(1.0, max(0.0, sum(float(n["queue_depth"]) for n in self._nodes_obs) / (N_NODES * MAX_QUEUE_NORM))),
            current_cost_per_hour=self._compute_cost(self._nodes_true),
            lyapunov_energy=self._prev_lyapunov,
            nodes=node_obs,
            step=self._state.step_count,
            max_steps=MAX_STEPS,
            sla_violations=self._sla_violations,
            invalid_action_count=self._sim.invalid_action_count,
            vip_failure_count=self._vip_failure_count(self._nodes_true),
            metric_timestamp=self._last_metric_time,
            data_freshness_ms=freshness,
            action_ack_status=self._action_ack_status,
            action_id=self._last_action_id,
            executor_latency_ms=self._last_executor_latency_ms,
            executor_error_code=self._last_executor_error_code,
            choke_level=0.0,
            done=False,
            reward=0.0,
        )


