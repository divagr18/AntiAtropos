import time
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SREAction, ClusterObservation, NodeObservation, NodeStatus, EnvironmentMode
    from ..simulator import ClusterSimulator, COST_PER_CAPACITY_UNIT_PER_HOUR
    from ..stability import compute_lyapunov, compute_reward
    from ..telemetry import PrometheusClient
    from ..control import KubernetesExecutor, ActionValidator
except ImportError:
    from models import SREAction, ClusterObservation, NodeObservation, NodeStatus, EnvironmentMode  # type: ignore[no-redef]
    from simulator import ClusterSimulator, COST_PER_CAPACITY_UNIT_PER_HOUR  # type: ignore[no-redef]
    from stability import compute_lyapunov, compute_reward  # type: ignore[no-redef]
    from telemetry import PrometheusClient  # type: ignore[no-redef]
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
        
        self._nodes_true: list[dict] = []
        self._nodes_obs: list[dict] = []
        self._prev_lyapunov: float = 0.0
        self._sla_violations: int = 0
        self._action_ack_status: str = "success"
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
        
        # 1. Action Validation & Execution
        valid_targets = [n["node_id"] for n in self._nodes_true]
        is_valid, error = self._validator.validate(
            action.action_type, 
            action.target_node_id, 
            action.parameter,
            valid_targets=valid_targets
        )
        
        if not is_valid:
            self._action_ack_status = f"Rejected: {error}"
            # Increment invalid action count on the simulator so it's consistent
            self._sim.invalid_action_count += 1
            # Still advance time but the action didn't happen
        else:
            if self._mode == EnvironmentMode.LIVE:
                # In LIVE mode, we actually hit the cluster
                self._action_ack_status = self._executor.execute(action.action_type, action.target_node_id, action.parameter)
            else:
                self._action_ack_status = "success (simulated)"
            
            # Always update the physics engine to keep tracking expectations
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
        return obs

    @property
    def state(self) -> State:
        return self._state

    # -----------------------------------------------------------------------
    # Logic Helpers
    # -----------------------------------------------------------------------

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
            choke_level=0.0,
            done=False,
            reward=0.0,
        )


