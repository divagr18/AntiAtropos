import time
import json
import os
import logging
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SREAction, ClusterObservation, NodeObservation, NodeStatus, EnvironmentMode
    from ..simulator import ClusterSimulator, COST_PER_CAPACITY_UNIT_PER_HOUR
    from ..stability import (
        compute_lyapunov,
        compute_reward,
        compute_barrier,
        normalize_reward,
        smooth_sla_penalty,
        compute_drift,
        BARRIER_NORM_SCALE,
        REWARD_SCALE_VERSION,
    )
    from ..telemetry import PrometheusClient, get_observability_tracker
    from ..control import KubernetesExecutor, ActionValidator
except ImportError:
    from models import SREAction, ClusterObservation, NodeObservation, NodeStatus, EnvironmentMode  # type: ignore[no-redef]
    from simulator import ClusterSimulator, COST_PER_CAPACITY_UNIT_PER_HOUR  # type: ignore[no-redef]
    from stability import (  # type: ignore[no-redef]
        compute_lyapunov,
        compute_reward,
        compute_barrier,
        normalize_reward,
        smooth_sla_penalty,
        compute_drift,
        BARRIER_NORM_SCALE,
        REWARD_SCALE_VERSION,
    )
    from telemetry import PrometheusClient, get_observability_tracker  # type: ignore[no-redef]
    from control import KubernetesExecutor, ActionValidator  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Reward hyper-parameters (synchronized with stability.py constants)
# ---------------------------------------------------------------------------

ALPHA: float = 0.002   # Weight on Lyapunov energy drift DeltaV(s) (Increased for faster feedback)
BETA:  float = 0.01    # Weight on infrastructure cost (Reduced to prevent cheap-but-dead strategies)
GAMMA: float = 10.0    # Weight on per-step SLA violation indicator (Increased to force reactive scaling)
DELTA: float = 0.005   # Weight on control-barrier function penalty (queue safety zone)

MAX_QUEUE_NORM = 200.0
MAX_LATENCY_NORM = 1000.0
MAX_REQUEST_RATE_NORM = 100.0

MAX_STEPS: int = 100      # Episode length
N_NODES:   int = 5        # Cluster size
REWARD_OUTPUT_MODES = {"normalized", "raw"}


class AntiAtroposEnvironment(Environment):
    """
    Autonomous SRE simulation environment.

    The agent observes a microservice cluster and issues management commands
    (SCALE_UP, SCALE_DOWN, REROUTE_TRAFFIC, SHED_LOAD, NO_OP) each step.
    The environment advances one discrete time-tick per step, computes the
    Lyapunov reward, and returns the updated ClusterObservation.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    @staticmethod
    def _parse_mode(raw_mode: str | None) -> EnvironmentMode:
        candidate = (raw_mode or os.getenv("ANTIATROPOS_ENV_MODE", "simulated")).strip().lower()
        alias = {
            "prod": "aws",
            "production": "aws",
        }
        normalized = alias.get(candidate, candidate)
        try:
            return EnvironmentMode(normalized)
        except ValueError:
            return EnvironmentMode.SIMULATED

    def _uses_real_telemetry(self) -> bool:
        return self._mode in [EnvironmentMode.HYBRID, EnvironmentMode.LIVE, EnvironmentMode.AWS]

    def _uses_real_executor(self) -> bool:
        return self._mode in [EnvironmentMode.LIVE, EnvironmentMode.AWS]

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
        self._prev_nodes_true: list[dict] = []  # For per-node queue delta + reward
        self._prev_lyapunov: float = 0.0
        self._sla_violations: int = 0
        self._action_ack_status: str = "success"
        self._last_action_id: str = ""
        self._last_executor_latency_ms: float = 0.0
        self._last_executor_error_code: str = ""
        self._last_raw_reward: float = 0.0
        self._last_normalized_reward: float = 0.0
        self._last_reward_drift: float = 0.0
        self._last_reward_cost: float = 0.0
        self._last_reward_sla: float = 0.0
        self._last_reward_barrier: float = 0.0
        self._reward_output_mode: str = os.getenv("ANTIATROPOS_REWARD_OUTPUT_MODE", "normalized").strip().lower()
        if self._reward_output_mode not in REWARD_OUTPUT_MODES:
            self._reward_output_mode = "normalized"
        self._last_metric_time: float = 0.0

    def reset(self, task_id: str = "task-1", mode: str = "simulated", seed: int | None = None) -> ClusterObservation:
        """
        Start a fresh episode with a specific task profile and mode.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = task_id
        self._mode = self._parse_mode(mode)
            
        self._sla_violations = 0
        self._action_ack_status = "success"
        self._last_action_id = ""
        self._last_executor_latency_ms = 0.0
        self._last_executor_error_code = ""
        self._last_raw_reward = 0.0
        self._last_normalized_reward = 0.0
        
        # Only set baseline metric time for hybrid/live to prevent misleading freshness in SIM
        if self._uses_real_telemetry():
            self._last_metric_time = time.time()
        else:
            self._last_metric_time = 0.0

        # Initialize core components based on mode
        if self._mode != EnvironmentMode.SIMULATED:
            # In Hybrid/Live mode, we might want to connect to real endpoints
            # self._telemetry = PrometheusClient(url=os.getenv("PROMETHEUS_URL"))
            pass

        self._sim.reset(task_id=task_id, seed=seed)
        
        # If in hybrid mode, immediately pull a baseline
        if self._uses_real_telemetry():
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
        self._last_raw_reward = 0.0
        self._last_normalized_reward = 0.0
        
        # 1. Action Validation & Execution
        valid_targets = [n["node_id"] for n in self._nodes_true]
        is_enabled, mode_error = self._is_action_enabled_for_mode(action.action_type)
        if not is_enabled:
            self._action_ack_status = f"Rejected: {mode_error}"
            self._last_executor_error_code = ""
            is_valid = False
            error = mode_error
            cooldown_penalty = 0.0
        else:
            self._validator.set_tick(self._state.step_count)
            is_valid, error, cooldown_penalty = self._validator.validate(
            action.action_type, 
            action.target_node_id, 
            action.parameter,
            valid_targets=valid_targets
            )
        
        apply_to_simulator = False

        if not is_valid:
            self._action_ack_status = f"Rejected: {error}"
            # Keep capability-gate rejections out of executor error metrics.
            if not self._last_executor_error_code and not str(error).startswith("Live mode rejected"):
                self._last_executor_error_code = "VALIDATION_FAILED"
            # Increment invalid action count on the simulator so it's consistent
            self._sim.invalid_action_count += 1
            # Still advance time but the action didn't happen
        else:
            if self._uses_real_executor():
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
        if self._uses_real_telemetry():
            node_ids = [n["node_id"] for n in self._nodes_true]
            metrics = self._telemetry.fetch_latest_metrics(node_ids)
            self._sim.reconcile_state(metrics)
            self._last_metric_time = time.time()
        
        # 4. Extract states (Ground Truth for reward; Observation for agent)
        self._prev_nodes_true = self._nodes_true  # Save for per-node delta
        self._nodes_true = self._sim.state(for_agent=False)
        self._nodes_obs  = self._sim.state(for_agent=True)

        # 5. SLA Check (smooth sigmoid penalty instead of binary cliff)
        avg_latency_norm = self._avg_latency(self._nodes_true) / MAX_LATENCY_NORM
        error_rate  = self._error_rate(self._nodes_true)
        sla_penalty_step = smooth_sla_penalty(avg_latency_norm, error_rate)
        # Track binary violations for the grader (backward compat)
        if avg_latency_norm > 0.20 or error_rate > 0.05:
            self._sla_violations += 1

        # 6. Compute Lyapunov stability metrics from Ground Truth
        current_lyapunov = compute_lyapunov(self._nodes_true)
        
        # 7. Compute scalar reward (with barrier function)
        cost = self._compute_cost(self._nodes_true)
        barrier = compute_barrier(self._nodes_true)
        raw_reward = compute_reward(
            v_prev=self._prev_lyapunov,
            v_curr=current_lyapunov,
            cost=cost,
            sla_violation_step=sla_penalty_step,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA,
            barrier=barrier,
            delta=DELTA,
        )
        normalized_reward = normalize_reward(raw_reward)
        # Apply soft cooldown penalty: reduces reward for rapid re-scaling
        # without blocking the action (emergency scaling still goes through)
        if cooldown_penalty > 0:
            normalized_reward = max(0.0, normalized_reward - cooldown_penalty * 0.1)
        reward = normalized_reward if self._reward_output_mode == "normalized" else raw_reward
        self._last_raw_reward = raw_reward
        self._last_normalized_reward = normalized_reward
        # Store reward component breakdown for the observation
        delta_v = compute_drift(self._prev_lyapunov, current_lyapunov)
        barrier_norm = barrier / BARRIER_NORM_SCALE if BARRIER_NORM_SCALE > 0 else barrier
        self._last_reward_drift = -(ALPHA * delta_v)
        self._last_reward_cost = -(BETA * cost)
        self._last_reward_sla = -(GAMMA * sla_penalty_step)
        self._last_reward_barrier = -(DELTA * barrier_norm)
        
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
            reward_output=reward,
            reward_raw=raw_reward,
            reward_normalized=normalized_reward,
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
                    "reward_output": reward,
                    "reward_raw": raw_reward,
                    "reward_normalized": normalized_reward,
                    "reward_output_mode": self._reward_output_mode,
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

        if self._mode in [EnvironmentMode.LIVE, EnvironmentMode.AWS]:
            capability_error = self._executor.live_capability_error(action)
            if capability_error:
                if self._mode == EnvironmentMode.AWS:
                    return False, f"AWS mode rejected {action}: {capability_error}"
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
        # Build a lookup for previous node state (for queue_delta and node_reward)
        prev_by_id: dict[str, dict] = {n["node_id"]: n for n in self._prev_nodes_true}

        node_obs = []
        for n in self._nodes_obs:
            # Per-node queue delta (normalized)
            true_n = next((t for t in self._nodes_true if t["node_id"] == n["node_id"]), n)
            prev_n = prev_by_id.get(n["node_id"])
            if prev_n:
                queue_delta_raw = float(n["queue_depth"]) - float(prev_n.get("queue_depth", 0))
                queue_delta = max(-1.0, min(1.0, queue_delta_raw / MAX_QUEUE_NORM))
            else:
                queue_delta = 0.0

            # Per-node reward contribution (normalized)
            # Uses same formula as global reward but per-node
            weight = float(n.get("importance_weight", 1.0))
            if prev_n:
                prev_q = float(prev_n.get("queue_depth", 0))
                curr_q = float(true_n["queue_depth"])
                node_drift = weight * (curr_q ** 2 - prev_q ** 2)
                node_barrier = max(0, curr_q - 150.0) ** 2  # Q_BARRIER_MAX=150
                node_cost = float(true_n.get("capacity_units", 0)) * COST_PER_CAPACITY_UNIT_PER_HOUR
                node_reward_raw = -(ALPHA * node_drift + DELTA * (node_barrier / 10000.0) + BETA * node_cost)
                # Normalize to [-1, 0] range
                node_reward_val = max(-1.0, min(0.0, node_reward_raw / 10.0))
            else:
                node_reward_val = 0.0

            # SLA proximity: how close this node is to violating (normalized)
            node_latency_norm = min(1.0, max(0.0, float(n["latency_ms"]) / MAX_LATENCY_NORM))
            sla_prox = max(0.0, min(1.0, node_latency_norm / 0.20))  # 0.20 is SLA threshold

            node_obs.append(NodeObservation(
                node_id=n["node_id"],
                status=n["status"],
                queue_depth=min(1.0, max(0.0, float(n["queue_depth"]) / MAX_QUEUE_NORM)),
                latency_ms=min(1.0, max(0.0, float(n["latency_ms"]) / MAX_LATENCY_NORM)),
                incoming_request_rate=min(1.0, max(0.0, float(n["incoming_request_rate"]) / MAX_REQUEST_RATE_NORM)),
                cpu_utilization=min(1.0, max(0.0, float(n["cpu_utilization"]))),
                is_vip=bool(n.get("is_vip", False)),
                importance_weight=float(n.get("importance_weight", 1.0)),
                capacity=float(n.get("capacity_units", 0)) / 5.0,  # Normalize to [0,1]
                pending_capacity=float(n.get("pending_capacity_units", 0)) / 5.0,
                queue_delta=queue_delta,
                sla_proximity=sla_prox,
                node_reward=node_reward_val,
                done=False,
                reward=0.0,
            ))

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
            raw_reward=self._last_raw_reward,
            normalized_reward=self._last_normalized_reward,
            reward_scale_version=REWARD_SCALE_VERSION,
            reward_drift=self._last_reward_drift,
            reward_cost=self._last_reward_cost,
            reward_sla=self._last_reward_sla,
            reward_barrier=self._last_reward_barrier,
            choke_level=0.0,
            done=False,
            reward=0.0,
        )

