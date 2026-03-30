"""
AntiAtropos Cluster Simulator — Phase 2.

Implements a discrete-time fluid-queue model for a microservice cluster.
Each node maintains its own queue governed by:

    Q_i(t+1) = max(Q_i(t) + λ_i(t) - μ_i(t), 0)

where:
    λ_i(t) = arriving requests this tick  (traffic profile–dependent)
    μ_i(t) = processed requests this tick = capacity_i × BASE_SERVICE_RATE

Latency is derived from queue depth (proxy for Little's Law):
    latency_ms = BASE_LATENCY_MS + queue_depth × MS_PER_QUEUED

CPU utilisation is the traffic intensity  ρ_i = λ_i / μ_i  (clamped to 1.0).

The three traffic profiles map to the three hackathon tasks:
    task-1  Predictive Scaling   — linear traffic ramp.
    task-2  Fault Tolerance      — stable traffic then random node failure.
    task-3  Stability Under Surge — stochastic DDoS bursts on non-protected nodes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

try:
    from .models import NodeStatus
except ImportError:
    from models import NodeStatus  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

BASE_SERVICE_RATE: int = 15
"""Requests a single capacity unit can process per tick."""

INITIAL_CAPACITY: int = 3
"""Starting capacity units per node  →  μ_0 = 3 × 15 = 45 req/tick."""

BASE_LATENCY_MS: float = 20.0
"""Intrinsic processing latency at zero queue depth (ms)."""

MS_PER_QUEUED: float = 2.0
"""Extra latency added per request sitting in the queue (ms)."""

OVERLOAD_THRESHOLD: int = 80
"""Queue depth at which a node transitions to DEGRADED status.
At this threshold: latency = 20 + 80×2 = 180 ms  (near the 200 ms SLA)."""

MAX_CAPACITY: int = 10
"""Hard ceiling on SCALE_UP to prevent infinite free scaling."""

MIN_CAPACITY: int = 1
"""Hard floor on SCALE_DOWN so costs cannot reach zero by degrading all nodes."""

COST_PER_CAPACITY_UNIT_PER_HOUR: float = 0.05
"""USD/hr per capacity unit.  Cluster cost = Σ capacity_i × this value."""

BOOT_DELAY_TICKS: int = 5
SENSOR_DROPOUT_PROB: float = 0.05
CONTROL_PLANE_DROPOUT_PROB: float = 0.05
MAX_SCALING_STEP: int = 3


# ---------------------------------------------------------------------------
# Traffic profile constants
# ---------------------------------------------------------------------------

#  Task 1 — Predictive Scaling
T1_INITIAL_LAMBDA: float = 35.0
"""Baseline arrival rate per node at t=0 (req/tick).  ρ₀ ≈ 0.78."""

T1_RAMP_SLOPE: float = 1.0
"""Additional req/tick added to every node each tick.
λ exceeds μ=45 at t≈10; queues reach OVERLOAD_THRESHOLD (~tick 25)."""

#  Task 2 — Fault Tolerance
T2_INITIAL_LAMBDA: float = 40.0
"""Stable arrival rate per node.  ρ = 40/45 = 0.89 (stable but loaded)."""

T2_FAIL_TICK: int = 25
"""Tick at which a randomly chosen node is marked FAILED and its traffic
is redistributed to surviving nodes, pushing them above μ."""

#  Task 3 — Stability Under Surge
T3_INITIAL_LAMBDA: float = 35.0
"""Baseline arrival rate.  Payment Gateway node is always at this level."""

T3_SURGE_AMPLITUDE: float = 70.0
"""Extra requests per tick during a DDoS spike on a non-protected node."""

T3_SURGE_PROBABILITY: float = 0.20
"""Per-node probability of a spike occurring each tick."""

T3_PROTECTED_NODE: str = "node-0"
"""The Payment Gateway node. Never targeted by surge and never given SHED_LOAD."""

# Default cluster size (overridable via constructor)
DEFAULT_N_NODES: int = 5


# ---------------------------------------------------------------------------
# Per-node internal state
# ---------------------------------------------------------------------------

@dataclass
class NodeState:
    """
    Mutable state for a single cluster node.

    All quantities are in per-tick units unless noted.
    """

    node_id: str
    capacity: int = INITIAL_CAPACITY

    # Observed state
    status: NodeStatus = NodeStatus.HEALTHY
    queue_depth: float = 0.0         # continuous internally; int when exported
    incoming_request_rate: float = 0.0
    latency_ms: float = BASE_LATENCY_MS
    cpu_utilization: float = 0.0

    # Per-tick accounting (reset each tick)
    dropped_requests: float = 0.0
    pending_capacity_queue: list[int] = field(default_factory=list)

    # Derived (recomputed whenever capacity or status changes)
    _service_rate: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._service_rate = float(self.capacity * BASE_SERVICE_RATE)

    # ----

    @property
    def service_rate(self) -> float:
        return self._service_rate

    def recompute_service_rate(self) -> None:
        """Call after any change to capacity or status."""
        if self.status == NodeStatus.FAILED:
            self._service_rate = 0.0
        else:
            self._service_rate = float(self.capacity * BASE_SERVICE_RATE)

    def to_dict(self) -> dict:
        """Export to the plain dict format expected by environment.py."""
        return {
            "node_id": self.node_id,
            "status": self.status,
            "queue_depth": int(self.queue_depth),
            "latency_ms": round(self.latency_ms, 2),
            "incoming_request_rate": round(self.incoming_request_rate, 2),
            "cpu_utilization": round(min(1.0, self.cpu_utilization), 4),
            "dropped_requests": int(self.dropped_requests),
        }


# ---------------------------------------------------------------------------
# ClusterSimulator
# ---------------------------------------------------------------------------

class ClusterSimulator:
    """
    Discrete-time fluid-queue simulator for the AntiAtropos cluster.

    Public API (called by AntiAtroposEnvironment):
        reset(task_id)          — restart episode with given task profile.
        apply_action(action)    — mutate node state based on SREAction.
        tick()                  — advance time by one step.
        state() → list[dict]    — snapshot of all per-node metrics.

    The fluid-queue update is:
        Q_i(t+1) = max(Q_i(t) + λ_i(t) - μ_i(t), 0)

    This is deliberately deterministic by default (seed=42) so
    episodes are reproducible. The task-3 stochastic bursts use the
    seeded RNG so grader scores remain comparable across runs.
    """

    def __init__(
        self,
        n_nodes: int = DEFAULT_N_NODES,
        task_id: str = "task-1",
        seed: Optional[int] = 42,
    ) -> None:
        self._n_nodes = n_nodes
        self._task_id = task_id
        self._rng = random.Random(seed)
        self._tick_count: int = 0
        self._failed_node_id: Optional[str] = None
        self._nodes: list[NodeState] = []
        self._reset_nodes()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def reset(self, task_id: str = "task-1") -> None:
        """Restart the simulator for a fresh episode."""
        self._task_id = task_id
        self._tick_count = 0
        self._failed_node_id = None
        self._reset_nodes()

    def state(self) -> list[dict]:
        """Return current per-node state as a list of plain dicts."""
        state_list = []
        for n in self._nodes:
            d = n.to_dict()
            if self._rng.random() < SENSOR_DROPOUT_PROB:
                d["queue_depth"] = 0
                d["latency_ms"] = -1.0
            state_list.append(d)
        return state_list

    def apply_action(self, action) -> None:
        """
        Apply an SREAction to the cluster *before* this tick's physics run.

        Actions and their effects:
            SCALE_UP     — increase capacity_i by int(parameter); if DEGRADED → HEALTHY.
            SCALE_DOWN   — decrease capacity_i (floor: MIN_CAPACITY).
            REROUTE_TRAFFIC — move parameter-fraction of λ from target to healthy peers.
            SHED_LOAD    — immediately drain parameter-fraction of the target's queue.
            NO_OP        — no-op.
        """
        if action.action_type.value == "NO_OP":
            return

        if self._rng.random() < CONTROL_PLANE_DROPOUT_PROB:
            # Simulate Control-Plane Dropout API failure
            return

        target = self._find_node(action.target_node_id)
        if target is None:
            return  # unknown node; silently ignore

        at = action.action_type.value
        param = min(1.0, max(0.0, action.parameter)) # Bound tightly to [0,1]

        if at == "SCALE_UP":
            delta = max(1, int(param * MAX_SCALING_STEP))
            current_max = target.capacity + len(target.pending_capacity_queue)
            actual_delta = min(delta, MAX_CAPACITY - current_max)
            for _ in range(actual_delta):
                target.pending_capacity_queue.append(BOOT_DELAY_TICKS)
            if target.status == NodeStatus.DEGRADED:
                target.status = NodeStatus.HEALTHY

        elif at == "SCALE_DOWN":
            delta = max(1, int(param * MAX_SCALING_STEP))
            target.capacity = max(MIN_CAPACITY, target.capacity - delta)
            target.recompute_service_rate()

        elif at == "REROUTE_TRAFFIC":
            frac = min(1.0, param)
            rerouted_load = target.incoming_request_rate * frac
            target.incoming_request_rate -= rerouted_load

            # Distribute evenly among healthy / degraded peers
            peers = [
                n for n in self._nodes
                if n.node_id != target.node_id and n.status != NodeStatus.FAILED
            ]
            if peers:
                share = rerouted_load / len(peers)
                for peer in peers:
                    peer.incoming_request_rate += share

        elif at == "SHED_LOAD":
            frac = min(1.0, param)
            shed = target.queue_depth * frac
            target.queue_depth = max(0.0, target.queue_depth - shed)
            target.dropped_requests += shed

    def tick(self) -> None:
        """
        Advance the simulation by one discrete time tick.

        Execution order:
            1. Inject traffic according to the active traffic profile.
            2. Update queues:  Q(t+1) = max(Q(t) + λ - μ, 0).
            3. Derive latency and CPU utilisation from the new queue state.
            4. Classify node status (HEALTHY / DEGRADED / FAILED).
        """
        self._tick_count += 1
        self._update_capacity()
        self._inject_traffic()
        self._update_queues()
        self._update_derived_metrics()
        self._update_statuses()

    def _update_capacity(self) -> None:
        """Process pending capacity from SCALE_UP actions"""
        for node in self._nodes:
            if node.pending_capacity_queue:
                # Decrement all timers by 1 tick
                node.pending_capacity_queue = [t - 1 for t in node.pending_capacity_queue]
                
                # Turn ready units into actual capacity
                ready_count = sum(1 for t in node.pending_capacity_queue if t <= 0)
                if ready_count > 0:
                    node.capacity = min(MAX_CAPACITY, node.capacity + ready_count)
                    node.recompute_service_rate()
                
                # Keep only units still booting
                node.pending_capacity_queue = [t for t in node.pending_capacity_queue if t > 0]

    # -----------------------------------------------------------------------
    # Private — initialisation
    # -----------------------------------------------------------------------

    def _randomize_domain(self) -> None:
        """Randomize physical constraints to prevent overfitting (Domain Randomization)"""
        self._t1_ramp_slope = self._rng.uniform(0.5, 2.0)
        self._t1_init_lambda = self._rng.uniform(25.0, 40.0)
        
        self._t2_fail_tick = self._rng.randint(10, 80)
        self._t2_init_lambda = self._rng.uniform(35.0, 45.0)

    def _reset_nodes(self) -> None:
        self._nodes = [
            NodeState(
                node_id=f"node-{i}",
                capacity=INITIAL_CAPACITY,
                incoming_request_rate=self._initial_lambda(),
            )
            for i in range(self._n_nodes)
        ]

    def _initial_lambda(self) -> float:
        """Return the task-appropriate starting arrival rate."""
        if self._task_id == "task-2":
            return self._t2_init_lambda
        if self._task_id == "task-3":
            return T3_INITIAL_LAMBDA
        return self._t1_init_lambda   # task-1 default

    def _find_node(self, node_id: str) -> Optional[NodeState]:
        for n in self._nodes:
            if n.node_id == node_id:
                return n
        return None

    # -----------------------------------------------------------------------
    # Private — traffic profiles
    # -----------------------------------------------------------------------

    def _inject_traffic(self) -> None:
        t = self._tick_count
        if self._task_id == "task-1":
            self._profile_task1(t)
        elif self._task_id == "task-2":
            self._profile_task2(t)
        elif self._task_id == "task-3":
            self._profile_task3(t)

    def _profile_task1(self, t: int) -> None:
        """
        Linear traffic ramp — uniform across all nodes.
        Uses randomized ramp slope for domain randomization.
        """
        lambda_t = self._t1_init_lambda + self._t1_ramp_slope * t
        for node in self._nodes:
            if node.status != NodeStatus.FAILED:
                node.incoming_request_rate = lambda_t
            else:
                node.latency_ms = 9999.0
                node.cpu_utilization = 0.0

    def _profile_task2(self, t: int) -> None:
        """
        Stable traffic until random T2_FAIL_TICK, then a random node fails.
        """
        # Set baseline for all live nodes
        for node in self._nodes:
            if node.status != NodeStatus.FAILED:
                node.incoming_request_rate = self._t2_init_lambda

        # Inject failure exactly once at the designated tick
        if t == self._t2_fail_tick and self._failed_node_id is None:
            victim = self._rng.choice(self._nodes)
            victim.status = NodeStatus.FAILED
            victim.incoming_request_rate = 0.0
            victim.recompute_service_rate()
            self._failed_node_id = victim.node_id

        # Redistribute the failed node's load to the survivors
        if self._failed_node_id is not None:
            survivors = [n for n in self._nodes if n.status != NodeStatus.FAILED]
            n_survivors = len(survivors)
            if n_survivors > 0:
                total_lambda = self._n_nodes * self._t2_init_lambda
                per_survivor = total_lambda / n_survivors
                for node in survivors:
                    node.incoming_request_rate = per_survivor

    def _profile_task3(self, t: int) -> None:
        """
        Stochastic DDoS-style bursts on non-protected nodes.

        Each non-protected, non-failed node independently receives a surge
        with probability T3_SURGE_PROBABILITY each tick.

            λ_i(t) = T3_INITIAL_LAMBDA + T3_SURGE_AMPLITUDE × Bernoulli(p)

        Expected λ on a non-protected node ≈ 35 + 70×0.2 = 49  >  μ=45.
        The cluster is unstable on average: the agent must SHED_LOAD
        strategically to protect node-0 (the Payment Gateway).
        """
        for node in self._nodes:
            if node.status == NodeStatus.FAILED:
                node.incoming_request_rate = 0.0
                continue

            if node.node_id == T3_PROTECTED_NODE:
                # Payment Gateway always gets baseline load only
                node.incoming_request_rate = T3_INITIAL_LAMBDA
            else:
                spike = (
                    T3_SURGE_AMPLITUDE
                    if self._rng.random() < T3_SURGE_PROBABILITY
                    else 0.0
                )
                node.incoming_request_rate = T3_INITIAL_LAMBDA + spike

    # -----------------------------------------------------------------------
    # Private — queue physics
    # -----------------------------------------------------------------------

    def _update_queues(self) -> None:
        """
        Fluid-queue update for all nodes.

            Q_i(t+1) = max(Q_i(t) + λ_i(t) − μ_i(t), 0)

        Failed nodes are zeroed out (traffic is simply lost; error rate
        accounts for them via fraction-of-failed-nodes in environment.py).
        """
        for node in self._nodes:
            node.dropped_requests = 0.0  # reset per-tick counter

            if node.status == NodeStatus.FAILED:
                node.queue_depth = 0.0
                continue

            excess = node.incoming_request_rate - node.service_rate
            node.queue_depth = max(0.0, node.queue_depth + excess)

    def _update_derived_metrics(self) -> None:
        """
        Compute latency_ms and cpu_utilization from the post-tick queue state.

        Latency model:
            — Stable region (λ < μ):  queue drains; latency = BASE_LATENCY_MS.
            — Overloaded region (λ ≥ μ): latency = BASE_LATENCY_MS + Q × MS_PER_QUEUED.

        This is a linear proxy for Little's Law (L = λW → W = Q/λ),
        scaled so that the 200 ms SLA boundary occurs at Q ≈ 90 req.
        """
        for node in self._nodes:
            if node.status == NodeStatus.FAILED:
                node.latency_ms = 9999.0  # Use a large finite number instead of float("inf") for JSON
                node.cpu_utilization = 0.0
                continue

            mu = max(node.service_rate, 1.0)   # guard against div/0
            lam = node.incoming_request_rate

            # CPU utilisation (traffic intensity ρ)
            node.cpu_utilization = min(1.0, lam / mu)

            # Latency
            if lam < mu:
                # Stable: queue drains each tick → latency is essentially intrinsic
                node.latency_ms = BASE_LATENCY_MS
            else:
                # Overloaded: latency grows with accumulated backlog
                node.latency_ms = min(
                    5_000.0,
                    BASE_LATENCY_MS + node.queue_depth * MS_PER_QUEUED,
                )

    def _update_statuses(self) -> None:
        """
        Classify each node as HEALTHY or DEGRADED based on queue depth.

        FAILED nodes are never touched here — their status is permanent
        until the agent issues SCALE_UP or the environment resets.
        """
        for node in self._nodes:
            if node.status == NodeStatus.FAILED:
                continue
            if node.queue_depth >= OVERLOAD_THRESHOLD:
                node.status = NodeStatus.DEGRADED
            else:
                node.status = NodeStatus.HEALTHY
