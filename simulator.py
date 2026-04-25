# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AntiAtropos Core Simulation Physics.

A discrete-time fluid-queue model simulating a 5-node microservice cluster.
Each node has stateful queues, capacities, and failure probabilities. 
Dynamic traffic is injected per tick, and management actions shift capacity
and routing parameters.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Simulator Constants (World Physics)
# ---------------------------------------------------------------------------

DEFAULT_CAPACITY:     float = 3.0     # Baseline service rate (μ) per node
MAX_CAPACITY:         float = 5.0     # Maximum SCALE_UP limit (halved to close brute-force headroom)
MAX_SCALING_STEP:     int   = 3       # Largest allowed SCALE_UP param (was 5)
BOOT_DELAY_TICKS:     int   = 5       # Time it takes to bring up infrastructure
BASE_LATENCY_MS:      float = 20.0    # Minimum processing time
OVERLOAD_THRESHOLD:   int   = 80      # Request count where node begins to "fail" (DEGRADED)
LATENCY_STEEPNESS:    float = 2.0     # Increased to ensure SLA violations before death
FATAL_FAIL_THRESHOLD: int   = 200     # Hard cap on queue depth (catastrophic failure boundary)
CASCADE_WINDOW_TICKS: int = 3     # Ticks after a failure to check for cascade effects
CASCADE_QUEUE_MULTIPLIER: float = 1.2  # Queue must exceed FATAL_FAIL_THRESHOLD * this to cascade
NODE_RECOVERY_TICKS: int   = 20      # Ticks before a FAILED node auto-recovers
BACKPRESSURE_THRESHOLD: float = 60.0   # Queue depth that triggers backpressure
BACKPRESSURE_MAX_FACTOR: float = 0.4   # Maximum service rate reduction (40%)

SENSOR_DROPOUT_PROB:  float = 0.05    # P(node.queue, latency reports 0 or -1.0)
NODE_FAILURE_PROB:    float = 0.00    # P(node fails naturally) — largely driven by task profile

# Cost model constants
COST_PER_CAPACITY_UNIT_PER_HOUR: float = 0.05

# Task Profiles (Domain Randomization)
# Task 1: Start at 92-99% of ingress capacity (randomised in _randomize_domain).
# DAG ingress capacity = 2 ingress nodes * DEFAULT_CAPACITY * 15 = 90 req/tick.
# lambda_init ≈ 83-89 so each ingress node sees ~41-44 req/tick (just under 45 capacity).
T1_INITIAL_LAMBDA: float = 86.0   # midpoint of [82.8, 89.1]; overridden by _randomize_domain
T1_RAMP_SLOPE:     float = 0.5    # +0.5 req/tick globally per tick
# Task 2: lambda at 100-110% of ingress capacity — guarantees immediate ingress overload.
T2_INITIAL_LAMBDA: float = 95.0   # midpoint of [90, 99]; overridden by _randomize_domain
T2_FAIL_TICK:      int   = 20
T3_INITIAL_LAMBDA: float = 60.0

# Task 3 surge parameters — base window, jitter applied per episode
T3_SURGE_CYCLE:        int   = 60   # Cycle length (ticks)
T3_SURGE_BASE_START:   int   = 30   # Nominal start of surge within cycle
T3_SURGE_BASE_END:     int   = 40   # Nominal end of surge within cycle
T3_SURGE_JITTER:       int   = 10   # ±jitter applied to start/end each episode
T3_SURGE_MAGNITUDE:    float = 140.0 # Extra req/tick added to node-1 and node-2

# Hardening: Critical infrastructure that CANNOT be shed
# In Task 3, these receive the surge. Forcing the agent to SCALE.
CRITICAL_NODES: list[str] = ["node-0", "node-1", "node-2"]

# VIP / business-critical node weights.
# node-0 is the payment portal, so its queue growth or failure matters more.
# Reduced from 4.0 → 2.0 to prevent reward gradient from creating
# a local optimum where the agent only scales node-0.
# At 2×, node-0 is still prioritized but other nodes remain viable targets.
VIP_NODE_WEIGHTS: dict[str, float] = {
    "node-0": 2.0,
}

# ---------------------------------------------------------------------------
# Graph Topology (DAG — fixed 5-node cluster architecture)
# ---------------------------------------------------------------------------

# Directed edges: parent -> list of direct children.
# node-0 (payments/VIP) is the primary ingress; node-4 (auth) is independent.
CLUSTER_TOPOLOGY: dict[str, list[str]] = {
    "node-0": ["node-1", "node-2"],
    "node-1": [],
    "node-2": ["node-3"],
    "node-3": [],
    "node-4": [],
}

# Nodes that receive raw external traffic directly.
EXTERNAL_TRAFFIC_NODES: set[str] = {"node-0", "node-4"}

# 50/50 external λ split between the two ingress nodes.
# node-0 (payments/VIP) and node-4 (auth) each receive half of total_lambda.
# total_lambda is the cluster-wide external arrival rate (req/tick).
# Each ingress node therefore sees total_lambda * 0.5 req/tick at its input.
EXTERNAL_LAMBDA_FRACTION: float = 0.5

# Default upstream-to-downstream routing weights (parent → child splits).
# These represent the baseline traffic split before agent rerouting.
DEFAULT_ROUTING_SPLIT: dict[str, dict[str, float]] = {
    "node-0": {"node-1": 0.5, "node-2": 0.5},
    "node-2": {"node-3": 1.0},
}

# Pre-computed topological order (Kahn's BFS on CLUSTER_TOPOLOGY).
# Ensures parents are always processed before their children in _inject_traffic().
# Order: node-0, node-4 (roots) → node-1, node-2 (node-0 children) → node-3 (node-2 child).
_TOPOLOGICAL_ORDER: tuple[str, ...] = ("node-0", "node-4", "node-1", "node-2", "node-3")


class NodeStatus(str, Enum):
    HEALTHY  = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED   = "FAILED"


@dataclass
class NodeState:
    node_id: str
    status: NodeStatus = NodeStatus.HEALTHY
    is_vip: bool = False
    
    # Physics parameters
    capacity: float = DEFAULT_CAPACITY
    importance_weight: float = 1.0
    queue_depth: float = 0.0
    latency_ms: float = BASE_LATENCY_MS
    incoming_request_rate: float = 0.0
    cpu_utilization: float = 0.0
    
    # Per-tick accounting (reset each tick)
    dropped_requests: float = 0.0
    shed_fraction: float = 0.0       # Fraction of incoming traffic to drop this tick
    pending_capacity_queue: list[int] = field(default_factory=list)
    recovery_timer: int = 0          # Countdown to auto-recovery from FAILED status
    is_scripted_failure: bool = False  # True if failed due to task scripting (no auto-recovery)
    outflow_rate: float = 0.0         # Requests/tick actually dispatched downstream (DAG edge signal)

    # Derived (recomputed whenever capacity or status changes)
    @property
    def service_rate(self) -> float:
        """The total service capacity (μ) in requests per tick."""
        if self.status == NodeStatus.FAILED:
            return 0.0
        return self.capacity * 15.0  # Base unit = 15 req/tick processing power

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "status": self.status,
            "is_vip": self.is_vip,
            "capacity": self.capacity,
            "importance_weight": self.importance_weight,
            "queue_depth": int(self.queue_depth),
            "latency_ms": round(self.latency_ms, 2),
            "incoming_request_rate": round(self.incoming_request_rate, 2),
            "cpu_utilization": round(min(1.0, self.cpu_utilization), 4),
            "dropped_requests": int(self.dropped_requests),
            "shed_fraction": round(self.shed_fraction, 4),
            "capacity_units": int(self.capacity),
            "pending_capacity_units": int(len(self.pending_capacity_queue)),
            "recovery_timer": self.recovery_timer,
            "is_scripted_failure": self.is_scripted_failure,
            "outflow_rate": round(self.outflow_rate, 2),
        }


class ClusterSimulator:
    """
    Multi-node fluid queue simulator.
    
    Operates in discrete ticks. 
    1. Action: Control plane actions (Scaling/Routing/Shedding) are applied.
    2. Tick: Physics engine updates queues based on λ (incoming) and μ (service rate).
    3. Failure Logic: Queue overflows trigger status degradation/node death.
    """

    def __init__(self, n_nodes: int = 5, task_id: str = "task-1", seed: Optional[int] = None):
        self._n_nodes = n_nodes
        self._task_id = task_id
        # Default to non-deterministic RNG seeding so fresh simulator instances
        # do not replay identical domain-randomization sequences.
        # Pass an explicit seed for reproducible experiments.
        self._seed: Optional[int] = seed
        self._rng = random.Random(seed)
        self._tick_count: int = 0
        self._failed_node_id: Optional[str] = None
        self._t1_ramp_slope: float = T1_RAMP_SLOPE
        self._t1_init_lambda: float = T1_INITIAL_LAMBDA
        self._t2_fail_tick: int = T2_FAIL_TICK
        self._t2_init_lambda: float = T2_INITIAL_LAMBDA
        # Task-3 surge window — randomised per episode in _randomize_domain()
        self._t3_surge_start: int = T3_SURGE_BASE_START
        self._t3_surge_end:   int = T3_SURGE_BASE_END
        # Per-node reroute weights for REROUTE_TRAFFIC (node_id → fraction)
        self._reroute_weights: dict[str, float] = {}
        self._cascade_tick: int = 0  # Tick counter for cascade detection window
        self._cascade_triggered: bool = False  # Set True when a NEW overload failure occurs
        self._nodes: list[NodeState] = []
        self.invalid_action_count: int = 0
        self._randomize_domain()
        self._reset_nodes()

    def _randomize_domain(self) -> None:
        """Apply domain randomization for RL robustness across tasks."""
        self._t1_ramp_slope = self._rng.uniform(0.8, 2.0)
        # DAG calibration: total_lambda is split across 2 ingress nodes (node-0, node-4).
        # Each ingress node's capacity is DEFAULT_CAPACITY * 15 req/tick.
        # Ingress cluster capacity = len(EXTERNAL_TRAFFIC_NODES) * DEFAULT_CAPACITY * 15 = 90.
        # Task 1: start between 92-99% of ingress capacity so the ingress nodes are
        # near saturation immediately, producing rich early reward signal.
        n_ingress = len(EXTERNAL_TRAFFIC_NODES)  # 2
        ingress_mu_total = n_ingress * DEFAULT_CAPACITY * 15.0  # 90 req/tick
        self._t1_init_lambda = self._rng.uniform(
            ingress_mu_total * 0.92, ingress_mu_total * 0.99
        )
        self._t2_fail_tick = self._rng.randint(10, 40)
        # Task 2: guarantee immediate ingress overload (slightly above ingress saturation).
        # Each ingress node sees total_lambda/2; target ~102% of individual ingress capacity.
        self._t2_init_lambda = self._rng.uniform(
            ingress_mu_total * 1.00, ingress_mu_total * 1.10
        )
        # Task 3: jitter the surge window so the LLM can't memorise it.
        jitter = self._rng.randint(-T3_SURGE_JITTER, T3_SURGE_JITTER)
        self._t3_surge_start = T3_SURGE_BASE_START + jitter
        self._t3_surge_end   = T3_SURGE_BASE_END   + jitter
        # Reset reroute weights to uniform on each new episode
        self._reroute_weights = {}

    def _reset_nodes(self) -> None:
        self._nodes = [
            NodeState(
                node_id=f"node-{i}",
                is_vip=f"node-{i}" in VIP_NODE_WEIGHTS,
                importance_weight=VIP_NODE_WEIGHTS.get(f"node-{i}", 1.0),
                is_scripted_failure=False,
            )
            for i in range(self._n_nodes)
        ]

    def reset(self, task_id: str = "task-1", seed: Optional[int] = None) -> None:
        """Restart the simulator for a fresh episode."""
        if seed is not None:
            self._seed = seed
            # Reinitialize RNG so episode generation is reproducible for a given seed.
            self._rng = random.Random(seed)
        self._task_id = task_id
        self._tick_count = 0
        self._failed_node_id = None
        self._reroute_weights = {}
        self._cascade_tick = 0
        self._cascade_triggered = False
        self.invalid_action_count = 0
        self._randomize_domain()
        self._reset_nodes()

    def state(self, for_agent: bool = True) -> list[dict]:
        """Return current per-node state as a list of plain dicts."""
        state_list = []
        for n in self._nodes:
            d = n.to_dict()
            if for_agent and self._rng.random() < SENSOR_DROPOUT_PROB:
                d["queue_depth"] = 0
                d["latency_ms"] = -1.0
            state_list.append(d)
        return state_list

    def apply_action(self, action_model) -> None:
        """Update simulator world-state parameters based on agent command."""
        at = action_model.action_type if hasattr(action_model, "action_type") else action_model["action_type"]
        node_id = action_model.target_node_id if hasattr(action_model, "target_node_id") else action_model["target_node_id"]
        param = action_model.parameter if hasattr(action_model, "parameter") else action_model["parameter"]

        # 1. Target node lookup
        target = next((n for n in self._nodes if n.node_id == node_id), None)
        if not target:
            return

        # 2. Command implementation
        if at == "SCALE_UP":
            delta = max(1, int(param * MAX_SCALING_STEP))
            current_max = target.capacity + len(target.pending_capacity_queue)
            actual_delta = int(min(delta, MAX_CAPACITY - current_max))
            for _ in range(actual_delta):
                target.pending_capacity_queue.append(BOOT_DELAY_TICKS)
            
            # If scaling up, we forcefully 'repair' DEGRADED status (SRE manual intervention)
            if target.status == NodeStatus.DEGRADED:
                target.status = NodeStatus.HEALTHY

        elif at == "SCALE_DOWN":
            delta = max(1, int(param * MAX_SCALING_STEP))
            target.capacity = max(1, target.capacity - delta)

        elif at == "REROUTE_TRAFFIC":
            # Physically offload traffic FROM the target node by proportion `param`.
            # `param` ∈ [0, 1]: fraction of target node's share to redistribute.
            # The shed amount is split evenly across healthy peer nodes.
            frac = min(1.0, max(0.0, float(param)))
            # Record a reroute weight for the target so _inject_traffic() can
            # reduce its lambda and bump peers accordingly.
            self._reroute_weights[node_id] = frac

        elif at == "SHED_LOAD":
            # Rule: Cannot shed critical nodes (database/control plane).
            # This forces the agent to handle Task-3 surge via Scaling/Rerouting.
            if node_id in CRITICAL_NODES:
                self.invalid_action_count += 1
                return 

            frac = min(1.0, param)
            target.shed_fraction = frac
            # Note: physically applied in _update_queues() to incoming traffic

    def tick(self) -> None:
        """
        Advance simulated time by one step.
        Evaluates physics: Capacity growth, Traffic injection, Queue processing.
        """
        self._tick_count += 1
        self._update_capacity()
        self._inject_traffic()
        # Save original capacities; backpressure temporarily reduces service_rate
        # for this tick only. Restore after _update_queues so the reduction does
        # not compound across ticks and permanently cripple parent nodes.
        saved_capacities = {n.node_id: n.capacity for n in self._nodes}
        self._apply_backpressure()
        # Reset per-tick shed counters before physics update
        for node in self._nodes:
            node.dropped_requests = 0.0
        self._update_queues()
        # Restore capacities so the next tick starts from the true provisioned level
        for n in self._nodes:
            n.capacity = saved_capacities.get(n.node_id, n.capacity)
        self._update_derived_metrics()
        self._update_statuses()
        self._cascade_failures()
        self._process_recovery()
        # Decay shed fractions gradually
        # *= 0.5 retains 50% per tick (fast decay).
        for node in self._nodes:
            node.shed_fraction *= 0.5
            if node.shed_fraction < 0.05:
                node.shed_fraction = 0.0

    def _update_capacity(self) -> None:
        """Process pending capacity from SCALE_UP actions"""
        for node in self._nodes:
            for i in range(len(node.pending_capacity_queue)):
                node.pending_capacity_queue[i] -= 1
            
            # Move ready capacity to live
            ready = sum(1 for delay in node.pending_capacity_queue if delay <= 0)
            node.capacity += ready
            node.pending_capacity_queue = [delay for delay in node.pending_capacity_queue if delay > 0]

    def _inject_traffic(self) -> None:
        """
        Distribute traffic through the cluster DAG in three phases.

        Phase 1 — Task lambda + scripted events:
            Compute total external λ for this tick and apply any task-specific
            mutations (node failure scripting, surge flags).  No early returns;
            all branches fall through to the shared DAG in Phase 2.

        Phase 2 — Topological DAG distribution:
            Traverse _TOPOLOGICAL_ORDER (roots first).  Each parent's
            processed outflow (min(incoming, service_rate)) is split across
            its children via DEFAULT_ROUTING_SPLIT.  A FAILED node has
            service_rate=0, so outflow=0 and its children are naturally
            starved — this is the causal failure chain the RL agent must
            learn to route around.

        Phase 3 — Reroute weight correction:
            Apply REROUTE_TRAFFIC weight adjustments post-DAG, then decay
            weights.  Keeps reroute semantics identical to pre-DAG behaviour.
        """
        # -------------------------------------------------------------------
        # Phase 1: task-specific lambda + scripted events (no early returns)
        # -------------------------------------------------------------------
        total_lambda: float = 0.0
        # direct_injections: extra traffic added directly to a node ON TOP OF
        # the DAG distribution.  Used for Task-3 surge bursts that model a
        # side-channel load source (e.g. bulk import hitting checkout/catalog
        # directly), while the base λ still travels through node-0 as ingress.
        direct_injections: dict[str, float] = {}

        if self._task_id == "task-1":
            # Linear ramp — starts near cluster capacity
            total_lambda = self._t1_init_lambda + (self._t1_ramp_slope * self._tick_count)

        elif self._task_id == "task-2":
            total_lambda = self._t2_init_lambda
            # Scripted node failure fires at the configured tick
            if self._tick_count >= self._t2_fail_tick and not self._failed_node_id:
                self._failed_node_id = self._rng.choice(
                    [n.node_id for n in self._nodes if n.node_id != "node-0"]
                )
                target = next((n for n in self._nodes if n.node_id == self._failed_node_id), None)
                if target:
                    target.is_scripted_failure = True
            # No early return: DAG distributes traffic to the failed node normally.
            # The dead node's service_rate=0 means outflow=0, so its children are
            # starved. _update_queues() converts all its incoming traffic to
            # dropped_requests.  The agent must issue REROUTE_TRAFFIC to shift
            # the parent's split away from the dead child.

        elif self._task_id == "task-3":
            total_lambda = T3_INITIAL_LAMBDA
            phase = self._tick_count % T3_SURGE_CYCLE
            if self._t3_surge_start <= phase <= self._t3_surge_end:
                # Surge is modelled as a direct external burst arriving at the
                # checkout (node-1) and catalog (node-2) services from a side
                # channel that bypasses the payment gateway ingress.
                # Base λ still routes through the DAG; the surge is overlaid so
                # CRITICAL_NODE protections (no SHED_LOAD on node-1/2) still apply.
                for nid in ["node-1", "node-2"]:
                    direct_injections[nid] = T3_SURGE_MAGNITUDE

        # -------------------------------------------------------------------
        # Phase 2: DAG topological distribution
        # -------------------------------------------------------------------
        node_incoming: dict[str, float] = {n.node_id: 0.0 for n in self._nodes}
        node_map: dict[str, "NodeState"] = {n.node_id: n for n in self._nodes}

        # Seed ingress nodes with their share of external λ
        node_incoming["node-0"] = total_lambda * EXTERNAL_LAMBDA_FRACTION
        node_incoming["node-4"] = total_lambda * (1.0 - EXTERNAL_LAMBDA_FRACTION)

        # Overlay task-specific direct injections (Task-3 surge)
        for nid, extra in direct_injections.items():
            node_incoming[nid] = node_incoming.get(nid, 0.0) + extra

        # Propagate outflow through the graph in topological order
        for parent_id in _TOPOLOGICAL_ORDER:
            parent = node_map.get(parent_id)
            if parent is None:
                continue
            parent.incoming_request_rate = node_incoming[parent_id]
            # Outflow = requests the parent actually forwards downstream.
            # FAILED nodes have service_rate=0 → outflow=0 → children starved.
            outflow = min(parent.incoming_request_rate, parent.service_rate)
            parent.outflow_rate = outflow
            for child_id, split in DEFAULT_ROUTING_SPLIT.get(parent_id, {}).items():
                node_incoming[child_id] = node_incoming.get(child_id, 0.0) + outflow * split

        # -------------------------------------------------------------------
        # Phase 3: REROUTE_TRAFFIC weight corrections (post-DAG)
        # -------------------------------------------------------------------
        self._apply_reroute_weights()

        # Recalculate outflow after reroute so the agent sees accurate
        # per-node dispatch rates.  Without this, a node whose incoming was
        # halved by reroute would still report its pre-reroute outflow.
        for n in self._nodes:
            n.outflow_rate = min(n.incoming_request_rate, n.service_rate)

    def _apply_reroute_weights(self) -> None:
        """
        Apply REROUTE_TRAFFIC adjustments.

        For each node with a reroute weight w, reduce its incoming_request_rate
        by (w × its current rate) and distribute that offloaded load equally
        across healthy peer nodes.

        Important: rerouting must still work when the target is FAILED (Task 2).
        We therefore allow offload FROM failed targets. Only absorber nodes are
        constrained to non-failed nodes.

        Decays each reroute weight by 50% per tick so the effect must be
        re-issued to be maintained (forces the agent to act).
        """
        if not self._reroute_weights:
            return

        total_overflow: float = 0.0
        rerouted_ids = set(self._reroute_weights.keys())

        for n in self._nodes:
            w = self._reroute_weights.get(n.node_id, 0.0)
            if w <= 0.0:
                continue

            # Offload from the target regardless of node status.
            offload = n.incoming_request_rate * w
            if offload <= 0.0:
                continue
            n.incoming_request_rate -= offload
            total_overflow += offload

        # Spread overflow across all non-rerouted healthy nodes
        if total_overflow > 0:
            absorbers = [
                n for n in self._nodes
                if n.node_id not in rerouted_ids
                and n.status != NodeStatus.FAILED
            ]
            # If every healthy node is rerouted this tick, fall back to all
            # healthy non-rerouted nodes first, then finally all healthy nodes.
            # CRITICAL: A FAILED node should NEVER be an absorber.
            if not absorbers:
                absorbers = [n for n in self._nodes if n.status != NodeStatus.FAILED and n.node_id not in rerouted_ids]
            if not absorbers:
                absorbers = [n for n in self._nodes if n.status != NodeStatus.FAILED]
            
            if absorbers:
                share = total_overflow / len(absorbers)
                for n in absorbers:
                    n.incoming_request_rate += share

        # Decay weights — agent must keep re-issuing to maintain effect
        # *= 0.5 retains 50% per tick (fast decay).
        for nid in list(self._reroute_weights.keys()):
            self._reroute_weights[nid] *= 0.5
            if self._reroute_weights[nid] < 0.05:
                del self._reroute_weights[nid]

    def _apply_backpressure(self) -> None:
        """Reduce parent service rate when children are overloaded."""
        for parent_id, children in CLUSTER_TOPOLOGY.items():
            parent = next((n for n in self._nodes if n.node_id == parent_id), None)
            if not children or not parent or parent.status == NodeStatus.FAILED:
                continue
            
            # Compute pressure from overloaded children
            total_pressure = 0.0
            for child_id in children:
                child = next((n for n in self._nodes if n.node_id == child_id), None)
                if child:
                    excess = max(0.0, child.queue_depth - BACKPRESSURE_THRESHOLD)
                    total_pressure += excess / FATAL_FAIL_THRESHOLD  # normalise to [0, 1]
            
            # Reduce parent's effective capacity proportionally
            pressure_factor = min(BACKPRESSURE_MAX_FACTOR, total_pressure * 0.6)
            parent.capacity = max(1.0, parent.capacity * (1.0 - pressure_factor))

    def _update_queues(self) -> None:
        """
        Fluid-queue update for all nodes.

            Q_i(t+1) = max(Q_i(t) + λ_eff_i(t) − μ_i(t), 0)

        where λ_eff = λ_incoming * (1 - shed_fraction).
        """
        for node in self._nodes:
            if node.status == NodeStatus.FAILED:
                node.queue_depth = 0.0
                node.dropped_requests = node.incoming_request_rate
                continue

            # Apply shedding to incoming traffic
            lambda_eff = node.incoming_request_rate * (1.0 - node.shed_fraction)
            node.dropped_requests = node.incoming_request_rate - lambda_eff

            excess = lambda_eff - node.service_rate
            node.queue_depth = max(0.0, node.queue_depth + excess)

    def _update_derived_metrics(self) -> None:
        """Compute CPU and Latency based on current queue and status."""
        for n in self._nodes:
            if n.status == NodeStatus.FAILED:
                n.cpu_utilization = 0.0
                n.latency_ms = float("inf")
                continue

            # Utilization = Ratio of λ to μ
            service_rate = n.service_rate
            n.cpu_utilization = n.incoming_request_rate / service_rate if service_rate > 0 else 1.0

            # Latency: Hybrid M/M/1 + backlog term
            # M/M/1 gives exponential blow-up as utilization->1 (the "hockey stick")
            # Backlog term ensures queue_depth still contributes signal even when
            # utilization is capped at 0.99, preventing the flattening problem.
            utilization = min(0.99, n.cpu_utilization)  # cap to prevent infinity
            mm1_latency = BASE_LATENCY_MS / (1.0 - utilization)
            backlog_latency = n.queue_depth * LATENCY_STEEPNESS
            n.latency_ms = mm1_latency + backlog_latency

    def _update_statuses(self) -> None:
        """Transition node health based on queue boundaries.

        Recovery rules:
        - Scripted failures (Task 2 forced node kill): permanent, never auto-recover.
          Marked by is_scripted_failure=True, recovery_timer=0.
        - Overload failures (queue > FATAL_FAIL_THRESHOLD): auto-recover after
          NODE_RECOVERY_TICKS. The agent can learn to reroute away and let the
          node heal.
        """
        for n in self._nodes:
            # Scripted (task-forced) failures are permanent
            if n.is_scripted_failure:
                n.status = NodeStatus.FAILED
                n.recovery_timer = 0
                continue

            if n.queue_depth > FATAL_FAIL_THRESHOLD:
                if n.status != NodeStatus.FAILED:
                    n.status = NodeStatus.FAILED
                    n.recovery_timer = NODE_RECOVERY_TICKS
                    n.capacity = 0.5   # starts at half capacity when recovery begins
                    self._cascade_triggered = True  # Signal cascade detection
            elif n.queue_depth > OVERLOAD_THRESHOLD:
                n.status = NodeStatus.DEGRADED
            elif n.status == NodeStatus.DEGRADED and n.queue_depth < (OVERLOAD_THRESHOLD / 2):
                n.status = NodeStatus.HEALTHY

    def _cascade_failures(self) -> None:
        """Detect cascading failure: if a peer node's queue exceeds a heightened
        threshold within CASCADE_WINDOW_TICKS of a *new* failure, degrade it.

        Guardrails:
        - Only triggers when a NEW failure occurred this tick (not any failed node).
        - Graph-bounded: cascade only propagates along edges (parents or children).
        - Scripted failures (Task 2) do not trigger cascades.
        """
        if not self._cascade_triggered:
            self._cascade_tick = 0
            return

        self._cascade_tick += 1
        if self._cascade_tick > CASCADE_WINDOW_TICKS:
            self._cascade_triggered = False
            self._cascade_tick = 0
            return

        # Find all currently failed nodes
        failed_ids = {n.node_id for n in self._nodes if n.status == NodeStatus.FAILED}
        
        # Build set of nodes adjacent to any failed node (upstream or downstream)
        at_risk = set()
        for failed_id in failed_ids:
            # Downstream children of the failed node
            at_risk.update(CLUSTER_TOPOLOGY.get(failed_id, []))
            # Upstream parents of the failed node
            for parent_id, children in CLUSTER_TOPOLOGY.items():
                if failed_id in children:
                    at_risk.add(parent_id)

        cascade_threshold = FATAL_FAIL_THRESHOLD * CASCADE_QUEUE_MULTIPLIER
        for n in self._nodes:
            if n.node_id not in at_risk:
                continue   # Not adjacent to failure — cannot cascade
            if n.status == NodeStatus.FAILED or n.is_scripted_failure:
                continue
            if n.queue_depth > cascade_threshold:
                n.status = NodeStatus.DEGRADED

    def _process_recovery(self) -> None:
        """Count down recovery timers and bring FAILED nodes back online.

        Only overload-failed nodes (recovery_timer > 0) can recover.
        Scripted failures (is_scripted_failure=True) are excluded.
        """
        RECOVERY_RAMP_PER_TICK: float = 0.5   # capacity added per tick during recovery
        
        for n in self._nodes:
            if n.is_scripted_failure:
                continue
            # Check recovery_timer > 0, not status == FAILED: the first recovery
            # tick transitions the node to DEGRADED, but the timer must keep
            # counting until it reaches 0 and the node becomes HEALTHY.
            if n.recovery_timer > 0:
                n.recovery_timer -= 1
                if n.recovery_timer <= 0:
                    n.status = NodeStatus.HEALTHY
                    n.queue_depth = 0.0
                    n.latency_ms = BASE_LATENCY_MS
                    n.cpu_utilization = 0.0
                    # capacity stays at whatever it ramped to
                else:
                    # Still in recovery: ramp capacity up, stay DEGRADED
                    n.capacity = min(DEFAULT_CAPACITY, n.capacity + RECOVERY_RAMP_PER_TICK)
                    n.status = NodeStatus.DEGRADED   # not HEALTHY until fully ramped

    def reconcile_state(self, telemetry_map: dict) -> None:
        """
        Reconcile internal simulator state with external telemetry signals.
        Used in 'hybrid' or 'live' modes to align the physics engine with reality.
        
        telemetry_map: node_id -> TelemetryRecord (or dict)
        """
        for node in self._nodes:
            if node.node_id in telemetry_map:
                record = telemetry_map[node.node_id]
                # If record is an object (TelemetryRecord), access fields; otherwise treat as dict
                if hasattr(record, "queue_depth"):
                    q_ext = float(record.queue_depth)
                    r_ext = float(record.request_rate)
                    c_ext = float(record.cpu_utilization)
                    e_ext = float(record.error_rate)
                    l_ext = float(record.latency_ms)
                else:
                    q_ext = float(record.get("queue_depth", node.queue_depth))
                    r_ext = float(record.get("request_rate", node.incoming_request_rate))
                    c_ext = float(record.get("cpu_utilization", node.cpu_utilization))
                    e_ext = float(record.get("error_rate", 0.0))
                    l_ext = float(record.get("latency_ms", node.latency_ms))

                # Smoothly blend the external state into physics to prevent step jumps
                # trust: 0.7 (reality) / 0.3 (simulation prediction)
                node.queue_depth = (node.queue_depth * 0.3) + (q_ext * 0.7)
                node.incoming_request_rate = (node.incoming_request_rate * 0.3) + (r_ext * 0.7)
                node.cpu_utilization = (node.cpu_utilization * 0.3) + (c_ext * 0.7)
                node.latency_ms = (node.latency_ms * 0.3) + (l_ext * 0.7)

                # Status reconciliation: Fail the node if error rate is high
                if e_ext > 0.5:
                    node.status = NodeStatus.FAILED
                elif e_ext > 0.1 and node.status == NodeStatus.HEALTHY:
                    node.status = NodeStatus.DEGRADED
                elif e_ext <= 0.05 and node.status == NodeStatus.DEGRADED:
                    # Allow recovery if telemetry says it's clean (physics will still check queue)
                    node.status = NodeStatus.HEALTHY

        # Crucial: re-derive metrics so latency/cpu are consistent (except for the blended values)
        # Note: We blend latency/cpu above, but _update_derived_metrics might overwrite them.
        # So we update them after blending if we want physics to win, or before if telemetry wins.
        # Usually, for SRE dashboard, we want the blended 'reality'.
        # However, _update_derived_metrics is used to compute 'current' state in pure sim.
        # We'll skip it if we just reconciled to keep the blended values, OR refine it.
        # For now, let's just make sure statuses are updated based on new queue depths.
        self._update_statuses()
