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
MAX_CAPACITY:         float = 10.0    # Maximum SCALE_UP limit
MAX_SCALING_STEP:     int   = 5       # Largest allowed SCALE_UP param
BOOT_DELAY_TICKS:     int   = 5       # Time it takes to bring up infrastructure
BASE_LATENCY_MS:      float = 20.0    # Minimum processing time
OVERLOAD_THRESHOLD:   int   = 80      # Request count where node begins to "fail" (DEGRADED)
LATENCY_STEEPNESS:    float = 0.5     # M/M/1-like queue-to-latency scaling factor
FATAL_FAIL_THRESHOLD: int   = 200     # Hard cap on queue depth (catastrophic failure boundary)

SENSOR_DROPOUT_PROB:  float = 0.05    # P(node.queue, latency reports 0 or -1.0)
NODE_FAILURE_PROB:    float = 0.00    # P(node fails naturally) — largely driven by task profile

# Cost model constants
COST_PER_CAPACITY_UNIT_PER_HOUR: float = 0.05

# Task Profiles (Domain Randomization)
T1_INITIAL_LAMBDA: float = 35.0
T1_RAMP_SLOPE:     float = 1.0  # +1 req per tick globally
T2_INITIAL_LAMBDA: float = 35.0
T2_FAIL_TICK:      int   = 25
T3_INITIAL_LAMBDA: float = 30.0


class NodeStatus(str, Enum):
    HEALTHY  = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED   = "FAILED"


@dataclass
class NodeState:
    node_id: str
    status: NodeStatus = NodeStatus.HEALTHY
    
    # Physics parameters
    capacity: float = DEFAULT_CAPACITY
    queue_depth: float = 0.0
    latency_ms: float = BASE_LATENCY_MS
    incoming_request_rate: float = 0.0
    cpu_utilization: float = 0.0
    
    # Per-tick accounting (reset each tick)
    dropped_requests: float = 0.0
    shed_fraction: float = 0.0       # Fraction of incoming traffic to drop this tick
    pending_capacity_queue: list[int] = field(default_factory=list)

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
            "capacity": self.capacity,
            "queue_depth": int(self.queue_depth),
            "latency_ms": round(self.latency_ms, 2),
            "incoming_request_rate": round(self.incoming_request_rate, 2),
            "cpu_utilization": round(min(1.0, self.cpu_utilization), 4),
            "dropped_requests": int(self.dropped_requests),
            "shed_fraction": round(self.shed_fraction, 4),
            "capacity_units": int(self.capacity),
            "pending_capacity_units": int(len(self.pending_capacity_queue)),
        }


class ClusterSimulator:
    """
    Multi-node fluid queue simulator.
    
    Operates in discrete ticks. 
    1. Action: Control plane actions (Scaling/Routing/Shedding) are applied.
    2. Tick: Physics engine updates queues based on λ (incoming) and μ (service rate).
    3. Failure Logic: Queue overflows trigger status degradation/node death.
    """

    def __init__(self, n_nodes: int = 5, task_id: str = "task-1", seed: int = 42):
        self._n_nodes = n_nodes
        self._task_id = task_id
        self._rng = random.Random(seed)
        self._tick_count: int = 0
        self._failed_node_id: Optional[str] = None
        self._t1_ramp_slope: float = T1_RAMP_SLOPE
        self._t1_init_lambda: float = T1_INITIAL_LAMBDA
        self._t2_fail_tick: int = T2_FAIL_TICK
        self._t2_init_lambda: float = T2_INITIAL_LAMBDA
        self._nodes: list[NodeState] = []
        self._randomize_domain()
        self._reset_nodes()

    def _randomize_domain(self) -> None:
        """Apply domain randomization for RL robustness across tasks."""
        self._t1_ramp_slope = self._rng.uniform(0.5, 2.0)
        self._t1_init_lambda = self._rng.uniform(30.0, 45.0)
        self._t2_fail_tick = self._rng.randint(20, 70)
        self._t2_init_lambda = self._rng.uniform(30.0, 40.0)

    def _reset_nodes(self) -> None:
        self._nodes = [
            NodeState(node_id=f"node-{i}") 
            for i in range(self._n_nodes)
        ]

    def reset(self, task_id: str = "task-1") -> None:
        """Restart the simulator for a fresh episode."""
        self._task_id = task_id
        self._tick_count = 0
        self._failed_node_id = None
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
            actual_delta = min(delta, MAX_CAPACITY - current_max)
            for _ in range(actual_delta):
                target.pending_capacity_queue.append(BOOT_DELAY_TICKS)
            
            # If scaling up, we forcefully 'repair' DEGRADED status (SRE manual intervention)
            if target.status == NodeStatus.DEGRADED:
                target.status = NodeStatus.HEALTHY

        elif at == "SCALE_DOWN":
            delta = max(1, int(param * MAX_SCALING_STEP))
            target.capacity = max(1, target.capacity - delta)

        elif at == "REROUTE_TRAFFIC":
            # In a real SRE scenario, we'd shift a percentage of incoming traffic 
            # away from a node. In this sim, we record routing preferences 
            # used during _inject_traffic().
            # Note: physically handled by environment.py as this is a routing-layer change.
            pass

        elif at == "SHED_LOAD":
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
        # Reset per-tick shed counters before physics update
        for node in self._nodes:
            node.dropped_requests = 0.0
        self._update_queues()
        self._update_derived_metrics()
        self._update_statuses()
        # decay/reset shed fractions for next tick
        for node in self._nodes:
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
        """Determine λ_i per node based on task and routing state."""
        total_lambda = 0.0

        if self._task_id == "task-1":
            # Task 1: Linear Ramp
            total_lambda = self._t1_init_lambda + (self._t1_ramp_slope * self._tick_count)
        elif self._task_id == "task-2":
            # Task 2: Fault Tolerance
            total_lambda = self._t2_init_lambda
            if self._tick_count >= self._t2_fail_tick and not self._failed_node_id:
                # Kill a node!
                self._failed_node_id = self._rng.choice([n.node_id for n in self._nodes if n.node_id != "node-0"])
        elif self._task_id == "task-3":
            # Task 3: Surge
            total_lambda = T3_INITIAL_LAMBDA
            # Periodic surge of +70 items to node-1, node-2
            if 30 <= self._tick_count % 60 <= 40:
                surge = 70.0
                for n in self._nodes:
                    if n.node_id in ["node-1", "node-2"]:
                        n.incoming_request_rate = (total_lambda / self._n_nodes) + surge
                        continue
                    n.incoming_request_rate = total_lambda / self._n_nodes
                return

        # Distribute traffic evenly (default baseline)
        for n in self._nodes:
            if n.node_id == self._failed_node_id:
                n.status = NodeStatus.FAILED
                n.incoming_request_rate = total_lambda / self._n_nodes
            else:
                n.incoming_request_rate = total_lambda / self._n_nodes

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
            
            # Latency (simplified M/M/1 wait-time model)
            n.latency_ms = BASE_LATENCY_MS + (n.queue_depth * LATENCY_STEEPNESS)

    def _update_statuses(self) -> None:
        """Transition node health based on queue boundaries."""
        for n in self._nodes:
            if n.node_id == self._failed_node_id:
                n.status = NodeStatus.FAILED
                continue

            if n.queue_depth > FATAL_FAIL_THRESHOLD:
                n.status = NodeStatus.FAILED
            elif n.queue_depth > OVERLOAD_THRESHOLD:
                n.status = NodeStatus.DEGRADED
            elif n.status == NodeStatus.DEGRADED and n.queue_depth < (OVERLOAD_THRESHOLD / 2):
                n.status = NodeStatus.HEALTHY
