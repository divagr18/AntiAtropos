# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AntiAtropos Environment — Server-Side Implementation.

This module is the central orchestration layer. It:
  1. Receives a typed SREAction from the OpenEnv framework.
  2. Feeds it into the simulator (simulator.py) to advance time by one tick.
  3. Queries stability.py to compute Lyapunov energy and the scalar reward.
  4. Packages the resulting cluster state into a ClusterObservation and returns it.

The environment implements a discrete-time fluid-queue model where stability 
is governed by Lyapunov Drift-Plus-Penalty theory.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SREAction, ClusterObservation, NodeObservation, NodeStatus
    from ..simulator import ClusterSimulator, COST_PER_CAPACITY_UNIT_PER_HOUR
    from ..stability import compute_lyapunov, compute_reward
except ImportError:
    from models import SREAction, ClusterObservation, NodeObservation, NodeStatus  # type: ignore[no-redef]
    from simulator import ClusterSimulator, COST_PER_CAPACITY_UNIT_PER_HOUR  # type: ignore[no-redef]
    from stability import compute_lyapunov, compute_reward  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Reward hyper-parameters (synchronized with stability.py constants)
# ---------------------------------------------------------------------------

ALPHA: float = 1e-5   # Massively scaled down Weight on Lyapunov energy drift ΔV(s)
BETA:  float = 1.0    # Weight on infrastructure cost
GAMMA: float = 1.0    # Weight on SLA violations

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
        self._sim: ClusterSimulator = ClusterSimulator(n_nodes=N_NODES, task_id="task-1")
        self._nodes_true: list[dict] = []
        self._nodes_obs: list[dict] = []
        self._prev_lyapunov: float = 0.0
        self._sla_violations: int = 0

    def reset(self, task_id: str = "task-1") -> ClusterObservation:
        """
        Start a fresh episode with a specific task profile.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = task_id
        self._sla_violations = 0

        # Initialize the production simulator
        self._sim.reset(task_id=task_id)
        self._nodes_true = self._sim.state(for_agent=False)
        self._nodes_obs  = self._sim.state(for_agent=True)
        self._prev_lyapunov = compute_lyapunov(self._nodes_true)

        return self._build_observation()

    def step(self, action: SREAction) -> ClusterObservation:  # type: ignore[override]
        """
        Advance the simulation by one discrete time tick.
        """
        self._state.step_count += 1

        # 1. Apply management action and advance physics
        self._sim.apply_action(action)
        self._sim.tick()
        
        # 2. Extract states (Ground Truth for reward; Observation for agent)
        self._nodes_true = self._sim.state(for_agent=False)
        self._nodes_obs  = self._sim.state(for_agent=True)

        # 3. SLA Check (must happen BEFORE reward so it is synchronized)
        avg_latency = self._avg_latency(self._nodes_true)
        error_rate  = self._error_rate(self._nodes_true)
        if avg_latency > 200.0 or error_rate > 0.05:
            self._sla_violations += 1

        # 4. Compute Lyapunov stability metrics from Ground Truth
        current_lyapunov = compute_lyapunov(self._nodes_true)
        
        # 5. Compute scalar reward using updated counters
        cost = self._compute_cost(self._nodes_true)
        reward = compute_reward(
            v_prev=self._prev_lyapunov,
            v_curr=current_lyapunov,
            cost=cost,
            sla_violations=self._sla_violations,
            alpha=ALPHA,
            beta=BETA,
            gamma=GAMMA
        )
        
        self._prev_lyapunov = current_lyapunov

        # 6. Termination check
        done = (
            self._state.step_count >= MAX_STEPS
            or all(n["status"] == NodeStatus.FAILED for n in self._nodes_true)
        )

        # 7. Package Observation (from OBSERVED state)
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
            # Both live and pending units are billed as infrastructure is provisioned.
            total_capacity_units += int(node.get("capacity_units", 0))
            total_capacity_units += int(node.get("pending_capacity_units", 0))
        return total_capacity_units * COST_PER_CAPACITY_UNIT_PER_HOUR

    def _avg_latency(self, nodes: list[dict]) -> float:
        """Computes mean latency across all non-failed nodes."""
        active = [n for n in nodes if n["status"] != NodeStatus.FAILED]
        if not active:
            return float("inf")
        return sum(n["latency_ms"] for n in active) / len(active)

    def _error_rate(self, nodes: list[dict]) -> float:
        """Calculates the cluster-wide fraction of dropped or lost requests."""
        total_incoming = sum(n.get("incoming_request_rate", 0.0) for n in nodes)
        if total_incoming <= 0:
            return 0.0

        # Sum explicit drops (SHED_LOAD) and traffic lost to failed nodes
        shed_drops = sum(n.get("dropped_requests", 0) for n in nodes)
        failed_drops = sum(
            n.get("incoming_request_rate", 0.0)
            for n in nodes if n["status"] == NodeStatus.FAILED
        )

        return min(1.0, (shed_drops + failed_drops) / total_incoming)

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
                done=False,
                reward=0.0,
            )
            for n in self._nodes_obs
        ]

        # Aggregate metrics for the cluster-level dashboard
        # CRITICAL: We use TRUE state for the objective metrics (grader-facing) 
        # so that sensor dropout (-1.0 latency) doesn't fake a 'good' score.
        return ClusterObservation(
            cluster_id=self._state.episode_id,
            task_id=self._task_id,
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
            done=False,
            reward=0.0,
        )
