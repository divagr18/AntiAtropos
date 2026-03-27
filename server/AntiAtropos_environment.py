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
    from ..simulator import ClusterSimulator
    from ..stability import compute_lyapunov, compute_reward
except ImportError:
    from models import SREAction, ClusterObservation, NodeObservation, NodeStatus  # type: ignore[no-redef]
    from simulator import ClusterSimulator  # type: ignore[no-redef]
    from stability import compute_lyapunov, compute_reward  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Reward hyper-parameters (synchronized with stability.py constants)
# ---------------------------------------------------------------------------

ALPHA: float = 1.0   # Weight on Lyapunov energy drift ΔV(s)
BETA:  float = 0.05  # Weight on infrastructure cost
GAMMA: float = 2.0   # Weight on SLA violations

MAX_STEPS: int = 100      # Episode length
N_NODES:   int = 5        # Cluster size
COST_PER_NODE_PER_HOUR: float = 0.10  # Baseline USD/hr per active node


class AntiAtroposEnvironment(Environment):
    """
    Autonomous SRE simulation environment.

    The agent observes a microservice cluster and issues management commands
    (SCALE_UP, SCALE_DOWN, REROUTE_TRAFFIC, SHED_LOAD, NO_OP) each step.
    The environment advances one discrete time-tick per step, computes the
    Lyapunov reward, and returns the updated ClusterObservation.

    Reward formula:
        R_t = -(α·ΔV(s)  +  β·Cost  +  γ·SLA_Violations)

    where:
        V(s) = Σ Q_i²  (sum of squared queue depths — Lyapunov energy)

    Stability is the primary objective (α term). Cost optimisation and SLA
    compliance are secondary regularisers (β, γ terms).

    Tasks (set via reset(task_id=...)):
        task-1  Predictive Scaling  — linearly rising traffic, keep latency <200ms.
        task-2  Fault Tolerance     — random node failure, reroute before queues explode.
        task-3  Stability Under Surge — stochastic DDoS burst, shed load on non-critical nodes.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialise environment metadata and the simulation core."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: str = "task-1"
        self._sim: ClusterSimulator = ClusterSimulator(n_nodes=N_NODES, task_id="task-1")
        self._nodes: list[dict] = []
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
        self._sim = ClusterSimulator(n_nodes=N_NODES, task_id=task_id)
        self._nodes = self._sim.state()
        self._prev_lyapunov = compute_lyapunov(self._nodes)

        return self._build_observation()

    def step(self, action: SREAction) -> ClusterObservation:  # type: ignore[override]
        """
        Advance the simulation by one discrete time tick.
        """
        self._state.step_count += 1

        # 1 & 2: Apply management action and advance physics
        self._sim.apply_action(action)
        self._sim.tick()
        self._nodes = self._sim.state()

        # 3: Compute Lyapunov stability metrics
        current_lyapunov = compute_lyapunov(self._nodes)
        
        # 4: Compute scalar reward using the stability layer logic
        cost = self._compute_cost(self._nodes)
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

        # 5: SLA Check
        avg_latency = self._avg_latency(self._nodes)
        error_rate  = self._error_rate(self._nodes)
        if avg_latency > 200.0 or error_rate > 0.05:
            self._sla_violations += 1

        # 6: Termination check
        done = (
            self._state.step_count >= MAX_STEPS
            or all(n["status"] == NodeStatus.FAILED for n in self._nodes)
        )

        # 7: Package Observation
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

    def _compute_cost(self, nodes: list[dict]) -> float:
        """Calculates current running cost based on active nodes."""
        active = sum(1 for n in nodes if n["status"] != NodeStatus.FAILED)
        return active * COST_PER_NODE_PER_HOUR

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
        """Assembles the ClusterObservation from the current simulator state."""
        node_obs = [
            NodeObservation(
                node_id=n["node_id"],
                status=n["status"],
                queue_depth=n["queue_depth"],
                latency_ms=n["latency_ms"],
                incoming_request_rate=n["incoming_request_rate"],
                cpu_utilization=n["cpu_utilization"],
                done=False,
                reward=0.0,
            )
            for n in self._nodes
        ]

        return ClusterObservation(
            cluster_id=self._state.episode_id,
            task_id=self._task_id,
            active_nodes=sum(1 for n in self._nodes if n["status"] != NodeStatus.FAILED),
            average_latency_ms=self._avg_latency(self._nodes),
            error_rate=self._error_rate(self._nodes),
            total_queue_backlog=sum(n["queue_depth"] for n in self._nodes),
            current_cost_per_hour=self._compute_cost(self._nodes),
            lyapunov_energy=self._prev_lyapunov,
            nodes=node_obs,
            step=self._state.step_count,
            max_steps=MAX_STEPS,
            sla_violations=self._sla_violations,
            done=False,
            reward=0.0,
        )