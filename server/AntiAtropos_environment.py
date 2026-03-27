# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AntiAtropos Environment — Server-Side Implementation.

This module is the central orchestration layer.  It:
  1. Receives a typed SREAction from the OpenEnv framework.
  2. Feeds it into the simulator (simulator.py) to advance time by one tick.
  3. Queries stability.py to compute Lyapunov energy and the scalar reward.
  4. Packages the resulting cluster state into a ClusterObservation and returns it.

Simulator and stability modules are NOT imported yet — they are stubbed with
TODO markers so the math can be filled in as Phase 2 & 3 without breaking this
file.  The environment will run and return valid observations immediately.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SREAction, ClusterObservation, NodeObservation, NodeStatus
except ImportError:
    from models import SREAction, ClusterObservation, NodeObservation, NodeStatus


# ---------------------------------------------------------------------------
# Reward hyper-parameters (tuned in Phase 3 alongside stability.py)
# ---------------------------------------------------------------------------

ALPHA: float = 1.0   # Weight on Lyapunov energy drift  ΔV(s)
BETA:  float = 0.05  # Weight on cost penalty
GAMMA: float = 2.0   # Weight on SLA violation count

MAX_STEPS: int = 100      # Episode length (steps)
N_NODES:   int = 5        # Default cluster size
COST_PER_NODE_PER_HOUR: float = 0.10  # USD/hr per active node


# ---------------------------------------------------------------------------
# Main Environment Class
# ---------------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def __init__(self):
        """
        Initialise the environment.

        Note: actual simulator state is set in reset().  __init__ only
        establishes the Python-level attributes so the object is valid before
        a reset() call comes in from the OpenEnv framework.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id: str = "task-1"
        self._nodes: list[dict] = []    # Internal mutable node state (simulator.py will own this)
        self._prev_lyapunov: float = 0.0
        self._sla_violations: int = 0

    # -----------------------------------------------------------------------
    # reset()
    # -----------------------------------------------------------------------

    def reset(self, task_id: str = "task-1") -> ClusterObservation:
        """
        Start a fresh episode.

        Reinitialises the simulator to the starting conditions for `task_id`
        and returns the very first ClusterObservation (step 0).

        Args:
            task_id: Which scenario to run.  One of 'task-1', 'task-2', 'task-3'.

        Returns:
            ClusterObservation representing the cluster at t=0.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_id = task_id
        self._sla_violations = 0

        # ------------------------------------------------------------------
        # TODO (Phase 2): Replace this stub with simulator.py initialisation.
        #
        #   from ..simulator import ClusterSimulator
        #   self._sim = ClusterSimulator(n_nodes=N_NODES, task_id=task_id)
        #   raw_state = self._sim.state()
        #
        # For now, we build a placeholder initial state so the server boots.
        # ------------------------------------------------------------------
        self._nodes = self._build_initial_nodes(N_NODES)
        self._prev_lyapunov = self._compute_lyapunov(self._nodes)

        return self._build_observation()

    # -----------------------------------------------------------------------
    # step()
    # -----------------------------------------------------------------------

    def step(self, action: SREAction) -> ClusterObservation:  # type: ignore[override]
        """
        Advance the simulation by one discrete time tick.

        Execution order each tick:
            1. Apply action to the simulator (mutate node state).
            2. Advance the simulator clock (queue accumulation, traffic events).
            3. Compute the new Lyapunov energy V(s_t).
            4. Compute reward R_t = -(α·ΔV + β·Cost + γ·SLA_violations).
            5. Check SLA conditions and increment violation counter.
            6. Check episode termination (max_steps reached or cluster fully failed).
            7. Package and return ClusterObservation.

        Args:
            action: The SREAction chosen by the agent this tick.

        Returns:
            ClusterObservation containing the updated cluster state,
            the scalar reward, and a done flag.
        """
        self._state.step_count += 1

        # ------------------------------------------------------------------
        # STEP 1 & 2 — Apply action + advance simulator clock
        # TODO (Phase 2): Replace stubs with real simulator calls.
        #
        #   self._sim.apply_action(action)
        #   self._sim.tick()
        #   self._nodes = self._sim.state()
        # ------------------------------------------------------------------
        self._nodes = self._stub_apply_action(self._nodes, action)
        self._nodes = self._stub_tick(self._nodes)

        # ------------------------------------------------------------------
        # STEP 3 — Compute Lyapunov energy V(s_t) = Σ Q_i²
        # TODO (Phase 3): Replace with stability.py
        #
        #   from ..stability import compute_lyapunov, compute_barrier
        #   current_lyapunov = compute_lyapunov(self._nodes)
        # ------------------------------------------------------------------
        current_lyapunov = self._compute_lyapunov(self._nodes)
        delta_v = current_lyapunov - self._prev_lyapunov
        self._prev_lyapunov = current_lyapunov

        # ------------------------------------------------------------------
        # STEP 4 — Compute scalar reward
        # R_t = -(α·ΔV(s) + β·Cost + γ·SLA_Violations)
        # ------------------------------------------------------------------
        cost = self._compute_cost(self._nodes)
        reward = -(ALPHA * delta_v + BETA * cost + GAMMA * self._sla_violations)

        # ------------------------------------------------------------------
        # STEP 5 — SLA check
        # SLA violated if avg latency > 200ms OR error_rate > 5%
        # ------------------------------------------------------------------
        avg_latency = self._avg_latency(self._nodes)
        error_rate  = self._error_rate(self._nodes)
        if avg_latency > 200.0 or error_rate > 0.05:
            self._sla_violations += 1

        # ------------------------------------------------------------------
        # STEP 6 — Termination check
        # ------------------------------------------------------------------
        done = (
            self._state.step_count >= MAX_STEPS
            or all(n["status"] == NodeStatus.FAILED for n in self._nodes)
        )

        # ------------------------------------------------------------------
        # STEP 7 — Build and return observation
        # ------------------------------------------------------------------
        obs = self._build_observation()
        obs.done   = done
        obs.reward = reward
        return obs

    # -----------------------------------------------------------------------
    # state property (required by OpenEnv interface)
    # -----------------------------------------------------------------------

    @property
    def state(self) -> State:
        """Return the current episode metadata (episode_id, step_count)."""
        return self._state

    # -----------------------------------------------------------------------
    # Private helpers — stub implementations
    # These will be replaced when simulator.py and stability.py are written.
    # -----------------------------------------------------------------------

    def _build_initial_nodes(self, n: int) -> list[dict]:
        """
        Create n nodes at t=0 with idle/baseline metrics.

        TODO (Phase 2): This entire method will be replaced by ClusterSimulator.
        """
        return [
            {
                "node_id": f"node-{i}",
                "status": NodeStatus.HEALTHY,
                "queue_depth": 0,
                "latency_ms": 20.0,
                "incoming_request_rate": 50.0,
                "cpu_utilization": 0.2,
            }
            for i in range(n)
        ]

    def _stub_apply_action(self, nodes: list[dict], action: SREAction) -> list[dict]:
        """
        Naive action application — placeholder for simulator.py logic.

        Currently:
          - SCALE_UP  : marks target HEALTHY (no real capacity change yet).
          - SCALE_DOWN: marks target DEGRADED.
          - REROUTE   : zeroes out request rate on target (routes traffic away).
          - SHED_LOAD : reduces queue depth by parameter fraction.
          - NO_OP     : does nothing.

        TODO (Phase 2): Replace with proper capacity / queueing model.
        """
        for node in nodes:
            if node["node_id"] != action.target_node_id:
                continue
            if action.action_type.value == "SCALE_UP":
                node["status"] = NodeStatus.HEALTHY
            elif action.action_type.value == "SCALE_DOWN":
                node["status"] = NodeStatus.DEGRADED
            elif action.action_type.value == "REROUTE_TRAFFIC":
                node["incoming_request_rate"] *= max(0.0, 1.0 - action.parameter)
            elif action.action_type.value == "SHED_LOAD":
                node["queue_depth"] = int(node["queue_depth"] * max(0.0, 1.0 - action.parameter))
        return nodes

    def _stub_tick(self, nodes: list[dict]) -> list[dict]:
        """
        Advance time by one tick — placeholder for simulator.py physics.

        Currently: queues grow by 5 per tick for HEALTHY nodes, unchecked.
        Latency scales linearly with queue depth (20ms + 2ms/request).

        TODO (Phase 2): Replace with proper M/M/c queue equations and
                        task-specific traffic injection patterns.
        """
        for node in nodes:
            if node["status"] != NodeStatus.FAILED:
                node["queue_depth"] = max(0, node["queue_depth"] + 5)
                node["latency_ms"]  = 20.0 + 2.0 * node["queue_depth"]
        return nodes

    def _compute_lyapunov(self, nodes: list[dict]) -> float:
        """
        V(s) = Σ Q_i²

        TODO (Phase 3): Move to stability.py and import from there.
        """
        return float(sum(n["queue_depth"] ** 2 for n in nodes))

    def _compute_cost(self, nodes: list[dict]) -> float:
        """
        Cost = active_nodes × cost_per_node_per_hour.

        TODO (Phase 3): Refine with spot-pricing model if time permits.
        """
        active = sum(1 for n in nodes if n["status"] != NodeStatus.FAILED)
        return active * COST_PER_NODE_PER_HOUR

    def _avg_latency(self, nodes: list[dict]) -> float:
        active = [n for n in nodes if n["status"] != NodeStatus.FAILED]
        if not active:
            return float("inf")
        return sum(n["latency_ms"] for n in active) / len(active)

    def _error_rate(self, nodes: list[dict]) -> float:
        """
        Proxy error rate: fraction of FAILED nodes in the cluster.

        TODO (Phase 2): Replace with a proper dropped-request ratio from
                        the simulator's request accounting.
        """
        if not nodes:
            return 1.0
        return sum(1 for n in nodes if n["status"] == NodeStatus.FAILED) / len(nodes)

    def _build_observation(self) -> ClusterObservation:
        """
        Assemble the ClusterObservation from current internal node state.
        """
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

        active_nodes = sum(1 for n in self._nodes if n["status"] != NodeStatus.FAILED)
        avg_latency  = self._avg_latency(self._nodes)
        error_rate   = self._error_rate(self._nodes)
        total_queue  = sum(n["queue_depth"] for n in self._nodes)
        cost         = self._compute_cost(self._nodes)
        lyapunov     = self._compute_lyapunov(self._nodes)

        return ClusterObservation(
            cluster_id=self._state.episode_id,
            task_id=self._task_id,
            active_nodes=active_nodes,
            average_latency_ms=avg_latency,
            error_rate=error_rate,
            total_queue_backlog=total_queue,
            current_cost_per_hour=cost,
            lyapunov_energy=lyapunov,
            nodes=node_obs,
            step=self._state.step_count,
            max_steps=MAX_STEPS,
            sla_violations=self._sla_violations,
            done=False,
            reward=0.0,
        )
