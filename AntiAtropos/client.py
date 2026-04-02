# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AntiAtropos Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SREAction, ClusterObservation, NodeObservation, NodeStatus


class AntiAtroposEnv(
    EnvClient[SREAction, ClusterObservation, State]
):
    """
    Client for the AntiAtropos Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with AntiAtroposEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.average_latency_ms)
        ...
        ...     action = SREAction(action_type="SCALE_UP", target_node_id="node-0", parameter=2.0)
        ...     result = client.step(action)
        ...     print(result.observation.lyapunov_energy)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AntiAtroposEnv.from_docker_image("AntiAtropos-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SREAction(action_type="NO_OP"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SREAction) -> Dict:
        """
        Convert SREAction to JSON payload for step message.

        Args:
            action: SREAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "action_type": action.action_type.value,
            "target_node_id": action.target_node_id,
            "parameter": action.parameter,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ClusterObservation]:
        """
        Parse server response into StepResult[ClusterObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with ClusterObservation
        """
        obs_data = payload.get("observation", {})

        # Parse per-node list into NodeObservation objects
        raw_nodes = obs_data.get("nodes", [])
        node_obs = [
            NodeObservation(
                node_id=n.get("node_id", ""),
                status=NodeStatus(n.get("status", NodeStatus.HEALTHY)),
                is_vip=n.get("is_vip", False),
                queue_depth=n.get("queue_depth", 0),
                latency_ms=n.get("latency_ms", 0.0),
                incoming_request_rate=n.get("incoming_request_rate", 0.0),
                cpu_utilization=n.get("cpu_utilization", 0.0),
                importance_weight=n.get("importance_weight", 1.0),
                done=n.get("done", False),
                reward=n.get("reward", 0.0),
            )
            for n in raw_nodes
        ]

        observation = ClusterObservation(
            cluster_id=obs_data.get("cluster_id", ""),
            task_id=obs_data.get("task_id", "task-1"),
            active_nodes=obs_data.get("active_nodes", 0),
            average_latency_ms=obs_data.get("average_latency_ms", 0.0),
            error_rate=obs_data.get("error_rate", 0.0),
            total_queue_backlog=obs_data.get("total_queue_backlog", 0),
            current_cost_per_hour=obs_data.get("current_cost_per_hour", 0.0),
            lyapunov_energy=obs_data.get("lyapunov_energy", 0.0),
            nodes=node_obs,
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 100),
            sla_violations=obs_data.get("sla_violations", 0),
            invalid_action_count=obs_data.get("invalid_action_count", 0),
            vip_failure_count=obs_data.get("vip_failure_count", 0),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
