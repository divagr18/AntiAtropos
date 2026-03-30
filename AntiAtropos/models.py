# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the AntiAtropos Environment.

AntiAtropos is an autonomous SRE environment. An agent observes a live
microservice cluster (latencies, queues, costs) and takes management actions
(scale, reroute, shed load) to maintain system stability, measured via
Lyapunov Energy.
"""

from enum import Enum

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """The four management primitives available to the SRE agent."""

    SCALE_UP = "SCALE_UP"
    """Add compute capacity to a target node or node group."""

    SCALE_DOWN = "SCALE_DOWN"
    """Remove compute capacity from a target node or node group."""

    REROUTE_TRAFFIC = "REROUTE_TRAFFIC"
    """Redirect a percentage of traffic away from a target node to healthy peers."""

    SHED_LOAD = "SHED_LOAD"
    """Drop a percentage of non-critical requests at a target node to relieve pressure."""

    NO_OP = "NO_OP"
    """Take no action. The agent explicitly chooses to observe and wait."""


class SREAction(Action):
    """
    A management command issued by the SRE agent.

    The agent must always specify an action_type. For targeted actions
    (anything other than NO_OP), target_node_id and parameter are required.

    Examples:
        Scale up node-2 by 3 instances:
            SREAction(action_type=ActionType.SCALE_UP, target_node_id="node-2", parameter=3.0)

        Reroute 40% of traffic away from a failing node-0:
            SREAction(action_type=ActionType.REROUTE_TRAFFIC, target_node_id="node-0", parameter=0.4)

        Do nothing this tick:
            SREAction(action_type=ActionType.NO_OP)
    """

    action_type: ActionType = Field(
        ...,
        description=(
            "The management command to execute. One of: SCALE_UP, SCALE_DOWN, "
            "REROUTE_TRAFFIC, SHED_LOAD, NO_OP."
        ),
    )

    target_node_id: str = Field(
        default="",
        description=(
            "The node on which to apply the action (e.g. 'node-0', 'node-2'). "
            "Required for all actions except NO_OP."
        ),
    )

    parameter: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Magnitude of the action strictly bounded to [0.0, 1.0]. Interpretation depends on action_type:\n"
            "  - SCALE_UP / SCALE_DOWN: scaled gracefully to MAX_SCALING_STEP internally.\n"
            "  - REROUTE_TRAFFIC / SHED_LOAD: fraction of load to move/drop [0.0, 1.0].\n"
            "  - NO_OP: ignored."
        ),
    )


# ---------------------------------------------------------------------------
# Observation Space  (the "Dashboard" the agent reads)
# ---------------------------------------------------------------------------

class NodeStatus(str, Enum):
    """Coarse health classification for a single node."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"   # High queue / latency but still serving
    FAILED = "FAILED"       # Node is down; all traffic must be rerouted


class NodeObservation(Observation):
    """
    Per-node metrics snapshot, exposed to the agent each step.

    This is the raw data the Lyapunov layer in stability.py consumes.
    The queue_depth field is the primary input to V(s) = Σ Q_i².
    """

    node_id: str = Field(..., description="Unique identifier for this node (e.g. 'node-0').")

    status: NodeStatus = Field(
        default=NodeStatus.HEALTHY,
        description="Coarse health classification: HEALTHY, DEGRADED, or FAILED.",
    )

    queue_depth: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized queue depth [0.0, 1.0]. Represents the % of theoretical max queue."
        ),
    )

    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Rolling average response latency for this node in milliseconds.",
    )

    incoming_request_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="Requests per second currently arriving at this node.",
    )

    cpu_utilization: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="CPU utilization as a fraction [0.0, 1.0].",
    )


class ClusterObservation(Observation):
    """
    Cluster-wide aggregated dashboard — the top-level observation returned
    by reset() and step() to the agent.

    This is the 'what a human SRE sees on their monitoring screen' abstraction.
    All per-node detail needed for Lyapunov computation is embedded in `nodes`.
    """

    # ---- Identity ----
    cluster_id: str = Field(
        ...,
        description="Unique identifier for this cluster episode (maps to episode_id).",
    )

    task_id: str = Field(
        default="task-1",
        description=(
            "Which scenario is active. One of: 'task-1' (Predictive Scaling), "
            "'task-2' (Fault Tolerance), 'task-3' (Stability Under Surge)."
        ),
    )

    # ---- Cluster-level aggregates ----
    active_nodes: int = Field(
        default=0,
        ge=0,
        description="Number of nodes currently in HEALTHY or DEGRADED status.",
    )

    average_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Mean response latency across all active nodes in milliseconds.",
    )

    error_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of requests [0.0, 1.0] returning errors cluster-wide. "
            "An SLA violation occurs when this exceeds 0.05 (5%)."
        ),
    )

    total_queue_backlog: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Normalized sum of queue_depth across all nodes."
        ),
    )

    current_cost_per_hour: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Running infrastructure cost in USD/hour. Scales linearly with active_nodes. "
            "The β·Cost term in the reward penalises over-provisioning."
        ),
    )

    lyapunov_energy: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Current Lyapunov energy V(s) = Σ Q_i². "
            "Computed by stability.py. Lower is more stable. "
            "Exposed here so agents can directly observe their stability metric."
        ),
    )

    # ---- Per-node drill-down ----
    nodes: list[NodeObservation] = Field(
        default_factory=list,
        description="Full per-node snapshot. Length equals the number of nodes in the cluster.",
    )

    # ---- Episode bookkeeping ----
    step: int = Field(
        default=0,
        ge=0,
        description="Current step number within the episode.",
    )

    max_steps: int = Field(
        default=100,
        description="Maximum steps before the episode terminates automatically.",
    )

    sla_violations: int = Field(
        default=0,
        ge=0,
        description=(
            "Cumulative count of steps where error_rate > 0.05 or "
            "average_latency_ms > 200ms. Used in the γ·SLA_Violations reward term."
        ),
    )
