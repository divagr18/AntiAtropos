from enum import Enum
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# SRE Action Schema (Control Plane)
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    NO_OP = "NO_OP"
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN" 
    REROUTE_TRAFFIC = "REROUTE_TRAFFIC"
    SHED_LOAD = "SHED_LOAD"

class SREAction(BaseModel):
    """
    Management action issued by the SRE agent.
    
    * SCALE_UP: Increment capacity on target_node_id by parameter (1-5 units).
    * SCALE_DOWN: Decrement capacity on target_node_id by parameter (1-5 units).
    * REROUTE_TRAFFIC: Shift 'parameter' [0, 1] of incoming traffic AWAY from
      target_node_id and redistribute across healthy peers.
    * SHED_LOAD: Drop 'parameter' [0, 1] of incoming traffic targeting target_node_id for 1 tick.
    """
    action_type: ActionType
    target_node_id: str
    parameter: float = Field(default=0.0, ge=0.0, le=10.0)

# ---------------------------------------------------------------------------
# Observation Schema (Data Plane)
# ---------------------------------------------------------------------------

class NodeStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"

class NodeObservation(BaseModel):
    """Telemetry for a single service instance (node)."""
    node_id: str
    status: NodeStatus
    
    # All numerical telemetry is normalized to [0, 1] for RL stability.
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
        le=1.0,
        description="Normalized processing latency [0.0, 1.0] relative to 1000ms SLA limit.",
    )

    incoming_request_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalized incoming request rate [0.0, 1.0] for this node (requests per tick).",
    )

    cpu_utilization: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Estimated CPU load [0.0, 1.0].",
    )

    # Episode interaction fields (handled by framework)
    done: bool = False
    reward: float = 0.0

class ClusterObservation(BaseModel):
    """System-wide telemetry representing the 'dashboard' for the agent."""
    cluster_id: str
    task_id: str
    step: int
    max_steps: int
    
    active_nodes: int = Field(ge=0, le=5)
    
    average_latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cluster-wide average latency (normalized [0.0, 1.0]).",
    )

    error_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cluster-wide fraction of dropped/failed requests [0.0, 1.0].",
    )

    total_queue_backlog: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Normalized sum of queue_depth across all nodes [0.0, 1.0]."
        ),
    )

    current_cost_per_hour: float = Field(
        default=0.0,
        ge=0.0,
        description="Infrastructure cost in USD/hr based on provisioned capacity.",
    )

    lyapunov_energy: float = Field(
        default=0.0,
        description="Stability metric (Sum of squares of queue depths). Low is good.",
    )

    sla_violations: int = Field(
        default=0,
        description="Cumulative count of SLA violations this episode.",
    )

    nodes: list[NodeObservation]

    # Episode interaction fields (handled by framework)
    done: bool = False
    reward: float = 0.0
