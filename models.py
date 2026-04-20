from enum import Enum
from typing import Annotated, Literal, Optional
from pydantic import BaseModel, Field

class EnvironmentMode(str, Enum):
    SIMULATED = "simulated"
    HYBRID = "hybrid"
    LIVE = "live"

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
    is_vip: bool = False
    
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

    importance_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Business criticality weight. VIP nodes have higher impact on scoring.",
    )

    capacity: float = Field(
        default=0.0,
        ge=0.0,
        description="Current capacity units provisioned for this node (0-5).",
    )

    pending_capacity: float = Field(
        default=0.0,
        ge=0.0,
        description="Capacity units being booted (will be live after boot delay).",
    )

    queue_delta: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Normalized queue depth change from previous tick (-1 to +1).",
    )

    sla_proximity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="How close this node is to SLA violation (0=safe, 1=violating).",
    )

    node_reward: float = Field(
        default=0.0,
        description="Per-node reward contribution for credit assignment.",
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
    
    mode: EnvironmentMode = EnvironmentMode.SIMULATED
    
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

    invalid_action_count: int = Field(
        default=0,
        description="Number of forbidden actions (e.g. SHED_LOAD on critical nodes).",
    )

    vip_failure_count: int = Field(
        default=0,
        description="Number of failed VIP nodes in the current observation.",
    )

    # New fields for Prometheus/Kubernetes integration
    metric_timestamp: float = 0.0
    data_freshness_ms: int = 0
    action_ack_status: str = "success"
    action_id: str = ""
    executor_latency_ms: float = Field(default=0.0, ge=0.0)
    executor_error_code: str = ""
    raw_reward: float = 0.0
    normalized_reward: float = Field(default=0.0, ge=0.0, le=1.0)
    reward_scale_version: str = "sigmoid-v1"
    # Reward components breakdown
    reward_drift: float = Field(
        default=0.0,
        description="Lyapunov drift component of the reward.",
    )
    reward_cost: float = Field(
        default=0.0,
        description="Infrastructure cost component of the reward.",
    )
    reward_sla: float = Field(
        default=0.0,
        description="SLA penalty component of the reward.",
    )
    reward_barrier: float = Field(
        default=0.0,
        description="Barrier function penalty component of the reward.",
    )

    choke_level: float = 0.0

    nodes: list[NodeObservation]

    # Episode interaction fields (handled by framework)
    done: bool = False
    reward: float = 0.0

