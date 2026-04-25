"""
Lightweight FastAPI control plane for local laptop Kubernetes testing.

Purpose:
- Accept simple SRE actions over HTTP
- Execute SCALE_UP / SCALE_DOWN / NO_OP against local deployments
- Keep a minimal in-memory action history for debugging

Run:
    uvicorn server.local_laptop_control:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from ..control import KubernetesExecutor
except (ImportError, ModuleNotFoundError):
    from control import KubernetesExecutor  # type: ignore


class ActionRequest(BaseModel):
    action_type: str = Field(description="NO_OP | SCALE_UP | SCALE_DOWN")
    target_node_id: str = Field(description="node-0 .. node-9")
    parameter: float = Field(default=0.0, ge=0.0, le=10.0)


class ActionResponse(BaseModel):
    ok: bool
    action_id: str
    action_type: str
    target_node_id: str
    parameter: float
    ack_status: str
    executor_latency_ms: float
    executor_error_code: str
    timestamp_utc: str


app = FastAPI(title="AntiAtropos Local Laptop Controller", version="1.0.0")
executor = KubernetesExecutor()

# Tiny in-memory state for quick verification
STATE: dict[str, Any] = {
    "step_count": 0,
    "last_action": None,
    "history": [],
}

_ALLOWED_ACTIONS = {"NO_OP", "SCALE_UP", "SCALE_DOWN"}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "is_mock": executor.is_mock,
        "namespace": executor.namespace,
        "kubeconfig": executor.kubeconfig,
        "mapped_targets": sorted(list(executor._node_workload_map.keys())),
        "allowed_actions": sorted(list(_ALLOWED_ACTIONS)),
    }


@app.post("/reset")
def reset() -> dict[str, Any]:
    STATE["step_count"] = 0
    STATE["last_action"] = None
    STATE["history"] = []
    return {"ok": True, "timestamp_utc": _now_utc_iso()}


@app.get("/state")
def state() -> dict[str, Any]:
    return {
        "step_count": STATE["step_count"],
        "last_action": STATE["last_action"],
        "history_size": len(STATE["history"]),
        "is_mock": executor.is_mock,
    }


@app.post("/step", response_model=ActionResponse)
def step(action: ActionRequest) -> ActionResponse:
    if executor.is_mock:
        raise HTTPException(
            status_code=400,
            detail="KubernetesExecutor is in mock mode. Set KUBECONFIG to your local kubeconfig path.",
        )

    action_type = str(action.action_type).upper()
    if action_type not in _ALLOWED_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported action_type '{action_type}'. Allowed: {sorted(_ALLOWED_ACTIONS)}",
        )

    if action_type != "NO_OP" and action.target_node_id not in executor._node_workload_map:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown target_node_id '{action.target_node_id}'. "
                f"Known targets: {sorted(list(executor._node_workload_map.keys()))}"
            ),
        )

    result = executor.execute_with_metadata(
        action_type=action_type,
        target=action.target_node_id,
        parameter=float(action.parameter),
    )

    STATE["step_count"] += 1
    record = {
        "step": STATE["step_count"],
        "action_type": action_type,
        "target_node_id": action.target_node_id,
        "parameter": float(action.parameter),
        "result": result,
        "timestamp_utc": _now_utc_iso(),
    }
    STATE["last_action"] = record
    STATE["history"].append(record)
    if len(STATE["history"]) > 200:
        STATE["history"] = STATE["history"][-200:]

    ack = str(result.get("ack_status", ""))
    ok = ack.startswith("Ack:")

    if not ok:
        raise HTTPException(
            status_code=409,
            detail={
                "ok": False,
                "action_type": action_type,
                "target_node_id": action.target_node_id,
                "parameter": float(action.parameter),
                "ack_status": ack,
                "executor_error_code": str(result.get("executor_error_code", "")),
            },
        )

    return ActionResponse(
        ok=True,
        action_id=str(result.get("action_id", "")),
        action_type=action_type,
        target_node_id=action.target_node_id,
        parameter=float(action.parameter),
        ack_status=ack,
        executor_latency_ms=float(result.get("executor_latency_ms", 0.0)),
        executor_error_code=str(result.get("executor_error_code", "")),
        timestamp_utc=_now_utc_iso(),
    )
