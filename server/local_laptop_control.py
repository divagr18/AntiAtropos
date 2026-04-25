"""
Lightweight FastAPI control plane for local laptop Kubernetes testing.

Purpose:
- Accept simple SRE actions over HTTP
- Execute SCALE_UP / SCALE_DOWN / REROUTE_TRAFFIC / SHED_LOAD / NO_OP against local deployments
- Keep a minimal in-memory action history for debugging

Run:
    uvicorn server.local_laptop_control:app --host 0.0.0.0 --port 8010
"""

from __future__ import annotations

import subprocess
import threading
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import os

try:
    from ..control import KubernetesExecutor
except (ImportError, ModuleNotFoundError):
    from control import KubernetesExecutor  # type: ignore


class ActionRequest(BaseModel):
    action_type: str = Field(description="NO_OP | SCALE_UP | SCALE_DOWN | REROUTE_TRAFFIC | SHED_LOAD")
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
    "last_trim": None,
}

_ALLOWED_ACTIONS = {"NO_OP", "SCALE_UP", "SCALE_DOWN", "REROUTE_TRAFFIC", "SHED_LOAD"}

# Background trim interval (seconds). Default 30 minutes.
TRIM_INTERVAL_S = int(os.getenv("ANTIATROPOS_TRIM_INTERVAL_S", "1800"))


def _run_kubectl_trim() -> dict[str, Any]:
    """
    Run the pod-trim logic inline via kubectl subprocess calls.

    Scales every deployment in the namespace back to min_replicas
    and force-deletes completed / failed / evicted pods.
    Returns a summary dict.
    """
    ns = executor.namespace
    min_r = executor.min_replicas
    kubeconfig = executor.kubeconfig
    result: dict[str, Any] = {
        "namespace": ns,
        "min_replicas": min_r,
        "deployments_scaled": 0,
        "pods_deleted": 0,
        "errors": [],
    }

    def _kubectl(args: list[str]) -> str:
        env = None
        if kubeconfig and kubeconfig.lower() not in ("mock", ""):
            import os as _os
            env = {**_os.environ, "KUBECONFIG": kubeconfig}
        try:
            proc = subprocess.run(
                ["kubectl"] + args,
                capture_output=True,
                text=True,
                timeout=30,
                env=env,
            )
            return proc.stdout.strip()
        except Exception as exc:
            result["errors"].append(str(exc))
            return ""

    # Scale deployments back to min_replicas
    deploys = _kubectl(["get", "deploy", "-n", ns, "-o", "jsonpath={.items[*].metadata.name}"])
    for name in deploys.split():
        if not name:
            continue
        cur = _kubectl(["get", "deploy", name, "-n", ns, "-o", "jsonpath={.spec.replicas}"])
        try:
            cur_r = int(cur)
        except ValueError:
            continue
        if cur_r > min_r:
            _kubectl(["scale", "deploy", name, "-n", ns, "--replicas", str(min_r)])
            result["deployments_scaled"] += 1

    # Delete completed and failed pods
    for phase in ("Succeeded", "Failed"):
        pods = _kubectl([
            "get", "pods", "-n", ns,
            "--field-selector", f"status.phase={phase}",
            "-o", "jsonpath={.items[*].metadata.name}",
        ])
        for pod in pods.split():
            if not pod:
                continue
            _kubectl(["delete", "pod", pod, "-n", ns, "--force", "--grace-period=0"])
            result["pods_deleted"] += 1

    # Delete evicted pods (some k3s versions don't surface these as Failed)
    evicted = _kubectl([
        "get", "pods", "-n", ns, "-o",
        'jsonpath={range .items[?(@.status.reason=="Evicted")]}{.metadata.name}{" "}{end}',
    ])
    for pod in evicted.split():
        if not pod:
            continue
        _kubectl(["delete", "pod", pod, "-n", ns, "--force", "--grace-period=0"])
        result["pods_deleted"] += 1

    return result


def _periodic_trim() -> None:
    """Background thread: trim pods every TRIM_INTERVAL_S seconds."""
    import time as _time
    while True:
        _time.sleep(TRIM_INTERVAL_S)
        try:
            if not executor.is_mock:
                _run_kubectl_trim()
        except Exception:
            pass  # best-effort; next cycle will retry


@app.on_event("startup")
def _start_trim_thread() -> None:
    """Start the background pod-trim thread on FastAPI startup."""
    if not executor.is_mock:
        t = threading.Thread(target=_periodic_trim, daemon=True, name="pod-trim")
        t.start()


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
        "trim_interval_s": TRIM_INTERVAL_S if not executor.is_mock else None,
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
        "last_trim": STATE["last_trim"],
        "is_mock": executor.is_mock,
    }


@app.post("/trim")
def trim() -> dict[str, Any]:
    """
    On-demand pod trim: scale all deployments to min_replicas
    and delete completed / failed / evicted pods.
    """
    if executor.is_mock:
        raise HTTPException(
            status_code=400,
            detail="KubernetesExecutor is in mock mode. Set KUBECONFIG to enable trimming.",
        )
    try:
        result = _run_kubectl_trim()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Trim failed: {exc}") from exc

    STATE["last_trim"] = {
        **result,
        "timestamp_utc": _now_utc_iso(),
    }
    return STATE["last_trim"]


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
