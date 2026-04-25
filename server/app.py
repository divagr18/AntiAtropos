# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the AntiAtropos Environment.

This module creates an HTTP server that exposes the AntiAtroposEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

import json
import os
from dotenv import load_dotenv

load_dotenv()

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    # Change these names to match models.py
    from ..models import SREAction, ClusterObservation
    from .AntiAtropos_environment import AntiAtroposEnvironment
    from ..telemetry import render_prometheus_metrics
except (ModuleNotFoundError, ImportError):
    # And here as well
    from models import SREAction, ClusterObservation
    from server.AntiAtropos_environment import AntiAtroposEnvironment
    from telemetry import render_prometheus_metrics


# Create the app with web interface and README integration
# Create the app with the correct class names
app = create_app(
    AntiAtroposEnvironment,
    SREAction,           # Changed from AntiAtroposAction
    ClusterObservation,  # Changed from AntiAtroposObservation
    env_name="AntiAtropos",
    max_concurrent_envs=100,
)


@app.get("/metrics")
def metrics():
    from fastapi import Response

    payload = render_prometheus_metrics()
    return Response(content=payload, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/config/runtime")
def runtime_config():
    raw_map = os.getenv("ANTIATROPOS_WORKLOAD_MAP", "")
    mapped_nodes: list[str] = []
    if raw_map:
        try:
            parsed = json.loads(raw_map)
            if isinstance(parsed, dict):
                mapped_nodes = sorted(str(k) for k in parsed.keys())
        except Exception:
            mapped_nodes = []

    raw_max_replicas = os.getenv("ANTIATROPOS_MAX_REPLICAS", "")
    max_replicas_display = raw_max_replicas if raw_max_replicas.strip() else "unbounded"

    return {
        "env_mode": os.getenv("ANTIATROPOS_ENV_MODE", "simulated"),
        "reward_output_mode": os.getenv("ANTIATROPOS_REWARD_OUTPUT_MODE", "normalized"),
        "prometheus_url_configured": bool(os.getenv("PROMETHEUS_URL")),
        "kubeconfig_configured": bool(os.getenv("KUBECONFIG")),
        "k8s_namespace": os.getenv("ANTIATROPOS_K8S_NAMESPACE", "default"),
        "min_replicas": os.getenv("ANTIATROPOS_MIN_REPLICAS", "1"),
        "max_replicas": max_replicas_display,
        "scale_step": os.getenv("ANTIATROPOS_SCALE_STEP", "3"),
        "strict_real": os.getenv("ANTIATROPOS_STRICT_REAL", "false"),
        "workload_map_configured": bool(raw_map),
        "workload_map_nodes": mapped_nodes,
        "supported_modes": ["simulated", "hybrid", "live", "aws"],
    }


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m AntiAtropos.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn AntiAtropos.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
