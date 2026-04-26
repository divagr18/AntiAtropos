---
title: AntiAtropos
emoji: 🚀
colorFrom: indigo
colorTo: blue
sdk: docker
app_file: server/app.py
pinned: false
---

# AntiAtropos: The Physics of Autonomous SRE

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://github.com/openenv/openenv)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://hf.co/spaces/PranavKK/AntiAtropos)

> **"Infrastructure is not a static set of configurations; it is a dynamic system of energy, flow, and stability."**

AntiAtropos is a production-grade Autonomous SRE (Site Reliability Engineering) Control Environment. It treats a microservice cluster not as a collection of scripts, but as a **Physics Engine**. By modeling infrastructure using **Fluid Queue Dynamics** and **Lyapunov Stability Theory**, AntiAtropos provides a training ground for agents that can reason about the "Thermodynamics of the Cloud."

---

## 🚀 The Vision: Beyond Runbooks

Traditional DevOps relies on static thresholds and "If-This-Then-That" runbooks. This doesn't scale with the complexity of modern microservice DAGs. AntiAtropos moves from reactive scripts to **Dynamical System Control**. 

Agents in AntiAtropos are trained to minimize the **Lyapunov Energy** of the cluster—balancing the potential energy of backlogs to maintain equilibrium under extreme pressure.

---

## 🧪 The Physics Engine

AntiAtropos simulates a 5-node cluster with high-fidelity operational dynamics:

- **Fluid Queue Dynamics**: Requests flow like water through reservoirs (nodes) and pipes (edges). Overloaded nodes create **Upstream Backpressure**, physically throttling parent service rates.
- **Lyapunov Stability**: System health is captured by a single scalar Energy Function ($V(s) = \sum w_i Q_i^2$). Squaring queue depths penalizes load concentration, forcing agents to balance the cluster.
- **The Hockey-Stick Curve**: Implements M/M/1 queueing dynamics where latency explodes exponentially as utilization hits 100%.
- **Operational Reality**: Includes **5-tick Boot Delays** for scaling, traffic reroute decay, and hard safety constraints on VIP nodes.

---

## 🛠️ OpenEnv Specification Compliance

AntiAtropos implements typed OpenEnv interfaces using Pydantic models and an OpenEnv-compatible FastAPI server:

- **Action Model**: `SREAction` in `models.py` (Typed fields for action type, node ID, and parameter).
- **Observation Model**: `ClusterObservation` + `NodeObservation` in `models.py` (High-fidelity telemetry).
- **Standard API**: Implements `reset()`, `step(action)`, and `state` according to the OpenEnv specification.
- **Manifest**: `openenv.yaml` at the root defines the runtime (`fastapi`), app entrypoint (`server.app:app`), and port (`7860`).

### Action Space (`SREAction`)
- `NO_OP`: Hold position (essential for cost discipline).
- `SCALE_UP`: Expand node capacity (triggering a cold-start delay).
- `SCALE_DOWN`: Remove capacity (prioritizing pending/booting pods).
- `REROUTE_TRAFFIC`: Shift load from target to healthy peers.
- `SHED_LOAD`: Drop traffic fraction (Safety-guarded; forbidden on VIP nodes).

### Observation Space (`ClusterObservation`)
- **Global**: Step count, Average Latency, Error Rate, Total Backlog, Cost per Hour, Lyapunov Energy.
- **Per-Node**: Queue depth, Status (HEALTHY/DEGRADED/FAILED), CPU Util, Capacity, Inflow/Outflow rates.

---

## 🏗️ Cluster Architecture & Control Plane

AntiAtropos models a 5-node production DAG with a centralized control plane.

### Topology (The Directed Graph)
Traffic flows through a hierarchical structure, enabling realistic cascading failure simulations:
```
node-0 (VIP Ingress) ──┬──► node-1 (Checkout)
                       └──► node-2 (Catalog) ──► node-3 (Database)
node-4 (Auth Ingress) ──┘
```
- **node-0**: The VIP Payment Gateway. Business-critical; load shedding is forbidden.
- **node-4**: Independent ingress for Auth services.
- **Backpressure propagation**: If `node-3` overflows, it throttles `node-2`, which in turn throttles `node-0`.

### The Live K8s Bridge
The environment includes a `KubernetesExecutor` that allows the same agent logic to control a live cluster:
- **Binding**: Uses `ANTIATROPOS_WORKLOAD_MAP` to map simulator "nodes" to real K8s Deployments.
- **Execution**: Translates high-level actions into `patch_namespaced_deployment_scale` calls with transient retry logic.
- **Reconciliation**: Ingests live Prometheus metrics to align simulator state with real infrastructure reality.

---

## 🏆 Reward Engineering: The Differentiable SRE

Our reward function is grounded in Neely’s **Drift-Plus-Penalty** framework, providing a dense, informative signal:

1.  **Lyapunov Drift ($\Delta V$)**: Measures the one-tick change in system energy. Negative drift means the cluster is stabilizing.
2.  **Smooth Sigmoid SLA**: Dual sigmoids (Latency and Error Rate) provide gradient **before** a violation.
3.  **Three-Tier Economics**: Distinguishes between "Paid-for" Baseline capacity, "Justified" scaling, and "Idle Waste" (penalized 20x).
4.  **Control-Barrier Function**: A quadratic "Danger Zone" penalty that fires near catastrophic failure ($Q > 150$).

---

## 📊 Task Curriculum & Results

| Task | Category | Weight | Mean Score (Baseline) | Mean Score (Trained) |
|---|---|---|:---:|:---:|
| **task-1** | **Capacity Ramp** | 40% | 0.69 | **0.88** |
| **task-2** | **Fault Tolerance** | 30% | 0.70 | **0.82** |
| **task-3** | **Surge Stability** | 30% | 0.21 | **0.94** |

---

## 🏁 Quick Start

### Local Installation
```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Evaluation
```bash
# Set your API key and run the evaluation harness (inference.py)
set OPENAI_API_KEY=your_key
python inference.py --task all --mode trained
```

---

*Built with ❤️ for the 2026 AntiAtropos Hackathon.*
