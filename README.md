# AntiAtropos 🛡️
> **Meta PyTorch OpenEnv Hackathon x SST — India AI Hackathon '26**
> *Keep the thread of life (and uptime) intact.*

AntiAtropos is an **autonomous infrastructure management environment** for AI agents, built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. It simulates a high-load microservice cluster and challenges an LLM-based SRE agent to maintain system stability, cost-efficiency, and SLA compliance.

---

## Why "AntiAtropos"? 🧶✂️

The name is derived from **Atropos**, one of the three Fates in Greek mythology—the one who chose the mechanism of death and ultimately **cut the thread** of life. 

In the world of SRE and DevOps, when the "thread is cut," the service goes down, the SLA is breached, and the system enters a cascade of failure. An agent in this environment acts as the **Anti-Atropos**: the protector whose sole purpose is to ensure the thread remains uncut by balancing load, scaling capacity, and managing stability.

---

## The Innovation: Lyapunov-Inspired SRE 🧮

Unlike generic gym environments, AntiAtropos is grounded in **Lyapunov Stability Theory** and **Neely's Drift-Plus-Penalty** framework. The agent is not just graded on "surviving" the traffic; it is graded on how mathematically stable it keeps the cluster's potential energy.

### Lyapunov Energy $V(s)$
We model the cluster's "unrest" as potential energy:
$$V(s) = \sum Q_i^2$$
Where $Q_i$ is the queue depth at node $i$. A rising $V(s)$ indicates the cluster is destabilizing and approaching a "cutoff" event.

### The Reward Signal
The reward function implements a drift-penalization strategy:
$$R_t = -(\alpha \cdot \Delta V(s) + \beta \cdot \text{Cost} + \gamma \cdot \text{SLA Violation (this step)})$$
The agent must **minimize drift** (keep $\Delta V \leq 0$) while optimizing for cost and strict 200ms latency SLAs.

---

## Three Tasks (Easy → Hard) 🚀

### Task 1 — Predictive Scaling *(Easy)*
- **Scenario:** Traffic increases linearly over time.
- **Goal:** Scale nodes proactively to stay ahead of the ramp without over-provisioning.
- **Key Metric:** Cost Efficiency + Uptime.

### Task 2 — Fault Tolerance *(Medium)*
- **Scenario:** A random high-traffic node suffers a permanent failure at Step 25.
- **Goal:** Detect the failure instantly and use `REROUTE_TRAFFIC` to prevent the remaining nodes from cascading.
- **Key Metric:** Time-to-Recovery + SLA Compliance.

### Task 3 — Stability Under Surge *(Hard)*
- **Scenario:** Stochastic DDoS-style bursts hit the cluster at random intervals.
- **Goal:** Use `SHED_LOAD` on non-critical endpoints while maintaining the Lyapunov stability of the VIP **Payment Gateway** (`node-0`), which carries extra business impact.
- **Key Metric:** Lyapunov Energy Variance.

---

## Action Space 🛠️

The agent issues management commands each tick:

| Command | Effect |
|---|---|
| `SCALE_UP` | Add compute capacity to a node |
| `SCALE_DOWN` | Remove compute capacity to save cost |
| `REROUTE_TRAFFIC` | Redirect load away from a degraded node |
| `SHED_LOAD` | Drop non-critical requests to relieve pressure |
| `NO_OP` | Observe and wait |

---

## Local Development 💻

### Installation
```bash
# Clone and install
cd AntiAtropos/AntiAtropos
pip install -e .
```

### Start the OpenEnv Server
```bash
# Start the FastAPI/WebSocket server
python -m uvicorn AntiAtropos.server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Run the Baseline Agent
```bash
# In a new terminal
python baseline_agent.py
```

## Kubernetes Integration (Refined)

Live mode now uses a strict capability matrix and explicit workload mapping.

- `simulated`: all benchmark actions run against simulator physics only.
- `hybrid`: all benchmark actions run in simulator while telemetry is reconciled from Prometheus.
- `live`: only actions with real Kubernetes executors enabled are allowed (`NO_OP`, `SCALE_UP`, `SCALE_DOWN`).

### Required live mapping

Set `ANTIATROPOS_WORKLOAD_MAP` so each logical node maps to a concrete deployment:

```bash
export KUBECONFIG=/path/to/kubeconfig
export ANTIATROPOS_K8S_NAMESPACE=sre-sandbox
export ANTIATROPOS_WORKLOAD_MAP='{
  "node-0": {"deployment": "payments-api", "namespace": "sre-sandbox"},
  "node-1": {"deployment": "checkout-api", "namespace": "sre-sandbox"},
  "node-2": {"deployment": "inventory-api", "namespace": "sre-sandbox"}
}'
```

If live mode is started without a real Kubernetes executor or without workload mappings, actions are rejected early with a clear safety error and simulator state is not mutated.

### Telemetry label mapping

Prometheus queries can now be vector-based (for example `by (pod)`) and are aggregated into internal `node-*` IDs through `MetricMapper`.

Set `ANTIATROPOS_LABEL_NODE_MAP` to map label values to internal nodes:

```bash
export ANTIATROPOS_LABEL_NODE_MAP='{
  "payments-pod-a": "node-0",
  "checkout-pod-a": "node-1",
  "inventory-pod-a": "node-2"
}'
```

### Action execution metadata

Each observation now includes control-plane metadata:
- `action_id`
- `executor_latency_ms`
- `executor_error_code`

## Live Dashboard (Prometheus + Grafana)

This repo includes open-source dashboard wiring for live agent visibility.

For a full end-to-end runbook (Kubernetes setup, namespace/workloads, Prometheus targets, port-forwarding, troubleshooting), see:
- `OBSERVABILITY_SETUP.md`

### 1) Start the AntiAtropos server

```bash
python -m uvicorn AntiAtropos.server.app:app --host 0.0.0.0 --port 8000 --log-level info
```

The server now exposes Prometheus metrics at:
- `http://127.0.0.1:8000/metrics`

### 2) Start Prometheus + Grafana

```bash
docker compose -f docker-compose.observability.yml up -d
```

### 2.5) Expose Kubernetes state metrics to Prometheus

Run this in a separate terminal:

```bash
kubectl -n kube-system port-forward svc/kube-state-metrics 8081:8080
```

If `kube-state-metrics` is missing, install it first (or skip Kubernetes dashboard panels).

### 3) Open dashboards

- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000` (default login: `admin` / `admin`)
- Preprovisioned dashboard: **AntiAtropos Live Control Plane**
- Preprovisioned dashboard: **Kubernetes Overview (sre-sandbox)**

### Included charts

- action throughput by type and ack status
- latest reward
- Lyapunov energy trend
- queue backlog and latency trend
- executor p95 latency
- executor error rate by code
- deployment desired vs available replicas
- running pods and pod phase mix
- pod restarts over 10 minutes

### Prometheus usage quickstart

Open Prometheus UI: `http://127.0.0.1:9090`

Try these queries:

- `sum(rate(antiatropos_actions_total[1m])) by (action_type, ack_class)`
- `antiatropos_reward`
- `histogram_quantile(0.95, sum(rate(antiatropos_executor_latency_ms_bucket[2m])) by (le, mode))`
- `kube_deployment_status_replicas_available{namespace="sre-sandbox"}`
- `sum by (phase) (kube_pod_status_phase{namespace="sre-sandbox"} == 1)`

---

## Architecture 🏗️

```
AntiAtropos/
├── AntiAtropos/                 
│   ├── models.py            ← Typed Pydantic schemas (SREAction, Observation)
│   ├── simulator.py         ← Discrete-time fluid-queue physics
│   ├── stability.py         ← Lyapunov math (Energy, Drift, Barrier)
│   ├── grader.py            ← Episode-level scoring (0.0-1.0)
│   ├── client.py            ← Refined EnvClient for AI agents
│   └── server/
│       ├── AntiAtropos_environment.py  ← Core Orchestration
│       └── app.py                      ← OpenEnv FastAPI Server
├── baseline_agent.py        ← Heuristic/LLM Agent implementation
└── .gitignore               ← Clean repo configuration
```

---

## Competition Context
- **Event:** Meta PyTorch OpenEnv Hackathon x SST
- **Submission:** Hugging Face Space + Reproducible Agent
- **Deadline:** April 7, 2026
