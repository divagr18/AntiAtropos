---
title: AntiAtropos Environment Server
colorFrom: gray
colorTo: red
sdk: docker
pinned: false
app_port: 7860
base_path: /
tags:
  - openenv
---

# AntiAtropos: Autonomous SRE Control Environment (OpenEnv)

> **A production-grade RL/agent environment for the future of autonomous DevOps — where intelligent agents replace fragile runbooks, reduce on-call toil, and keep infrastructure healthy without human intervention.**

AntiAtropos is an open, high-fidelity environment for training and benchmarking AI agents on site reliability engineering (SRE) — the discipline that keeps production infrastructure alive at scale. It models a live five-node microservice cluster operating under realistic production pressures: demand surges, cascading node failures, SLA deadlines, and hard safety constraints on critical services.

This is not a toy grid world or an abstract planning problem. Every action type, every penalty function, and every telemetry field in AntiAtropos was designed to mirror the exact decisions an on-call engineer faces when the PagerDuty alert fires at 3 AM.

## The Problem: Infrastructure Operations Don't Scale With Humans

Modern platform teams operate infrastructure that is orders of magnitude more complex than the teams managing it. The result is a well-documented set of pain points:

- **On-call toil.** Engineers are paged for incidents that follow predictable patterns — traffic spikes, memory pressure, node failures — and execute the same runbooks repeatedly. This is high-stress, low-leverage work that burns out senior engineers.
- **Reactive, not proactive.** Static autoscaling policies (HPA, VPA) react to thresholds but cannot reason ahead about demand trajectories, reroute traffic away from degrading nodes, or balance cost and reliability over time.
- **Runbook rot.** Documented procedures go stale. Edge cases accumulate. The institutional knowledge that makes incident response fast lives in engineers' heads, not in systems.

AntiAtropos is a training and evaluation ground for agents that solve this problem — systems that can observe cluster telemetry, reason about multi-step consequences, and issue control actions that keep services healthy, cost-efficient, and resilient.

## What AntiAtropos Trains Agents to Do

An agent operating in AntiAtropos executes the same core loop a platform engineer runs continuously:

**1. Observe the cluster state.**
The observation space mirrors real Prometheus/Grafana metrics: request rates, p99 latency, error rates, queue backlogs, CPU utilization, and per-node health — the same signals that drive every serious SRE incident workflow.

**2. Reason about what is wrong and why.**
The environment implements genuine queueing dynamics with boot delays, traffic reroute decay, and Lyapunov-based stability measurement. Agents that only react to threshold breaches perform poorly; agents that build a causal model of the cluster perform well.

**3. Issue control actions with real operational semantics.**
- `SCALE_UP` — expand node capacity (with a realistic `BOOT_DELAY_TICKS = 5` cold-start delay)
- `SCALE_DOWN` — reduce capacity and cost
- `REROUTE_TRAFFIC` — shift request load away from unhealthy nodes
- `SHED_LOAD` — drop a fraction of traffic to protect the cluster (forbidden on critical nodes)
- `NO_OP` — hold position when the system is stable

These are not abstract symbols. They map directly to `kubectl scale`, traffic policy overrides, and rate limiter controls used in production Kubernetes environments.

**4. Balance competing objectives across time.**
Uptime vs. cost vs. stability is the fundamental trade-off every platform team navigates. Brute-force overprovisioning fails the cost grader. Underprovisioning fails SLAs. The agent must plan — not just react.

**5. Respect hard safety constraints.**
Critical nodes cannot have load shed. Scale operations are bounded. Invalid actions are penalized. AntiAtropos enforces the same guardrails that production runbooks encode, rewarding agents that understand operational boundaries.

## Why This Matters for AIOps

The trajectory of platform engineering is clear: the toil layer gets automated, and engineers move up the stack. AntiAtropos provides the training and evaluation infrastructure to accelerate that transition responsibly:

- **Benchmark before you deploy.** An agent evaluated on AntiAtropos has been tested against capacity ramps, node failures, and burst surges with safety constraints — covering the incident categories that account for the majority of real production pages.
- **Dense, informative feedback.** Most production telemetry arrives in sparse, high-dimensional streams. AntiAtropos provides step-level Lyapunov-grounded reward signals that give learning algorithms meaningful gradient information at every tick — not just at episode end.
- **Composable with real infrastructure.** The Kubernetes executor (`control/kubernetes_executor.py`) and Prometheus ingestion (`telemetry/prometheus_client.py`) make it possible to wire a trained policy into a real cluster with minimal adaptation, enabling true hybrid-autonomy workflows where the agent handles routine incidents and escalates novel ones.
- **Deterministic grading.** Unlike production incidents where success is hard to measure objectively, AntiAtropos provides a clean `[0.0, 1.0]` composite score per episode — making benchmark comparisons across models and policies reproducible and auditable.

## LLM-as-SRE: Zero-Shot Incident Response Evaluation

`inference.py` provides a complete evaluation harness for testing frontier LLMs as zero-shot SRE agents. Set your API key, pick a model, and run — the script handles the full episode loop: observation formatting, action parsing, constraint enforcement, and final grading.

```bash
set OPENAI_API_KEY=your_key_here
set MODEL_NAME=gpt-4.1
set ANTIATROPOS_TASK=task-3
python inference.py
```

This makes AntiAtropos a drop-in benchmark for comparing how well different LLMs reason about infrastructure operations — a capability that is increasingly relevant as AI models are integrated into on-call tooling, runbook automation, and incident triage systems.

## OpenEnv Specification Compliance

AntiAtropos implements typed OpenEnv interfaces using Pydantic models and an OpenEnv-compatible FastAPI server:
- `Action` model: `SREAction` in `models.py`
- `Observation` model: `ClusterObservation` + `NodeObservation` in `models.py`
- `step(action)` returns observation with reward/done fields
- `reset()` returns initial cluster observation
- `state` is exposed through the OpenEnv `State` object
- `openenv.yaml` is present at repository root

OpenEnv manifest:
- name: `AntiAtropos`
- runtime: `fastapi`
- app: `server.app:app`
- port: `7860`

## Environment Dynamics

### Queueing Model

For each node `i`, AntiAtropos uses a fluid queue update:

`Q_i(t+1) = max(Q_i(t) + lambda_eff_i(t) - mu_i(t), 0)`

where:
- `lambda_eff_i = lambda_incoming_i * (1 - shed_fraction_i)`
- `mu_i = capacity_i * 15` requests/tick (or `0` if node has failed)

### Latency and CPU Utilization

- `cpu_i = lambda_incoming_i / mu_i`
- `latency_i = BASE_LATENCY_MS + LATENCY_STEEPNESS * Q_i`

### Lyapunov Stability

Core stability objective is weighted Lyapunov energy:

`V(s) = sum_i (w_i * Q_i^2)`

VIP/business-critical nodes carry higher weights `w_i`. Drift term:

`DeltaV(t) = V(s_t) - V(s_{t-1})`

### Reward Function

`R_raw_t = -(alpha * DeltaV_t + beta * Cost_t + gamma * SLA_violation_t)`

Default weights: `alpha = 0.002`, `beta = 0.01`, `gamma = 10.0`.

Normalized:

`R_norm_t = sigmoid((R_raw_t - midpoint) / temperature)`

Dense step-level signal — not sparse terminal reward — that strongly penalizes SLA failures, invalid actions, and destabilizing queue growth.

## Action Space

`SREAction` (`models.py`):
- `action_type`: `NO_OP` | `SCALE_UP` | `SCALE_DOWN` | `REROUTE_TRAFFIC` | `SHED_LOAD`
- `target_node_id`: `node-0` to `node-4`
- `parameter`: bounded float with action-dependent semantics

Safety constraints enforced by `control/validation.py`:
- `SHED_LOAD` is **forbidden on critical nodes** (`node-0`, `node-1`, `node-2`)
- Scale operations are bounded by node min/max capacity
- Invalid actions are counted and penalized in the final score

## Observation Space

`ClusterObservation`:
- Task/mode/episode step metadata
- Active node count, normalized latency, error rate, backlog
- Cost per hour, Lyapunov energy
- SLA and invalid-action counters
- Raw and normalized reward fields
- Per-node `NodeObservation` list

`NodeObservation` (per node):
- Queue depth, latency, incoming request rate
- CPU utilization, health status
- VIP flag and importance weight

## Task Suite

### `task-1` — Capacity Ramp (Easy)
Load starts near cluster capacity and ramps over the episode. The agent must proactively scale and contain queue growth without overprovisioning. A clean benchmark for predictive capacity planning — the most common form of infrastructure toil.

### `task-2` — Fault Tolerance (Medium)
A non-VIP node fails at a randomized tick. Traffic continues hitting failed capacity until the agent detects the failure and responds. Tests reactive incident response: detecting failure signals, rerouting affected traffic, and compensating with scaling — under realistic delay constraints.

### `task-3` — Stability Under Surge (Hard)
Major traffic surges target non-critical nodes, threatening to cascade. The agent must protect the VIP Payment Gateway (`node-0`). `SHED_LOAD` is forbidden on critical nodes (`node-0`, `node-1`, and `node-2`). The agent must coordinate pre-emptive `SCALE_UP` to absorb the surge before it arrives and use persistent `REROUTE_TRAFFIC` to redirect load, all while maintaining cost discipline. The closest analogue to a real high-severity incident: time pressure, safety constraints, and no single correct action.

## Grading (0.0–1.0)

Computed by `grader.py` — deterministic and reproducible:

| Component | Formula | Weight |
|---|---|---|
| Uptime | Fraction of steps with latency ≤ 0.20 and error rate ≤ 0.05 | 0.4 |
| Cost | `exp(-3.0 * over_ratio)` — punishes overprovisioning | 0.4 |
| Stability | `1 / (1 + (avg_energy / TARGET_ENERGY)^power)` | 0.2 |

- task-3: cost contribution disabled when uptime `< 0.5`
- Invalid-action penalty: `-0.05` per invalid action
- Final value clipped at `0.0`

## Observability Stack

AntiAtropos ships a full production-style observability stack:
- Prometheus scrapes environment metrics at `GET /metrics`
- Grafana `antiatropos-overview` dashboard: reward trajectory, queue heatmaps, latency timeseries, SLA violations, per-node state, action throughput, executor reliability
- NGINX reverse proxy exposes `/`, `/prometheus/`, and `/grafana/` on port `7860`
- `deploy/entrypoint.sh` boots the full stack in a single container

## Kubernetes Integration

For teams evaluating agents against real infrastructure:
- `control/kubernetes_executor.py` translates `SCALE_UP`/`SCALE_DOWN` into `kubectl` operations on mapped deployments
- Configure via `ANTIATROPOS_WORKLOAD_MAP` or `ANTIATROPOS_NODE_DEPLOYMENT_MAP`
- `telemetry/prometheus_client.py` ingests live PromQL metrics and reconciles them into simulator state via weighted blending — enabling a real-environment feedback loop with minimal code change

## Baseline Scores

Reproducible NO-OP baseline over 20 seeded runs (100 steps each):

| Task | Mean Composite | Min | Max |
|---|---:|---:|---:|
| task-1 | 0.6980 | 0.6845 | 0.7171 |
| task-2 | 0.7020 | 0.6400 | 0.7560 |
| task-3 | 0.2063 | 0.1721 | 0.2521 |

Task-3's low baseline score reflects the genuine difficulty of burst surge management under safety constraints — and the substantial headroom available for capable agents.

## Setup and Usage

### Local Python

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t antiatropos:latest .
docker run --rm -p 7860:7860 antiatropos:latest
```

### OpenEnv Validation

```bash
openenv validate
```

### Hugging Face Space Deployment

```bash
openenv push
```

## Project Structure

| Path | Description |
|---|---|
| `models.py` | Typed OpenEnv action/observation models |
| `simulator.py` | Queueing physics, task dynamics, action semantics |
| `stability.py` | Lyapunov/reward math |
| `grader.py` | Deterministic episode scoring |
| `inference.py` | OpenAI-compatible baseline runner |
| `client.py` | OpenEnv client wrapper |
| `openenv.yaml` | Environment manifest |
| `server/AntiAtropos_environment.py` | Environment runtime (`reset`, `step`, state handling) |
| `server/app.py` | FastAPI/OpenEnv app + `/metrics` |
| `control/` | Action validation and Kubernetes executor |
| `telemetry/` | Prometheus ingestion, metric mapping, exporter |
| `deploy/` | Entrypoint, NGINX, Prometheus, Grafana provisioning |
| `Dockerfile`, `server/Dockerfile` | Container build targets |

## Reproducibility

- Containerized execution (`Dockerfile`, `server/Dockerfile`)
- Pinned dependency lockfile (`uv.lock`)
- Deterministic grading equations (`grader.py`)
- Explicit reward equations in code — no black-box scoring
- Configurable environment variables for mode, telemetry endpoints, and policy runtime

For fixed-seed studies, use controlled simulator seeding in evaluation harnesses.

## Evaluation Alignment

| Criterion | AntiAtropos |
|---|---|
| Real-world utility | Genuine SRE/platform engineering control task with production-grade operational constraints |
| Task quality | 3 tasks with easy-medium-hard progression mapped to real incident categories |
| Grader quality | Deterministic, interpretable composite score in `[0, 1]` |
| Environment design | Dense Lyapunov-grounded reward, clean reset/step loop, explicit episode boundaries |
| Code quality | Typed Pydantic models, modular components, OpenEnv manifest, containerized runtime |
| Novelty | Lyapunov reward shaping + live K8s control plane + Prometheus telemetry + observability-first design |