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

> **"Infrastructure is not a static set of configurations; it is a dynamic system of energy, flow, and stability."**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://hf.co/spaces/Keshav051/AntiAtropos)
[![Code & Infrastructure](https://img.shields.io/badge/%F0%9F%92%BB%20Code-Source-green)](https://huggingface.co/Keshav051/AntiAtropos/tree/main)
[![Trained Models & Logs](https://img.shields.io/badge/%F0%9F%A7%A0%20Models-QLoRA-orange)](https://huggingface.co/Keshav051/antiatropos-qlora)
[![Demo Video](https://img.shields.io/badge/%F0%9F%93%B9%20Video-Demo-red)](https://youtu.be/46SX0HocpSs)

## Table of Contents
- [Demo Video](#demo-video)
- [The Vision](#the-vision-beyond-runbooks)
- [The Physics Engine](#the-physics-engine)
- [Architecture](#architecture)
- [Reward Engineering](#reward-engineering-the-differentiable-sre)
- [Task Curriculum & Results](#task-curriculum--results)
- [Training: RL with Unsloth + Hugging Face Jobs](#training-rl-with-unsloth--hugging-face-jobs)
- [Quick Start](#quick-start)

---

---

> **Hackathon Submission:** We are building for **"Theme #3: World Modelling for Professional Tasks."**  
> AntiAtropos governs clusters the way physics governs a pendulum—by minimizing Lyapunov energy. Perfect SLA at **50% lower cost**.

## Demo Video
[![AntiAtropos Demo Video](https://img.youtube.com/vi/46SX0HocpSs/0.jpg)](https://youtu.be/46SX0HocpSs)

AntiAtropos is a **Reinforcement Learning environment** where an AI agent learns to stabilize a 5-node microservice cluster by treating it as a physical system. Using **QLoRA REINFORCE** on a Qwen3.5-4B model, the agent is trained to minimize Lyapunov graph energy under a Drift-Plus-Penalty objective that balances stability against infrastructure cost. The trained policy scales predictively, reroutes around failures, and holds the line during traffic surges.

---

## The Vision: Beyond Runbooks

Traditional DevOps relies on static thresholds and "If-This-Then-That" runbooks. This doesn't scale with the complexity of modern microservice DAGs. AntiAtropos moves from reactive scripts to **Dynamical System Control**. 

Agents in AntiAtropos are trained to minimize the **Lyapunov Energy** of the cluster-balancing the potential energy of backlogs to maintain equilibrium under extreme pressure.

---

## The Physics Engine

AntiAtropos simulates a 5-node cluster with high-fidelity operational dynamics:

- **Fluid Queue Dynamics**: Requests flow like water through reservoirs (nodes) and pipes (edges). Overloaded nodes create **Upstream Backpressure**, physically throttling parent service rates.
- **Lyapunov Stability**: System health is captured by a single scalar Energy Function ($V(s) = \sum w_i Q_i^2$). Squaring queue depths penalizes load concentration, forcing agents to balance the cluster.
- **The Hockey-Stick Curve**: Implements M/M/1 queueing dynamics where latency explodes exponentially as utilization hits 100%.
- **Operational Reality**: Includes **5-tick Boot Delays** for scaling, traffic reroute decay, and hard safety constraints on VIP nodes.

---

## OpenEnv Specification Compliance

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

## Cluster Architecture & Control Plane

AntiAtropos models a 5-node production DAG with a centralized control plane.

### Topology (The Directed Graph)
Traffic flows through a hierarchical structure, enabling realistic cascading failure simulations:
```
node-0 (VIP Ingress) --+--> node-1 (Checkout)
                       +--> node-2 (Catalog) --> node-3 (Database)
node-4 (Auth Ingress) --+
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

## Reward Engineering: The Differentiable SRE

Our reward function is grounded in Neely's **Drift-Plus-Penalty** framework, providing a dense, informative signal:

1.  **Lyapunov Drift ($\Delta V$)**: Measures the one-tick change in system energy. Negative drift means the cluster is stabilizing.
2.  **Smooth Sigmoid SLA**: Dual sigmoids (Latency and Error Rate) provide gradient **before** a violation.
3.  **Three-Tier Economics**: Distinguishes between "Paid-for" Baseline capacity, "Justified" scaling, and "Idle Waste" (penalized 20x).
4.  **Control-Barrier Function**: A quadratic "Danger Zone" penalty that fires near catastrophic failure ($Q > 150$).

---

## Task Curriculum & Results

| Task | Category | Weight | Mean Score (Baseline) | Mean Score (Trained) |
|---|---|---|:---:|:---:|
| **task-1** | **Capacity Ramp** | 40% | 0.69 | **0.88** |
| **task-2** | **Fault Tolerance** | 30% | 0.70 | **0.82** |
| **task-3** | **Surge Stability** | 30% | 0.21 | **0.94** |

---

## Training: RL with Unsloth + Hugging Face Jobs

All training artifacts — model checkpoints, metrics logs, stderr/stdout, and evaluation plots — are pushed to the **[Keshav051/antiatropos-qlora](https://huggingface.co/Keshav051/antiatropos-qlora)** Hugging Face Hub repository. Each run lives under its own subdirectory (e.g., `run_0011/`).

### Reference Runs

| Run | Loss Type | Description | Link |
|-----|-----------|-------------|------|
| **run_0011** | REINFORCE + baseline | **Reference run** — fully converged policy after 500 iterations. This is the canonical trained model discussed in the blog. | [View on Hub](https://huggingface.co/Keshav051/antiatropos-qlora/tree/main/run_0011) |
| **grpo_run_001** | GRPO | Experimental GRPO run for comparison against the REINFORCE baseline. See the blog for analysis. | [View on Hub](https://huggingface.co/Keshav051/antiatropos-qlora/tree/main/grpo_run_001) |

Each run folder contains:
- `checkpoint-NNNN/` — LoRA adapter weights at every 5th iteration
- `metrics.jsonl` — per-step telemetry for every episode across all iterations
- `eval_results.jsonl` — heuristic vs trained comparison at each evaluation interval
- `plots/` — loss curves, reward curves, and action distribution plots
- `train.log` — full stderr/stdout from the training container

> **Note:** The `logs/` directory at the project root also contains local copies of key run artifacts for offline inspection.

### How Training Works

Training uses two core Hugging Face technologies:
1. **🤗 Hugging Face Jobs** — serverless GPU infrastructure. You define the container image, hardware flavor, and command; HF allocates the GPU, runs the job, and streams logs back. No SSH, no cluster management.
2. **Unsloth RL** — 4-bit QLoRA with REINFORCE/GRPO support. The base model (Qwen3.5-4B) is loaded in 4-bit via Unsloth's `FastLanguageModel`, and LoRA adapters (rank-64) are trained on top using a custom REINFORCE training loop.

Inside the job container, the AntiAtropos FastAPI simulator starts on CPU (localhost:8000) while the GPU handles model forward/backward passes. This **co-located architecture** eliminates network latency between action generation and environment feedback.

### Launching Training

The **only required argument** is `--run-id` — everything else has sensible defaults:

```bash
# Minimal launch — 15 iterations, 6 episodes/iter, 20 steps/episode
python training/launch_train.py --run-id run_007
```

This uses all defaults:
- **`--hub-model-repo`** = `Keshav051/antiatropos-qlora` (artifacts pushed here)
- **`--num-iterations`** = `15` (training iterations)
- **`--num-episodes`** = `6` (episodes per iteration; 2 per task for curriculum balance)
- **`--max-steps`** = `20` (max environment steps per episode)
- **`--eval-interval`** = `50` (evaluate vs heuristic every N iterations — rarely needed for short runs)
- **`--checkpoint-interval`** = `5` (save checkpoint every N iterations)
- **`--plot-interval`** = `10` (generate plots every N iterations)
- **`--loss-type`** = `reinforce_baseline` (REINFORCE with baseline; use `grpo` for GRPO)
- **`--flavor`** = `a10g-large` (NVIDIA A10G, 24 GiB, ~$0.34/hr)
- **`--timeout`** = `4h` (job timeout)

To override any default, just pass the flag:

```bash
# Full training (500 iterations, A10G, ~$7):
python training/launch_train.py --run-id run_012 --num-iterations 500 --num-episodes 6

# GRPO experiment:
python training/launch_train.py --run-id grpo_run_002 --loss-type grpo

# Longer timeout for deep training:
python training/launch_train.py --run-id run_013 --num-iterations 500 --timeout 12h
```

### Prerequisites

1. `pip install "huggingface_hub>=0.25.0"`
2. `huggingface-cli login` (or set `HF_TOKEN` environment variable)
3. A Hugging Face Pro or Team account (required for GPU Jobs)
4. The target Hub model repo is auto-created if it doesn't exist

For the full list of options:
```bash
python training/launch_train.py --help
```

---

## Quick Start

### Local Installation
```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Evaluation & Observation

The `inference.py` script is the primary tool for validating model performance. It provides a detailed breakdown of episodic reward, SLA compliance, and cluster stability. It is an excellent way to **baseline behavior** of a new model or compare different training iterations.

To configure the environment, use the `.env` file. Key "knobs" include:
- `ENV_URL`: The URL of the AntiAtropos simulation server (e.g., your HF Space).
- `MODEL_NAME`: The identifier for the model to test (supports Groq, Local, or HF).
- `GROQ_API_KEY`: Required if using Groq-based inference for rapid prototyping.
- `ANTIATROPOS_ENV_MODE`: Set to `simulated` for training or `live` for K8s control.

```bash
# Set your API key and run the evaluation harness
python inference.py --task all --mode trained
```

---

---

## Future Horizons: The Path to Autonomous Cloud Safety

AntiAtropos is the foundation for a new class of **Differentiable SRE**. Our roadmap includes:
- **Multi-Agent Coordination**: Training specialized agents (e.g., an "Ingress Governor" and a "Storage Optimizer") to collaborate via shared Lyapunov energy.
- **Formal Verification**: Using the Lyapunov certificates generated during training to provide mathematical guarantees of stability before an agent is deployed to production.
- **Predictive Traffic Shaping**: Moving from reactive scaling to predictive world-modeling of seasonal traffic surges.

---

*Built with passion for the 2026 AntiAtropos Hackathon.*
