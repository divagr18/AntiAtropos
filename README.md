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

## 🏆 Reward Engineering: The Differentiable SRE

Our reward function is grounded in Neely’s **Drift-Plus-Penalty** framework, providing a dense, informative signal for learning.

1.  **Lyapunov Drift ($\Delta V$)**: Measures the direction of travel. Negative drift means the cluster is stabilizing.
2.  **Smooth Sigmoid SLA**: Dual sigmoids (Latency and Error Rate) provide gradient **before** a violation. The agent learns the pre-scale window demanded by pod cold-starts.
3.  **Three-Tier Economics**: Distinguishes between "Paid-for" Baseline capacity, "Justified" scaling, and "Idle Waste" (penalized 20x).
4.  **Control-Barrier Function**: A quadratic "Danger Zone" penalty that fires near catastrophic failure ($Q > 150$).

---

## 📊 Task Curriculum

AntiAtropos features a three-stage curriculum designed to graduate agents from reactive to predictive control:

| Task | Category | Objective | Weight |
|---|---|---|---|
| **task-1** | **Capacity Ramp** | Scale up proactively to beat the 5-tick boot delay as traffic ramps linearly. | 40% |
| **task-2** | **Fault Tolerance** | Detect permanent node failure, reroute traffic, and scale up starved children. | 30% |
| **task-3** | **Surge Stability** | Manage a side-channel burst surge while respecting `SHED_LOAD` bans on critical nodes. | 30% |

---

## 🛠️ Production-Ready Integration

AntiAtropos isn't just a simulator—it's a bridge to real infrastructure.

- **Live K8s Bridge**: The `KubernetesExecutor` translates simulator actions (`SCALE_UP`, `REROUTE_TRAFFIC`) into real-world cluster mutations via the Kubernetes API.
- **Prometheus Telemetry**: The `TelemetryClient` ingests live metrics and reconciles them into the simulator physics, enabling **Hybrid Autonomy**.
- **Observability Stack**: Ships with a pre-provisioned Prometheus and Grafana stack, providing a "Command Center" view of agent actions and reward trajectories.

---

## 📈 Results

After training on **Qwen3.5-4B** using **Unsloth** and **HF TRL**, our agent demonstrated a massive leap in operational judgment:

| Scenario | Base LLM | AntiAtropos Agent | Delta |
| :--- | :---: | :---: | :---: |
| **task-1 (Ramp)** | 0.69 | **0.88** | +27% |
| **task-3 (Surge)** | 0.21 | **0.94** | **+347%** |

---

## 🏁 Quick Start

### Local Installation
```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker (Includes Prometheus/Grafana)
```bash
docker build -t antiatropos .
docker run -p 7860:7860 antiatropos
```

### OpenEnv Evaluation
```bash
# Set your API key and run the evaluation harness
set OPENAI_API_KEY=your_key
python inference.py --task all --mode trained
```

---

## 📂 Project Structure

- `simulator.py`: Fluid Queue physics, backpressure, and task dynamics.
- `stability.py`: Lyapunov math, Drift-Plus-Penalty, and Reward normalisation.
- `control/`: Kubernetes executor and action validation.
- `telemetry/`: Prometheus ingestion and metric mapping.
- `inference.py`: OpenAI-compatible evaluation harness with Episode Replay Buffer.
- `grader.py`: Deterministic episode scoring and composite grading.

---

*Built with ❤️ for the 2026 AntiAtropos Hackathon.*
