# AntiAtropos 🛡️

> **Meta PyTorch OpenEnv Hackathon x SST — India AI Hackathon '26**
> Deadline: April 7, 2026

AntiAtropos is an **autonomous infrastructure management environment** for AI agents, built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework. It simulates a live microservice cluster and challenges an LLM-based SRE agent to keep the system stable, cost-efficient, and within SLA — all at the same time.

The name is a play on *"à propos"* (French: appropriate/relevant) — an SRE who acts *anti-Atropos* triggers cascades. This environment rewards the agent that acts *precisely when and how it should*.

---

## What is it?

Most OpenEnv environments are games (chess, 2048, blackjack). AntiAtropos simulates a task that **Site Reliability Engineers (SREs) and DevOps teams perform every day**: balancing a microservice cluster under variable load.

An agent observes a real-time cluster dashboard and must issue management commands each tick:

| Command | Effect |
|---|---|
| `SCALE_UP` | Add compute capacity to a node |
| `SCALE_DOWN` | Remove compute capacity |
| `REROUTE_TRAFFIC` | Redirect load away from a degraded node |
| `SHED_LOAD` | Drop non-critical requests to relieve pressure |
| `NO_OP` | Observe and wait |

The agent is scored not just on whether the cluster survives, but on **how mathematically stable** it kept the system — measured via Lyapunov energy.

---

## The Math: Lyapunov-Inspired Control

The core innovation is the reward function, which is grounded in **Lyapunov stability theory** from control systems engineering.

### Lyapunov Energy

```
V(s) = Σ Qᵢ²
```

Where `Qᵢ` is the queue depth at node `i`. This is the system's "potential energy". A rising `V(s)` means the cluster is destabilising.

### Reward Function

```
Rₜ = -(α · ΔV(s) + β · Cost + γ · SLA_Violations)
```

| Term | Meaning | Default Weight |
|---|---|---|
| `α · ΔV(s)` | Lyapunov energy *change* this tick | `α = 1.0` |
| `β · Cost` | Infrastructure cost (USD/hr) | `β = 0.05` |
| `γ · SLA_Violations` | Cumulative breaches of latency/error SLAs | `γ = 2.0` |

The agent must **minimise drift** (keep `ΔV ≤ 0`) while also minimising cost. It cannot just scale up infinitely — that would blow the cost budget.

---

## Three Tasks (Easy → Hard)

### Task 1 — Predictive Scaling *(Easy)*
- Traffic increases **linearly** over time.
- Goal: Scale nodes proactively to keep latency < 200ms without over-provisioning.
- Graded on: latency compliance + cost efficiency.

### Task 2 — Fault Tolerance *(Medium)*
- Traffic is stable, then a **random node fails** mid-episode.
- Goal: Detect the failure and reroute traffic before queues cascade.
- Graded on: time-to-recovery + how many SLA violations occurred.

### Task 3 — Stability Under Surge *(Hard)*
- A stochastic **DDoS-style burst** hits the cluster.
- Goal: Use `SHED_LOAD` on non-critical nodes to preserve the Payment Gateway's Lyapunov stability.
- Graded on: Lyapunov energy variance over the episode.

---

## Architecture

```
AntiAntropos/
├── AntiAtropos/                      ← OpenEnv package root (openenv init AntiAtropos)
│   ├── models.py                     ← Pydantic schemas: SREAction, ClusterObservation, NodeObservation
│   ├── client.py                     ← EnvClient: async/sync interface for the AI agent
│   ├── openenv.yaml                  ← Environment manifest (name, version, entrypoint)
│   ├── pyproject.toml                ← Package dependencies
│   └── server/
│       ├── AntiAtropos_environment.py ← Core orchestration: step(), reset(), reward logic
│       ├── app.py                    ← FastAPI app (created by openenv scaffold)
│       ├── Dockerfile                ← Container image for HF Spaces deployment
│       └── requirements.txt          ← Docker-level dependencies
│
├── simulator.py          [Phase 2]   ← Discrete-time M/M/1 queue physics + traffic profiles
├── stability.py          [Phase 3]   ← Lyapunov energy V(s), drift ΔV, barrier functions h(s)
├── grader.py             [Phase 3]   ← Episode-level scoring (uptime %, cost score, stability variance)
├── baseline.py           [Phase 4]   ← LLM-based baseline agent (OpenAI / Groq API)
│
├── instructions.md                   ← Full SRS (Software Requirements Specification)
└── README.md                         ← This file
```

---

## Development Phases

| Phase | What Gets Built | Status |
|---|---|---|
| **Phase 1** — Interface Definition | `models.py`, `client.py`, `AntiAtropos_environment.py` schemas + step skeleton | ✅ Done |
| **Phase 2** — Simulator | `simulator.py`: M/M/1 queue equations, 3 traffic profiles, action physics | 🔲 Next |
| **Phase 3** — Lyapunov Grader | `stability.py` + `grader.py`: energy computation, drift, episode scoring | 🔲 Pending |
| **Phase 4** — Baseline Agent | `baseline.py`: LLM agent loop, local testing via `uv run server` | 🔲 Pending |
| **Phase 5** — Deployment | `server/requirements.txt` tuning + `openenv push` to Hugging Face Spaces | 🔲 Pending |

---

## Running Locally

```bash
# Install dependencies
cd AntiAtropos
pip install -e .

# Start the OpenEnv server
uv run server --host 0.0.0.0 --port 8000

# Open the web UI (optional)
ENABLE_WEB_INTERFACE=true uv run server
# Then visit http://localhost:8000/web
```

---

## Deployment

```bash
# From the AntiAtropos/ directory
openenv push --repo-id your-hf-username/AntiAtropos
```

This packages the server into the scaffolded Dockerfile and deploys it as a Hugging Face Space, where the baseline agent (and hackathon judges) can connect to it.

---

## Key Concepts

If you're unfamiliar with the theoretical underpinnings, the concepts below are worth reading before building Phases 2 and 3:

- **Lyapunov Stability Theory** (Control Systems) — The mathematical foundation for the reward function.
- **Queueing Theory / M/M/1 Queues** — How to model request arrivals, service rates, and latency.
- **Little's Law** — Relates queue length, arrival rate, and wait time. Used to compute latency.
- **Barrier Functions / CBFs** — How to model hard safety constraints (e.g., a node *must not fail*).
- **Discrete-Time Dynamical Systems** — How to simulate continuous physics in finite time steps.

---

## Competition Context

- **Event:** Meta PyTorch OpenEnv Hackathon × Scaler School of Technology
- **Round 1 Goal:** Build a real-world utility OpenEnv environment for AI agents.
- **Submission:** Hosted Hugging Face Space + reproducible inference script.
- **Deadline:** April 7, 2026
- **Link:** https://www.scaler.com/school-of-technology/meta-pytorch-hackathon/dashboard
