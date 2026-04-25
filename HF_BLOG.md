---
title: "AntiAtropos: Teaching LLMs the Physics of Site Reliability Engineering"
thumbnail: /blog/assets/antiatropos/thumbnail.png
authors:
- user: PranavKK
---

# AntiAtropos: Teaching LLMs the Physics of Site Reliability Engineering

> Infrastructure is not a static set of configurations — it is a dynamic system of energy, flow, and stability.

**TL;DR:** We built AntiAtropos, an OpenEnv-compatible training environment where LLMs learn to manage production microservice clusters by minimizing a Lyapunov energy function — the same control-theoretic framework used to stabilize power grids and robotic systems. We then trained a Qwen3.5-4B model with QLoRA + REINFORCE and watched it learn to scale, reroute, and shed load without ever seeing a runbook.

---

## The Problem: Platform Engineering Has a Training Gap

Every platform team has the same story. At 3 AM, PagerDuty fires for a traffic surge. An engineer wakes up, opens kubectl, scales a deployment, and goes back to sleep. Tomorrow, the same surge, the same runbook, the same interruption.

This is **high-stress, low-leverage work**. Static autoscaling (HPA, VPA) can't reason ahead — it reacts to thresholds after damage is done. Runbooks document known procedures but rot the moment an edge case arrives. And the institutional knowledge that makes incident response fast lives in engineers' heads, not in systems.

Training LLMs to do this work has been held back by one thing: **environments that don't capture the physics of production infrastructure**. Chat-based benchmarks measure conversation skill, not operational judgment. Grid worlds abstract away delay, cost, and cascading failure — the very things that make SRE hard.

AntiAtropos changes that.

---

## The Physics Engine

Traditional observability measures metrics — CPU, latency, error rate. We measure **stability**.

We model the cluster as a **fluid queue network** — request flow is water, nodes are reservoirs, and capacity is pipe diameter. The state of the system at any tick is captured by a single scalar: the **Lyapunov Energy Function**.

```
V(s) = Σ wᵢ · Qᵢ²
```

- **Qᵢ** is the queue depth at node i — its "potential energy"
- **wᵢ** is the node's business importance (the payment gateway carries 2× gravity)
- Squaring penalizes load concentration: one node at Q=100 is far worse than five at Q=20

But energy alone isn't enough. In a real cluster, node-0 feeds node-1 and node-2, which feeds node-3. If a parent is backed up but its child is idle, that mismatch is invisible to per-node metrics. So we add an **edge imbalance term**:

```
V_graph(s) = Σ wᵢQᵢ² + edge_weight · Σ|Qᵢ − Qⱼ|
```

This gives the agent gradient signal to **balance load across topology**, not just minimize individual queues.

Latency follows a hockey-stick curve (M/M/1 queueing theory): comfortable at 30% utilization, exponential at 90%+. Boot delay is 5 ticks — meaning the agent must act **before** the surge arrives, not after. This models the cold-start delay of Kubernetes pod scaling, where new containers take 30-90 seconds to become ready.

We call this the **"Point of No Return"** — the operational regime that real on-call engineers fear, and that grid-world environments never model.

---

## The Reward: Drift-Plus-Penalty

The reward function is grounded in Neely's **Drift-Plus-Penalty** framework from stochastic network optimization — the same theory used to prove stability guarantees for wireless networks and data center scheduling.

```
R_raw(t) = −(α·ΔV + β·Cost + γ·SLA + δ·Barrier)
```

### ΔV — Lyapunov Drift (α = 0.002)

The one-tick change in cluster energy. Negative drift means the system is stabilizing. Positive means it's destabilizing. The small weight makes this a directional nudge — the agent gets graded on where the system is *going*, not just where it is.

### Cost — Three-Tier Economics (β = 1.5)

Production infrastructure has a nonlinear cost curve. Baseline capacity (default 3 replicas) is already paid for — it's a sunk cost. Agent-added capacity costs 4× more. Idle capacity costs 20× more. This prevents the trivial strategy of "scale everything to max and call it a day."

| Tier | Range | Cost | Rationale |
|------|-------|------|-----------|
| Baseline | ≤ 3 replicas | $0.05/u/hr | Already provisioned — sunk cost |
| Justified | 3 → needed | $0.20/u/hr | Serving traffic — defensible |
| Idle Waste | > needed | $1.00/u/hr | Pure waste |

### SLA — Smooth Preventive Gradients (γ = 4.0)

Most SRE tools use binary thresholds: below 200ms is fine, above is a violation. This gives no gradient *before* the threshold — the agent has no signal to act preventively. AntiAtropos uses **dual sigmoids** on latency and error rate. The sigmoid provides measurable gradient well before the violation occurs, enabling the agent to learn the pre-scale window that boot delay demands.

### Barrier — Control-Barrier Function (δ = 0.1)

Zero below Q=150, quadratic above. This creates a hard danger zone near catastrophic failure (Q=200). Architecturally distinct from the SLA term: SLA says "pay attention," the barrier says "act now or the node collapses."

### Normalization

The raw reward (always negative, typically -1 to -8) is mapped to [0, 1] via a sigmoid with midpoint at -3.0. A healthy NO-OP scores ~0.72. One unnecessary SCALE_UP drops to ~0.54. Four wasteful SCALE_UPs tank to ~0.04. The LLM can read the trend and adapt immediately.

---

## The Action Space

Five actions, each with real operational semantics:

| Action | Meaning | Analogous To |
|--------|---------|-------------|
| SCALE_UP | Add capacity (5-tick boot delay) | `kubectl scale deployment` |
| SCALE_DOWN | Reduce capacity, lower cost | `kubectl scale --replicas=N` |
| REROUTE_TRAFFIC | Shift load away from unhealthy node | Traffic policy override |
| SHED_LOAD | Drop traffic fraction (forbidden on critical nodes) | Rate limiter / circuit breaker |
| NO_OP | Hold position | Doing nothing is a valid SRE action |

SHED_LOAD is forbidden on the payment gateway and other critical nodes — hard safety constraints that mirror production guardrails.

---

## Three Tasks, Progressive Difficulty

The curriculum is built into the environment:

- **task-1 (Capacity Ramp):** Load climbs linearly. The agent must scale predictively, not reactively. Boot delay means late scaling fails.
- **task-2 (Fault Tolerance):** A non-critical node fails at a random tick. The agent must detect the failure, reroute traffic, and compensate — the exact workflow of a real incident.
- **task-3 (Surge Stability):** A 75 req/tick surge bypasses the gateway and hits two internal services. SHED_LOAD is forbidden on critical nodes. The agent must absorb the surge with precision scaling while maintaining cost discipline. This is the hardest benchmark — and the closest to a real high-severity incident.

---

## Training: QLoRA + REINFORCE on A10G

We trained a **Qwen3.5-4B** model using QLoRA (rank-32, 7 target modules, 42M trainable parameters = 1.6% of total) with REINFORCE with baseline as the loss function.

### Architecture

```
HF Job (A10G, $0.34/hr)
┌──────────────────────────────────────┐
│  uvicorn :8000  ←──→  train.py       │
│  (simulator)          (Qwen3.5-4B)   │
└──────────┬──────────────────────────┘
           │  push every 25 iterations
           ▼
    HF Hub: checkpoints + plots + metrics
```

The simulator runs as a co-located FastAPI server inside the same HF Job — no network latency between action generation and environment feedback. The server is pure CPU; the GPU is dedicated to training.

### Optimization Tricks

1. **Parallel episode rollouts** — Instead of running 12 episodes sequentially (480 single-token forward passes per iteration), we batch all 12 episodes' observations and generate actions in a single forward pass. 40 forward passes instead of 480. ~10× speedup.

2. **Batched loss computation** — Transitions are padded and processed in groups of 32 instead of one at a time. ~30× speedup on the loss step.

3. **Observation compression** — JSON field keys abbreviated from full words to 1-2 character keys. ~40% fewer tokens per observation, directly reducing generation latency.

4. **Left-padded batch generation** — Critical for correct batch inference with causal LMs. Padding tokens go before the content, not after.

### Key Metrics

| Metric | Before | After |
|--------|--------|-------|
| Forward passes per iteration | 960 (480 gen + 480 loss) | ~55 (40 gen + ~15 loss) |
| Time per iteration (local server) | ~100s | ~18s |
| GPU utilization | 15% (3.5 GB) | ~40% (9 GB) |
| Total training cost (500 iters) | ~$4.72 | **$0.91** |

---

## Results: Improvement Across All Tasks

After 500 iterations of training (12 episodes per iteration, 40 steps each), the model showed measurable improvement across all three tasks compared to both the untrained baseline and a heuristic policy.

**Reward trajectories** (visible improvement over baseline):

- **task-1:** Baseline NO-OP: 0.70. Trained: consistently above 0.80, with the agent learning to scale down during idle periods — reducing cost without sacrificing SLA.
- **task-2:** The trained model learned to detect failure by observing queue depth collapse at the failed node and starved outflows at its children, then execute REROUTE_TRAFFIC followed by targeted SCALE_UP — the exact two-step incident response protocol an SRE would follow.
- **task-3:** The hardest task showed the most dramatic improvement. The baseline can't handle the surge at all (0.21). The trained model learned to read early queue buildup signals and pre-scale node-1 and node-2 before the surge stresses the cluster, demonstrating **predictive capacity planning**.

**Action distribution** shift:
- Before training: heavily biased toward SCALE_UP (~60% of actions) and NO_OP (~35%)
- After training: balanced distribution across all 5 action types, with SCALE_DOWN increasing from ~2% to ~20%

This is the most important behavioral change. An agent that only scales up is an agent that burns money. An agent that also scales down during quiet periods is an agent doing actual SRE.

---

## What Makes This Different

**1. Dynamical systems, not threshold monitoring.** Lyapunov drift measures *direction of travel*, not just current state. The agent learns whether its actions move the cluster toward or away from equilibrium — a fundamentally richer signal than "CPU > 80%."

**2. Topology-aware energy.** The edge imbalance term captures parent-child queue mismatch that flat per-node metrics miss. This lets the agent learn that when node-0 is overloaded but node-1 is idle, the problem isn't node-0's capacity — it's the traffic distribution.

**3. Preventive, not reactive.** Smooth SLA sigmoids provide gradient *before* violation. The agent learns the pre-scale window that boot delay demands, rather than waiting for alarms and reacting after damage is done.

**4. Baseline-anchored cost.** The three-tier cost model recognizes that baseline infrastructure is already paid for. This prevents the reward from penalizing the default cluster state — a failure mode of naive cost functions.

**5. Theoretical foundation.** The reward instantiates Neely's Drift-Plus-Penalty framework, providing formal guarantees of queue stability with bounded average cost. The agent doesn't learn ad-hoc heuristics — it learns a theoretically grounded control policy.

**6. Live Kubernetes bridge.** The same actions that control the simulator can drive real Kubernetes deployments via `kubernetes_executor.py`. Trained policies transfer to production infrastructure with minimal adaptation.

---

## Try It Yourself

The environment is live on Hugging Face Spaces:

- **Environment:** [hf.co/spaces/PranavKK/AntiAtropos](https://hf.co/spaces/PranavKK/AntiAtropos)
- **Code + training pipeline:** [hf.co/Keshav051/AntiAtropos](https://hf.co/Keshav051/AntiAtropos)
- **Trained model checkpoints:** [hf.co/Keshav051/antiatropos-qlora](https://hf.co/Keshav051/antiatropos-qlora)
- **Training metrics dataset:** [hf.co/datasets/Keshav051/antiatropos-training-metrics](https://hf.co/datasets/Keshav051/antiatropos-training-metrics)

### Quick Start

```python
# Install OpenEnv client
pip install openenv-core

# Connect to the environment
from openenv import Environment
env = Environment("https://pranavkk-antiatropos.hf.space")

obs = env.reset(task_id="task-3")
for step in range(60):
    # Your agent's action here
    obs = env.step({"action_type": "NO_OP", "target_node_id": "node-0", "parameter": 0.0})
```

### Train Your Own

```bash
# Launch training on HF Jobs
python training/launch_train.py \
    --hub-model-repo YourName/antiatropos-qlora \
    --hub-metrics-dataset YourName/antiatropos-training-metrics
```

---

## What's Next

**GNN Control.** We're exploring whether modeling the cluster as a dynamic graph lets the agent directly manipulate topology rather than individual nodes — thinking in edges instead of endpoints.

**Cross-Cluster Generalization.** A model trained on 5 nodes should transfer to 10 or 20 without retraining. We're testing whether the Lyapunov formulation enables this naturally.

**Multi-Token Attention for SRE.** Standard attention captures global averages. We're investigating frequency-selective transforms that capture p99 jitter — the "breathiness" of a cluster under stress — as a separate signal channel.

---

*Built with ❤️ for the 2026 OpenEnv Hackathon.*
