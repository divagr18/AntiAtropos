# The AntiAtropos Reward Function

A physics-grounded, multi-scale control signal for Autonomous SRE — built on Lyapunov stability theory, graph-aware energy, three-tier cost economics, and smooth preventive SLA gradients.

---

## Architecture at a Glance

The reward operates at three temporal scales:

| Scale | Purpose | Consumer |
|-------|---------|----------|
| **Per-step** | Immediate feedback on stability, cost, SLA, action quality | LLM agent |
| **Per-episode** | Holistic grading across uptime, cost, stability | Leaderboard |
| **Cross-episode** | Sustained-excellence bonuses | Leaderboard |

No single term dominates — a healthy cluster with wasted capacity scores poorly, as does a cheap cluster with burning queues.

---

## Layer 1: Lyapunov Graph Energy

```
V_graph(s) = Σ w_i · Q_i²  +  edge_weight · Σ_{(i,j)∈E} |Q_i − Q_j|
```

**Node energy** — Squared queue depths weighted by business importance (node-0 VIP = 2×). Squaring penalizes load concentration: one node at Q=100 is far worse than five at Q=20.

**Edge imbalance** — Penalizes flow mismatch across DAG edges. If a parent has deep queues but its child is idle, the edge term fires even though the child's individual energy is zero. This gives the agent gradient signal to **balance load across topology**, not just minimize individual queues.

---

## Layer 2: Reward Composition

```
R_t = −(α·ΔV + β·Cost + γ·SLA + δ·Barrier)
```

### ΔV — Lyapunov Drift (α = 0.002)

One-step change in cluster energy. Negative = stabilizing. Positive = destabilizing. The small weight makes it a **directional nudge**, not a sledgehammer. Grounded in Neely's Drift-Plus-Penalty framework, which guarantees queue stability with bounded average cost.

### Cost — Three-Tier Infrastructure Model (β = 1.5)

```
needed = ⌈incoming_rate / 15⌉
```

| Tier | Range | Rate | Rationale |
|------|-------|------|-----------|
| **Baseline** | ≤ DEFAULT_CAPACITY (3) | $0.05/u/hr | Already provisioned — sunk cost, no penalty |
| **Justified** | >3, ≤ needed | $0.20/u/hr (4×) | Extra capacity serving traffic — defensible spend |
| **Idle waste** | > needed | $1.00/u/hr (20×) | Capacity sitting idle — pure waste |

A naive two-tier model (cheap ≤ needed, expensive > needed) penalizes baseline capacity as "overprovisioned" because `needed` can be 1 while DEFAULT_CAPACITY is 3. The three-tier model recognizes that **baseline infrastructure is already paid for** — only agent-added capacity triggers premium pricing.

Baseline cost: 5 nodes × 3 units × $0.05 = **$0.75/hr**. Scaling one node to capacity=6 (3 justified + 0 idle) costs $0.75 + $0.60 = $1.35/hr.

### SLA — Smooth Preventive Penalty (γ = 4.0)

```
sla = max(σ(latency, threshold=0.20, temp=0.03), σ(errors, threshold=0.05, temp=0.01))
```

Dual sigmoids with **max** operator — worst dimension dominates. Unlike binary penalties (0 or 1), the sigmoid provides **gradient before the violation**, enabling the agent to learn preventive scaling. Asymmetric temperatures reflect operational reality: latency degrades gradually (wide band), error rates spike sharply (narrow band).

### Barrier — Control-Barrier Function (δ = 0.1)

```
H(s) = Σ max(0, Q_i − 150)² / 10000
```

Zero below Q=150, quadratic above. Creates a **hard danger zone** near catastrophic failure (Q=200). Architecturally distinct from SLA: SLA says "pay attention," barrier says "act now or the node dies." Layered defense at different urgency levels.

---

## Layer 3: Sigmoid Normalization

```
reward_01 = σ(raw_reward, midpoint=−3.0, temperature=2.0)
```

Maps raw reward (always negative) to [0, 1] for the LLM. The **midpoint = −3.0** centers the sigmoid where rewards actually cluster (≈ −1 to −8). Temperature = 2.0 gives visible per-action gradient:

| Action | Reward |
|--------|--------|
| Baseline NO_OP | ~0.72 |
| 1× SCALE_UP | ~0.54 |
| 2× SCALE_UP | ~0.29 |
| 3× SCALE_UP | ~0.14 |
| 4× SCALE_UP | ~0.04 |

The LLM can read the trend and adjust — each unnecessary scale-up is visibly worse.

---

## Layer 4: Action-Efficiency Penalties

**Cooldown** — Same action on same node within 3 ticks: `reward −= cooldown × 0.1`. Action still executes (emergencies aren't blocked), but the agent learns to wait for boot delay before re-scaling.

**Wasted action** (−0.05) — Rejected/invalid actions reduce reward immediately. The LLM sees the consequence in its very next step, not at episode end.

---

## Layer 5: Episode Grader

```
composite = 0.4·Uptime + 0.2·Stability + 0.4·Cost − invalid_penalty + bonus
```

| Dimension | Formula | Notes |
|-----------|---------|-------|
| **Uptime** | Fraction of ticks with SLA met | Latency ≤ 200ms, errors ≤ 5% |
| **Cost** | `exp(−3.0 × over_ratio)` | Exponential decay from baseline; 2× spend → score 0.05 |
| **Stability** | `1 / (1 + (avg_energy/2000)²)` | Inverse Lyapunov, no early saturation |

**Task-3 coupling** — Cost score zeroed if uptime < 50%. Prevents "cheap but dead" strategies.

**Prevention bonuses** (additive, no overlap with step reward):
- +0.10 zero VIP failures all episode
- +0.05 < 3 SLA violations all episode
- +0.05 zero invalid actions all episode

---

## Why This Is Innovative

1. **Dynamical systems, not threshold monitoring** — Lyapunov drift measures *direction of travel*, not just current state. The agent learns whether its actions move the cluster toward or away from equilibrium.

2. **Topology-aware energy** — The edge imbalance term captures parent-child queue mismatch that flat per-node metrics miss entirely.

3. **Baseline-anchored cost** — The three-tier model separates "infrastructure you already pay for" from "capacity you chose to add," preventing the reward from penalizing the default cluster state.

4. **Preventive, not reactive** — Smooth SLA sigmoids give gradient *before* violation. The agent learns the pre-scale window that boot delay demands, rather than waiting for alarms.

5. **Layered safety** — SLA + barrier = two-tier defense at different thresholds and urgencies. Not all danger is equally urgent.

6. **Action quality as a first-class signal** — Wasted actions, rapid re-scaling, and invalid commands produce immediate penalties. Prevents "spam SCALE_UP and hope."

7. **Simulator-to-K8s parity** — Every parameter has a real-world counterpart. DEFAULT_CAPACITY=3 = K8s replicas. Boot delay = pod startup time. Cost tiers = cloud pricing. Trained policies transfer to live infrastructure.

8. **Theoretical guarantee** — The reward structure instantiates Neely's Drift-Plus-Penalty optimization, providing formal guarantees of queue stability with bounded average cost. The agent implements a theoretically grounded control policy, not ad-hoc heuristics.
