---
title: "AntiAtropos: The Physics of Autonomous Infrastructure Control"
thumbnail: /blog/assets/antiatropos/thumbnail.png
authors:
- user: PranavKK
---

# AntiAtropos: The Physics of Autonomous Infrastructure Control

> **Hackathon Submission:** We are building for **"Theme #3: World Modelling for Professional Tasks."**
> AntiAtropos represents a leap in World Modeling for the SRE domain. By treating a microservice cluster as a physical environment with its own laws of motion—modeled via Lyapunov stability and fluid dynamics—we enable AI agents to "reason" through the physics of infrastructure, maintaining equilibrium where traditional heuristics fail.

### **Project Links**
*   **Demo Video:** [youtu.be/46SX0HocpSs](https://youtu.be/46SX0HocpSs)
*   **Live Space:** [keshav051-antiatropos.hf.space](https://keshav051-antiatropos.hf.space/)
*   **Models & Training Logs:** [huggingface.co/Keshav051/antiatropos-qlora](https://huggingface.co/Keshav051/antiatropos-qlora)
*   **Source Code:** [huggingface.co/Keshav051/AntiAtropos](https://huggingface.co/Keshav051/AntiAtropos)

## Table of Contents
- [The Problem](#the-problem-infrastructure-automation-cannot-solve)
- [Two Opposing Equations](#two-opposing-equations-one-equilibrium)
- [The Evolution: SFT to QLoRA](#the-evolution-from-sft-to-qlora)
- [Architecture: HF + VM](#the-dual-plane-architecture-hf--cloud-vm)
- [Reward Shaping](#how-the-reward-is-shaped)
- [The Road Not Taken: GRPO](#the-road-not-taken-grpo-experimentation)
- [Quick Start](#quick-start)

---

## The Problem Infrastructure Automation Cannot Solve

When a production service saturates at 3 AM, the sequence is predictable. Somewhere, an engineer wakes up, opens a terminal, and types `kubectl scale deployment --replicas=10`. Traffic gets rerouted. The alert clears. The engineer goes back to sleep.

In the best case, this workflow is partially automated. Horizontal Pod Autoscalers watch CPU and bump replica counts when a threshold crosses 80%. But thresholds are reactive and they fire after congestion has already built. They have no concept of boot delay, no awareness of topology, no ability to anticipate a surge before it arrives. And when an edge case appears — a side-channel burst, a cascading dependency failure — the automation falls silent and the pager goes off.

More importantly, **no existing system provides a mathematical guarantee**. There is no production autoscaler today that can prove it will keep queues bounded. There is no incident response tool that can prove its actions are cost-optimal. The industry runs on heuristics — rules of thumb encoded as YAML, battle-tested through painful outages, but ultimately ad-hoc.

AntiAtropos is the first infrastructure control environment to replace heuristics with **provable stability guarantees**. It models the cluster as a fluid queue network, defines equilibrium through a Lyapunov energy function, and trains agents to minimize a Drift-Plus-Penalty objective.

### The Bottom Line: Performance without the Premium
The unique selling point of AntiAtropos is **Efficiency**. In head-to-head benchmarks, our agent achieved perfect SLA compliance while being **50% cheaper** than traditional heuristic-based autoscalers. By understanding the underlying physics of the cluster, the agent eliminates the "panic-scaling" that plagues modern cloud infrastructure.

![Cost Comparison: AntiAtropos vs Heuristic Autoscaler](/images/Cost%20comparison.png)
*Figure 1: AntiAtropos (Blue) maintains stability at half the resource cost of a production-grade heuristic scaler (Orange).*

---

## Two Opposing Equations, One Equilibrium

Every natural system is governed by opposing forces. A pendulum swings because gravity and tension pull in different directions.

AntiAtropos is built on the same principle. Two equations — one pushing toward stability, the other pulling toward efficiency — work in tandem to define the agent's objective. Neither dominates. Neither is allowed to win alone. The system reaches equilibrium only when both are satisfied.

### The Arena: A Directed Acyclic Graph

Before the equations, the structure they operate on. AntiAtropos models a 5-node microservice cluster as a **directed acyclic graph (DAG)** — the standard topology of real service dependencies. Nodes are services. Edges are request flows from upstream to downstream.

The cluster has five services with a fixed dependency chain:

- **node-0** (Payment Gateway — VIP, 2× importance): the primary ingress. Splits its outflow evenly between node-1 and node-2.
- **node-1** (Checkout Service): a leaf node receiving half of node-0's traffic.
- **node-2** (Catalog Service): receives the other half of node-0's traffic and passes everything downstream to node-3.
- **node-3** (Inventory Service): depends entirely on node-2.
- **node-4** (Auth Service): an independent ingress with no downstream dependencies.

External traffic enters at two points — node-0 and node-4 — splitting the cluster-wide load equally. Every tick, the simulator traverses the graph in topological order (Kahn's BFS), computing each node's actual outflow as the minimum of incoming traffic and available service rate, then routing that outflow to downstream children according to fixed edge weights. A failed node has service rate zero, so its outflow stops and its children starve — a causal failure chain the agent must learn to route around.

![AntiAtropos DAG topology showing the 5-node cluster with external traffic ingress, edge splits, and dependency chain](/images/DAG.svg)

This topology is the substrate for every feature that follows. The edge imbalance term, backpressure propagation, cascading failure detection, and upstream pressure signals all derive their meaning from this specific graph structure.

### The First Equation: Lyapunov Graph Energy

This is the **stabilizing force**. It measures how much disorder has accumulated in the cluster and how unevenly it is distributed across the topology.

$$ V_{\text{graph}}(s) = \sum w_i \cdot Q_i^2  +  \lambda_{\text{edge}} \cdot \sum |Q_i - Q_j| $$

**Node energy ($\sum w_i \cdot Q_i^2$).** Each node's queue depth is squared, then weighted by its business criticality — the payment gateway (node-0) carries $2\times$ gravity. Squaring penalizes concentration: one node at queue depth $100$ contributes the same energy as five nodes at depth $20$. The system cares about how load clusters, not just how much exists.

**Edge imbalance ($\sum |Q_i - Q_j|$).** This is the term that makes the energy function graph-aware. If node-0 is drowning in requests but node-1 — its direct child — is completely idle, per-node metrics see nothing wrong with node-1. The edge term sees the mismatch and fires. It tells the agent: the problem is not node-0's capacity; the problem is the distribution of traffic across the topology.

Together, the node and edge terms form a single scalar that captures the **entire structural health of the cluster** — both how much load exists and how poorly or well it is spread across the service graph.

### The Second Equation: Drift-Plus-Penalty (Neely Framework)

This is the **efficiency force**. It pulls against the first equation by demanding that stability be achieved at minimum cost.

$$ R(t) = -(\alpha \cdot \Delta V + \beta \cdot \text{Cost} + \gamma \cdot \text{SLA} + \delta \cdot \text{Barrier}) $$

The four terms represent four separate, sometimes competing, demands on the agent:

- **$\Delta V$ — Lyapunov Drift ($\alpha = 0.002$).** The one-tick change in cluster energy. Negative drift means the system moved toward equilibrium; positive drift means it destabilized. The coefficient is deliberately small — it is a directional nudge, grading the agent on *where the system is going*, not just where it sits.

- **Cost — Three-Tier Infrastructure Model ($\beta = 1.5$).** Not all capacity costs the same. Baseline capacity at $3$ replicas is already provisioned — a sunk cost charged at the cheap base rate. Capacity the agent adds above baseline to serve real traffic costs $4\times$ more — defensible. Capacity sitting idle above what traffic actually needs costs **$20\times$ more** — pure operational waste. This prevents the trivial policy of "scale everything to max."

- **SLA — Dual Sigmoid Gradients ($\gamma = 4.0$).** Traditional SRE tools use binary thresholds: latency below $200$ ms is fine, above $200$ ms is a violation. This gives zero gradient *before* the threshold — the agent has no signal to act preventively. AntiAtropos replaces the binary cliff with two smooth sigmoids, one for latency and one for error rate. The curve provides measurable feedback well before a violation occurs, giving the agent time to scale ahead of the surge.

- **Barrier — Control-Barrier Function ($\delta = 0.1$).** Architecturally distinct from the SLA term. The sigmoid says "pay attention — you are approaching a problem." The barrier fires quadratically when any node exceeds queue depth $150$, creating a hard danger zone that says "act now or this node collapses."

### How They Balance

Imagine an agent that only cares about stability. It scales every node to maximum capacity at the first sign of traffic. Queues stay empty. Energy stays low. But the cost term destroys the reward — idle capacity at $20\times$ rates. The agent learns to scale down.

Now imagine an agent that only cares about cost. It never scales up. Queues accumulate. The drift term goes sharply positive. Nodes approach the barrier ceiling. Then the SLA sigmoid ramps. The agent learns to scale up — but only when the gradient signals it is necessary and only to the degree the traffic demands.

The opposing equations create a **domain of viable operation** — a region in the space of (stability, cost, latency) where neither force dominates. The agent is trained to find and stay inside that region.

This is the Drift-Plus-Penalty framework, formalized by Neely et al. [1] for stochastic network optimization. The theory provides a mathematical guarantee: minimizing this expression at every step produces a policy that keeps queues bounded with minimum average cost. The agent is not learning ad-hoc heuristics. It is learning a control policy with provable stability properties.

### The Evolution: From SFT to QLoRA
The path to autonomous SRE was not linear. We initially experimented with **Supervised Fine-Tuning (SFT)**, using the scripts found in `Experimental/ProofOfConcept.ipynb` to teach the model basic "if-then" operational logic from a dataset of expert rollouts. 

However, we quickly hit the ceiling of supervised learning: an agent can only be as good as its teacher. To achieve true **autonomous stability**, we moved to a Reinforcement Learning paradigm using **QLoRA Reinforce**. This allowed us to leverage high-rank (64) adapters for deep reasoning while keeping the model memory-efficient enough to run 60-step rollouts on a single GPU. The transition from mimicking experts to **minimizing the Lyapunov Energy Function** is what allowed the agent to discover novel stabilization strategies that a human operator would miss.

![Reinforcement Learning Training Metrics: Loss and Reward Convergence](/images/RL_training_metrics.png)
*Figure 2: QLoRA Training Progress — showing the steady rise in episodic reward as the agent learns to govern the cluster's Lyapunov energy.*

![AntiAtropos Agent Strategy Evolution: Action Distribution over Iterations](/images/Agent%20Actions.png)
*Figure 3: Strategy Evolution — The model learns to prioritize dynamic Rerouting and proactive Scaling, moving away from passive observation (NO_OP) as training converges.*

---

## The Dual-Plane Architecture: HF + Cloud VM
Our observability stack is split into a **Dual-Control Plane** to balance high-fidelity simulation with real-world infrastructure monitoring.

1.  **The Simulation & Grading Plane (Hugging Face):**
    The heavy lifting of QLoRA Reinforce training and the primary OpenEnv simulator runs on Hugging Face. This plane exposes a **FastAPI server** that handles the complex reward calculations, Lyapunov energy tracking, and advantage normalization.
    
    ![HF Grafana Dashboard: Reward Latency and Lyapunov Curves](/images/HF%20Grafana.png)
    *Figure 4: The HF-side telemetry focuses on the "Brain"—tracking how the agent's reward correlates with the draining of systemic potential energy ($V$).*

2.  **The Infrastructure Plane (Cloud VM):**
    For real-world validation, we maintain a lighter control plane on a dedicated Cloud VM running a **Live K8s Cluster**. This environment is instrumented with **Prometheus and Grafana** to capture raw node observability: CPU saturation, pod restart counts, and network jitter.
    
    ![VM Grafana Dashboard: K8s Node Observability and Prometheus Metrics](/images/VM_Grafana.png)
    *Figure 5: The VM-side telemetry tracks the "Body"—monitoring the physical health of the containers being governed by the agent.*

This hybrid approach ensures that the agent is not just "playing a game" in a simulator, but is being trained on a world model that has 1:1 parity with production-grade monitoring stacks.

---

## How the Reward Is Shaped

The reward is not a single monolithic number. It is computed through a multi-stage pipeline where each stage transforms raw cluster physics into progressively more informative learning signals.

### Stage 1: Compute Lyapunov Graph Energy

At every tick, the simulator computes $V_{\text{graph}}(s)$ from ground-truth node states — the weighted squared queues plus the edge imbalance penalty. This scalar captures the total structural disorder in the cluster. A rising $V_{\text{graph}}$ means the system is destabilizing; a falling $V_{\text{graph}}$ means the agent's actions are working.

### Stage 2: Compute $\Delta V$ — Lyapunov Drift

The one-step change in energy:

$$ \Delta V(t) = V_{\text{graph}}(s_t) - V_{\text{graph}}(s_{t-1}) $$

Negative drift is good — energy decreased, stability improved. Positive drift is a warning: the system is sliding away from equilibrium. The critical insight: the agent is graded on **direction of travel**, not absolute position. Two clusters can have identical queue depths, but the one with negative drift is on a healing trajectory.

### Stage 3: Compute Cost — Three-Tier Infrastructure Economics

The cost function separates capacity into three buckets:

$$ \text{needed} = \lceil \text{incoming\_request\_rate} / 15 \rceil $$
$$ \text{cost} = \text{baseline\_at\_cheap\_rate} + (\text{needed} - \text{baseline}) \times 4\times + (\text{active} - \text{needed}) \times 20\times $$

Capacity at or below $DEFAULT\_CAPACITY$ ($3$ replicas) is charged at the base rate — it is already provisioned, a sunk cost. Agent-added capacity that serves real traffic costs $4\times$ the base rate — justified. Active capacity sitting idle above what traffic needs costs **$20\times$** the base rate — pure operational waste. Pending capacity (still booting) is always charged at the justified tier — the agent cannot control boot delay and should not be penalized for it.

### Stage 4: Compute SLA — Smooth Dual Sigmoids

Latency and error rates are passed through two independent sigmoid functions, then the maximum is taken:

$$ \text{sla\_penalty} = \max\left( \frac{1}{1 + e^{-(\text{lat\_norm} - 0.20) / 0.03}}, \frac{1}{1 + e^{-(\text{err\_rate} - 0.05) / 0.01}} \right) $$

Each sigmoid produces a smooth value in $[0, 1]$. When latency is comfortably below $200$ ms ($lat_{\text{norm}} < 0.20$), the penalty is near zero. As latency approaches the threshold, the penalty rises continuously. The temperature parameters ($0.03$ for latency, $0.01$ for errors) control how sharp the transition is — smaller values create steeper curves that give stronger gradient near the boundary.

This is fundamentally different from binary SLA counters. A counter increments by $1$ when latency crosses $200$ ms — the gradient is zero until the moment of violation. The smooth sigmoid provides **preventive gradient** — the agent senses the metric approaching the boundary and adjusts before the violation occurs.

### Stage 5: Compute the Barrier Penalty

The Control-Barrier Function fires separately from SLA:

$$ \text{barrier} = \sum \max(0, Q_i - 150)^2 $$

This is zero for all queues below $150$. Above $150$, it grows quadratically. The architectural separation between SLA and barrier is intentional:

- **SLA sigmoid** says: "pay attention — you are approaching suboptimal performance."
- **Barrier function** says: "act now — this node is about to collapse."

The barrier fires closer to catastrophic failure ($Q = 200$), creating a distinct urgency gradient that the agent must learn to distinguish from normal SLA pressure.

### Stage 6: Combine into Raw Reward

All four terms are combined into a single scalar:

$$ \text{raw\_reward} = -(0.002 \times \Delta V + 1.5 \times \text{Cost} + 4.0 \times \text{SLA} + 0.1 \times \text{Barrier}_{\text{normalized}}) $$

At baseline — a stable 5-node cluster with no SLA violations and no drift — the raw reward sits around **−1 to −3**, dominated by the cost term ($\beta \times \$0.75/hr$ baseline $\approx -1.1$). It deepens toward **−6 to −12** as SLA pressure builds, queues accumulate, or the agent wastes capacity. The weight tuning is deliberate:

- **$\Delta V$ at $0.002$** — a directional nudge, not a sledgehammer. Stability drift should inform, not dominate.
- **Cost at $1.5$** — makes over-provisioning painful enough that the agent cannot ignore it.
- **SLA at $4.0$** — the strongest term. SLA violations matter more than either drift or cost individually.
- **Barrier at $0.1$** — after normalization, comparable in magnitude to drift.

### Stage 7: Sigmoid Normalization to [0, 1]

The raw reward is not directly usable for the LLM — negative numbers are harder to interpret and compare across episodes. We map it through a sigmoid:

$$ \text{normalized\_reward} = \frac{1}{1 + e^{-(\text{raw\_reward} - \text{midpoint}) / \text{temperature}}} $$

With midpoint = $−6.0$ and temperature = $2.0$, a healthy NO-OP ($raw \approx -1$ to $-3$) maps to a normalized reward near $0.8$, a single wasteful SCALE_UP pulls it toward $0.6$, and a cluster approaching the barrier threshold ($raw \approx -12$) collapses near $0.0$. The LLM sees these values alongside the observation text and learns to associate specific action patterns with specific reward ranges — each unnecessary scale-up is visibly worse than the last.

### Stage 8: Post-Processing Penalties

Two final adjustments are applied to the normalized reward:

- **Cooldown penalty** — scaling the same node within 3 ticks subtracts up to 0.1 from the reward. The action still executes (emergencies must not be blocked), but the agent receives negative reinforcement for oscillatory behavior.
- **Invalid-action penalty** — attempting SHED_LOAD on a critical node subtracts 0.05. The validator hard-rejects the action first; this penalty ensures the agent sees the rejection in its reward history.

Most RL environments compute reward in a single step: state → reward. AntiAtropos decomposes it into eight stages because each stage teaches a distinct concept:

This decomposition is not arbitrary. It mirrors the way a human SRE reasons about an incident: "Is the situation getting worse? What am I spending? How close am I to violating SLA? Is any service about to die?" AntiAtropos encodes each of these questions into a separate, tunable, inspectable stage of the reward pipeline.

---

## The Realism of Interactions

Physics gives the agent its objective. But to learn that objective, the agent must interact with an environment that behaves like a real production cluster. AntiAtropos models the following features — each drawn from operational reality, each contributing gradient signal that shapes learned behavior.

### Delay and Finite Deployment Time

Scaling in Kubernetes is not instantaneous. New pods take $30$-$90$ seconds to pull images, start containers, and register with the load balancer. AntiAtropos models this as a **$5$-tick boot delay**. A SCALE_UP command adds capacity to a pending queue; that capacity only becomes active five ticks later.

This creates a fundamental requirement for **predictive action**. The agent cannot wait until queues are overflowing to scale — by the time the new capacity boots, the node has already failed. It must read the early signal in its observations and act before the surge arrives.

### Jitter and Non-Determinism

Production traffic is never perfectly predictable. AntiAtropos introduces **jitter** into every critical event. The Task 3 surge has a nominal window of $10$ ticks, but each episode randomizes the start and end points by $\pm 10$ ticks. The domain parameters — initial load, ramp slope, failure timing — are randomized per episode using a configurable seed. The agent cannot memorize timing; it must learn to detect the *signal* of an impending event from the cluster's state.

### Sensor Noise and Partial Observability

Real observability pipelines are imperfect. Metrics scrape intervals, network delays, and transient sensor failures all contribute noise. AntiAtropos introduces a **$5\%$ sensor dropout probability**: on any given tick, any node's reported queue depth may read as zero and its latency may report $-1.0$ — simulating a failed scrape or lost sample. The agent must learn to operate under partial information, cross-referencing adjacent nodes and recent history to infer the true state.

### Backpressure Propagation

A node is not an island. When a downstream service is congested, it cannot accept forwarded requests fast enough — and that congestion travels backward through the Directed acyclic graph (DAG). AntiAtropos models **backpressure**: when any child node's queue exceeds the backpressure threshold ($Q > 60$), its parent's effective service rate is reduced proportionally, up to a cap of $40\%$.

This means the health of any node affects every node upstream of it. A problem at node-3 eventually starves node-2, which radiates back to node-0. The agent must see the graph as a connected system, not a collection of independent services.

### Cascading Failure Propagation

In a real incident, one node's failure can cascade — the remaining nodes absorb its traffic, become overwhelmed, and fail in sequence. AntiAtropos implements **bounded cascade detection**: when a node fails from overload ($queue > 200$), adjacent nodes (upstream parents and downstream children) are monitored for a window of $3$ ticks. If an adjacent node also approaches the failure threshold during that window, it degrades. The cascade does not spread infinitely — it is bounded to one graph hop — but it captures the precise chaining dynamic that SRE teams fear during high-severity incidents.

### Priority Queues and Business Criticality

Not every node matters equally. The payment gateway (node-0) carries **double the importance weight** of any other service. Its queue growth contributes twice as much to the Lyapunov energy. Its failure counts as a separate VIP failure metric. SHED_LOAD — the action that drops a percentage of incoming traffic — is **hard-blocked** on critical nodes. The agent receives immediate rejection for attempting it.

These guardrails are not learned. They are designed into the action validator as safety constraints, mirroring the production guardrails that real platform engineering teams enforce: you cannot shed the payment gateway. Ever.

### Edge Traversal and Topological Routing

Request flow is not random. It follows a specific directed acyclic graph (DAG) with fixed edges: node-0 pipes half its outflow to node-1 and half to node-2; node-2 pipes all of its outflow to node-3. Every tick, the simulator traverses this graph in **topological order** (computed via Kahn's BFS), processing parent nodes before their children, computing each node's outflow as min(incoming_traffic, service_rate) and routing it downstream via the edge weights.

This models the precise path a request takes through a microservice architecture. A failed node has $service\_rate = 0$, so its outflow stops and its children are starved — a causal failure chain the agent must learn to route around using REROUTE_TRAFFIC.

### Hockey-Stick Latency (M/M/1 Queueing)

Latency is not linear. Under light load, increasing traffic by 10% produces negligible latency growth. But as utilization crosses 90%, latency enters the **hockey-stick region** — small increases in traffic produce exponential growth in response time.

AntiAtropos uses a hybrid model:

$$ \text{latency} = \frac{\text{BASE\_LATENCY}}{1 - \text{utilization}} + \text{queue\_depth} \times \text{STEEPNESS} $$

The first term is the M/M/1 queuing formula — it spikes exponentially as $utilization$ approaches $1.0$. The second term ensures that even at the capped $utilization$ of $0.99$, the raw $queue\_backlog$ still contributes meaningful signal. Without this dual structure, the agent would see flattening latency at the cap and lose gradient for further action.

### Cooldown and Action Discipline

Rapid repeated scaling is destabilizing in production — it causes pod churn, wastes scheduler cycles, and indicates a policy that oscillates rather than stabilizes. AntiAtropos enforces a **soft cooldown**: if the agent issues a SCALE_UP or SCALE_DOWN on the same node within $3$ ticks, the action still executes (emergency scaling must not be blocked), but a linear penalty is subtracted from the normalized reward. The agent learns to scale decisively and then wait.

### Shed Decay and Reroute Persistence

SHED_LOAD and REROUTE_TRAFFIC are not permanent commands. A shed fraction decays by $50\%$ per tick, reaching zero within a few ticks unless the agent re-issues it. Similarly, reroute weights decay by $50\%$ per tick. This forces the agent to maintain active control — it cannot set a routing override once and walk away. If the reroute is necessary, the agent must keep issuing it.

---

## The Cluster as a Graph

All of the features above — edge imbalance, backpressure propagation, cascading failures, topological routing, upstream pressure signals — operate on a single connected structure: the **service dependency graph**.

AntiAtropos represents the cluster as a **directed acyclic graph** with labeled weighted edges, node-specific features, and graph-wide energy. Each observation includes not only per-node metrics but also:

- **upstream_nodes** — which parents feed this service
- **downstream_nodes** — which children depend on this service
- **upstream_pressure** — the mean queue depth of all parents, normalized
- **outflow_rate** — the actual requests per tick flowing out on each edge
- **queue_delta** — per-node change from the previous tick
- **sla_proximity** — how close each individual node is to violating its SLA
- **node_reward** — per-node reward decomposition for credit assignment

This means the model receives a **structured graph observation**, not a flat list of metrics. It sees the relationships. It sees which nodes are connected. It learns that when node-2's outflow to node-3 drops to zero, node-3 will starve unless traffic is rerouted.

For the model that operates on this observation, the cluster is effectively a **graph-structured input** — a set of node representations with edge connectivity. While the current architecture uses a transformer-based LLM, the observation format is explicitly structured to support graph-native architectures in future iterations. The edge imbalance term in the Lyapunov graph energy ensures that intermediate rewards already carry topological signal.

Real production systems are not independent servers. They are interdependent networks — databases feeding APIs feeding frontends, with dependencies that create both throughput and fragility. AntiAtropos is the only training environment that models infrastructure this way by design, not as an afterthought.

---

## Complete Feature Registry

The following features are active in the current simulator and reward function. Each contributes measurable gradient signal to the agent's learning trajectory.

| # | Feature | What It Models | Effect on Learning |
|---|---------|---------------|-------------------|
| 1 | **Lyapunov Graph Energy** | Structural disorder across the entire DAG, including edge-level imbalance | Agent learns to balance load across topology, not just minimize individual queues |
| 2 | **Drift-Plus-Penalty Reward** | Neely framework: opposes stability against cost at every step | Agent cannot trivially over-provision; must find minimum-cost stability |
| 3 | **$5$-Tick Boot Delay** | Kubernetes pod cold-start latency modeled as pending capacity queue | Agent must scale predictively; reactive scaling fails by design |
| 4 | **Jitter ($\pm 10$ ticks)** | Surge window randomized per episode | Agent cannot memorize event timing; must detect signal from cluster state |
| 5 | **Sensor Dropout ($5\%$)** | Random zero/invalid readings simulate scrape failures | Agent learns to operate under partial observability using context |
| 6 | **Hockey-Stick Latency** | Dual M/M/1 + backlog model prevents gradient flattening near saturation | Agent receives continuous gradient even at extreme utilization |
| 7 | **Backpressure Propagation** | Children overload → parent service-rate reduction (up to $40\%$ cap) | Agent respects downstream congestion as a causal constraint on upstream nodes |
| 8 | **Cascading Failure Detection** | Bounded $3$-tick window with graph-hop propagation | Agent sees threat of sequential failure chain during high-stress incidents |
| 9 | **Business-Critical Priority Weights** | VIP nodes have $2\times$ contribution to Lyapunov energy and SLA measurements | Agent preferentially protects critical services over auxiliary nodes |
| 10 | **Action Safety Guardrails** | SHED_LOAD hard-blocked on critical nodes; NO_OP always valid | Agent cannot find trivial shortcuts; safety constraints mirror production policy |
| 11 | **Cooldown Penalties** | Soft penalty for re-scaling same node within $3$ ticks | Agent learns to scale decisively and wait; discourages oscillatory policies |
| 12 | **Shed & Reroute Decay** | Effects decay $50\%$ per tick unless re-issued | Agent must maintain active control; cannot set-and-forget routing overrides |
| 13 | **Topological Traffic Routing** | DAG traversal in Kahn order ensures correct parent→child outflow | Agent sees causal dependency chains in observation structure |
| 14 | **Edge Imbalance Term** | Lyapunov includes $\sum |parent\_queue - child\_queue|$ | Agent receives gradient about load *distribution*, not just load *magnitude* |
| 15 | **Three-Tier Cost Model** | Baseline (sunk), Justified ($4\times$ base), Idle Waste ($20\times$ base) | Agent distinguishes between paid-for capacity and chosen waste |
| 16 | **Smooth SLA Sigmoids** | Dual sigmoid replaces binary threshold with continuous gradient | Agent receives preventive signal before violation, enabling pre-scale behavior |
| 17 | **Control-Barrier Function** | Quadratic penalty fires above $Q=150$, architecturally separate from SLA | Agent faces hard danger zone distinct from warning zone; urgency escalates correctly |
| 18 | **Per-Node Reward Decomposition** | Credit assignment broken out per node in observation | Agent learns which nodes are responsible for reward changes |
| 19 | **Domain Randomization** | All task parameters randomized per episode via configurable seed | Policy trained for robustness, not memorization of specific settings |
| 20 | **Multi-Mode Operation** | SIMULATED → HYBRID → LIVE → AWS deployment modes | Same policy transitions from training to production with zero retraining |
| 21 | **Live Kubernetes Bridge** | `kubernetes_executor.py` maps actions to real `kubectl` commands | Trained policies drive actual infrastructure; environment-to-production gap closed |

---

## Three Tasks, Progressive Challenge

These tasks are not separate scenarios. They form a **curriculum** designed to teach an agent progressively more complex operational reasoning.

- **Task 1 — Capacity Ramp.** Linear load growth from near-saturation. The agent must scale predictively ahead of the ramp, not reactively after queues build. Boot delay makes this a timing problem, not a raw capacity problem. A successful policy scales up early and scales down during idle windows — demonstrating both capacity management and cost discipline.

- **Task 2 — Fault Tolerance.** At a randomized tick, a non-critical node fails permanently. Its service rate drops to zero, inflow is fully dropped, and its downstream children are starved. The trained agent learns to detect this failure by noticing the queue collapse and outflow cessation, then issues REROUTE_TRAFFIC to shift the parent's traffic split away from the dead child, followed by targeted SCALE_UP to absorb the displaced load elsewhere — the exact sequence an on-call SRE would execute during a live incident.

- **Task 3 — Surge Stability.** A $60$ req/tick surge arrives directly at node-1 and node-2 from a side channel that bypasses the primary ingress. The surge is periodic but jittered — the agent cannot predict timing from history alone. SHED_LOAD is permitted on surge nodes (they are not critical), but the correct response is precision scaling — the agent must read the early queue buildup and pre-scale before the surge peak. This is the hardest problem in the suite and the closest analog to a real high-severity production incident.

---

## Training Architecture

We trained a **Qwen3.5-4B** model with **QLoRA** (rank-$64$, $7$ target modules, $\sim$1.6\% trainable parameters) using **REINFORCE with baseline** as the loss function. Training ran on a single NVIDIA A10G GPU ($24$ GB VRAM) at approximately $\$0.34$/hr on Hugging Face Jobs. The full training run was configured as:

| Parameter | Value | Rationale |
|-----------|-------|----------|
| Base model | Qwen/Qwen3.5-4B | Strong reasoning / code generation for structured JSON actions |
| LoRA rank | $64$ | $+83$ MiB vs rank-$32$ — negligible cost, meaningfully more expressiveness |
| Target modules | q, k, v, o, gate, up, down projections | Full attention + FFN adaptation |
| Loss function | REINFORCE + baseline | No value head needed; running-mean baseline reduces variance |
| Iterations | $500$ | Converged by iteration $\sim$300; remaining iterations refine edge behaviors |
| Episodes per iteration | $6$ | $2 \times$ tasks ensures curriculum balance each iteration |
| Max steps per episode | $20$ | Enough for failure to occur + recovery, keeps iteration time manageable |
| Learning rate | $1.0 \times 10^{-5}$ | Conservative for LoRA on 4-bit base; avoids catastrophic forgetting |
| Sequence length | $768$ | System prompt + observation + action + template $\leq 640$ peak; $20\%$ headroom |
| Generation temperature | $0.85$ | Encourages exploration diversity during policy gradient rollout |
| Entropy coefficient | $0.001$ | Per-token entropy bonus; prevents policy from collapsing to deterministic too early |

### Co-Located Simulator Architecture

The simulator runs as a **local FastAPI server** inside the same HF Job container on CPU only, while the GPU is fully dedicated to model forward/backward passes. This eliminates network latency between action generation and environment feedback — each HTTP call resolves to `localhost:8000` at sub-millisecond latency.

### Curriculum-Balanced Iterations

Each iteration collects $6$ episodes — $2$ per task — ensuring the gradient signal covers all three curriculum stages evenly. Without this balance, the agent would overfit to Task 1 (the simplest) and fail to generalize to failures or surges. Task rotation is deterministic within an iteration but seeds are randomized per episode, so the agent never sees the same sequence of cluster states twice.

> **Reference Run:** All artifacts from this training — checkpoints, metrics logs, plots, and evaluation results — are available on the Hugging Face Hub at [Keshav051/antiatropos-qlora/run_0011](https://huggingface.co/Keshav051/antiatropos-qlora/tree/main/run_0011). This is the **canonical REINFORCE run** that produced the results shown in this blog. The `logs/` directory in the project repository also contains local copies of these artifacts.

---

## The RL Training Pipeline

### Step 1: Parallel Episode Rollouts (Batched Generation)

The most expensive operation in training is LLM generation — each forward pass over $36$ layers with KV-cache. Running $6$ episodes sequentially would require $6 \times 20 = 120$ serial forward passes per iteration. Instead, we batch all active episodes' observations into a **single left-padded forward pass**:

1. All $6$ observations are tokenized independently.
2. They are **left-padded** to the same sequence length (crucial for causal LM inference — padding tokens before content ensures the model attends correctly).
3. A single `model.generate()` call produces actions for all episodes simultaneously.
4. Episode environments are stepped in parallel via `ThreadPoolExecutor`.

This reduces $120$ sequential generation calls to $20$ batched calls — a **$6\times$ speedup** on generation. The parallel env steps are I/O bound (HTTP to localhost) and complete in near-constant time regardless of batch size.

### Step 3: REINFORCE with Baseline

We use the standard REINFORCE policy gradient with a running-mean baseline:

$$ \nabla J(\theta) = \mathbb{E} \left[ \sum_t \nabla \log \pi_\theta(a_t | s_t) \cdot (G_t - \bar{b}) \right] $$

Where:
- $G_t = \sum_{k=0}^{T-t} \gamma^k r_{t+k}$ is the discounted return ($\gamma = 0.99$).
- $\bar{b}$ is the exponential moving average of past episode returns ($\alpha = 0.01$).
- Advantages are **normalized to zero mean, unit variance** across each batch — critical because raw rewards span orders of magnitude ($-0.01$ to $-12.0$).

#### Action-Token Masking

A critical detail often overlooked: we compute log-probabilities **only over the generated action tokens**, not the prompt tokens. The full sequence `[prompt | action]` is fed through the model, but the loss mask zeros out all prompt-position tokens. This prevents the gradient from pushing on tokens the model cannot control — without this masking, the REINFORCE gradient is dominated by noise from prompt reconstruction.

Log-probs are also **length-normalized** by the number of action tokens. This prevents sequences with longer action outputs from receiving disproportionately large gradients.

#### Entropy Bonus for Exploration

A per-token entropy bonus is added to the loss to prevent premature policy collapse:

$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{REINFORCE}} - \beta \cdot \frac{1}{N} \sum_t H(\pi_\theta(\cdot | s_t)) $$

Where $\beta = 0.001$. The entropy is computed over **action tokens only** — prompt tokens contribute no entropy signal.

---

## VRAM Engineering: Avoiding OOM on a $24$ GiB GPU

The A10G has $22.3$ GiB of usable VRAM. With a $4$-bit $4$B model occupying $\sim$3.3 GiB and a single forward pass requiring $\sim$8.9 GiB of activations, we had $\sim$10 GiB of headroom — but only if we managed memory carefully. Three techniques made training feasible:

### 1. Per-Mini-Batch Gradient Accumulation

The naive approach: collect all episode transitions, compute log-probs for all of them in one giant forward pass, then call `.backward()`. This materializes the computation graph for ALL transitions simultaneously — $\sim$26 GiB peak for just $3$ batches, causing an immediate OOM.

Instead, we split the transitions into mini-batches of size $1$ and process them in a loop:

```
for each mini-batch:
    1. Forward pass → compute NLL + entropy for this mini-batch only
    2. Compute REINFORCE loss for this mini-batch
    3. .backward() immediately → frees computation graph
    4. .zero_grad() the optimizer
    5. Discard logits, hidden states, and intermediate tensors
```

Only **one** forward pass worth of activations ($\sim$8.9 GiB) lives in VRAM at any time. The LoRA gradients accumulate in the parameter `.grad` buffers (only $\sim$170 MiB for $42$M parameters).

### 2. Fused Cross-Entropy (No Logit Materialization)

Standard cross-entropy loss in PyTorch materializes the full `[B, S-1, V]` log-probability matrix, where $V = 151,936$ (Qwen's vocabulary). At `B=1, S=768`, that's $\sim$950 MiB — just for one intermediate tensor.

`F.cross_entropy(reduction='none')` uses a single fused CUDA kernel that computes log-softmax + NLL in one pass, materializing only the per-position scalar NLL result ($\sim$6 KiB). This saves nearly $1$ GiB per mini-batch.

### 3. CPU Offloading of Rollout Data

Immediately after generation, all tokenized sequences are moved from GPU to CPU:

```python
for ep in episodes:
    for t in ep.transitions:
        t.input_ids = t.input_ids.cpu()
        t.attention_mask = t.attention_mask.cpu()
```

The generation KV-cache can peak at $\sim$12 GiB for a batch of $6$ sequences. Offloading immediately frees this for the loss forward pass. Only the current mini-batch's tensors are moved back to GPU during the loss loop.

These three optimizations together keep peak VRAM below $\sim$18 GiB, maintaining a $4$-$5$ GiB safety margin on the A10G.

---

## Smart Entropy Calculation: Chunked Vocab Processing

Computing the per-token entropy $H(p) = -\sum_v p_v \log p_v$ normally requires materializing the full `[B, S-1, V]` log-probability matrix — $950$ MiB for a single forward pass. AntiAtropos avoids this with a **chunked reduction** over the vocabulary dimension:

```python
log_Z = logits.logsumexp(dim=-1, keepdim=True)  # (B, S-1, 1) — 4 MiB
entropy = torch.zeros(B, S-1)
for v_start in range(0, V, 4096):
    chunk = logits[:, :, v_start:v_start + 4096]
    log_p_chunk = chunk - log_Z                     # (B, S-1, 4096) — 32 MiB
    p_chunk = log_p_chunk.exp()                     # (B, S-1, 4096) — 32 MiB
    entropy += -(p_chunk * log_p_chunk).sum(dim=-1) # accumulates in-place
```

Instead of allocating $950$ MiB for the full `[B, S-1, 151936]` matrix, we process the vocabulary in $4096$-column chunks. Each chunk allocates only $\sim$64 MiB temporarily. The entropy accumulator stays as a single `[B, S-1]` tensor ($\sim$6 KiB). Total savings: $\sim$900 MiB per forward pass.

This chunked approach is possible because entropy is a sum over independent dimensions — no information is shared across vocabulary columns, so the accumulation is trivially parallel and numerically identical to the full materialization.

---

## Evaluation Protocol

Every $50$ iterations, we run a full evaluation comparing the fine-tuned model against a **task-aware heuristic baseline** across all three tasks. The heuristic uses hand-crafted rules:

- **Task 1**: Scale up when queue depth $> 0.5$; scale down when capacity $> 0.7$ and queue $< 0.2$.
- **Task 2**: On detecting a FAILED node, reroute traffic away from it and scale up starved children.
- **Task 3**: Scale node-1/2 when queue exceeds $0.3$; shed load on node-3/4 as fallback.

Each evaluation runs $2$ episodes per task with deterministic seeds (same initial conditions for both FT and heuristic). The comparison is rigorous: same seed, same task, same max steps. Metrics tracked:

- **Average reward** per task (normalized $[0, 1])$.
- **Invalid action rate** — malformed JSON, critical-node violations.
- **SLA violations** — steps where latency $> 200$ ms or error rate $> 5\%$.

![Reward graph comparing fine-tuned model vs heuristic across tasks](/images/reward_graph.png)

### Results-Driven Checkpointing

Checkpoints are saved every $5$ iterations (approximately every $15$ minutes) to the Hugging Face Hub under `hub_model_repo/<run_id>/checkpoint-NNNN/`. At $\$0.34$/hr, a lost job costs mere cents, but losing $10$ iterations of training progress due to a transient error is wasteful. Frequent checkpointing ensures that even if an HF Job is preempted, the next run can resume from the latest Hub checkpoint automatically.

### Metrics Logging Pipeline

Every step of every episode is logged as a structured JSONL row with:
- Per-node queue depth, latency, inflow, outflow, capacity, pending capacity
- Cluster-level: average latency, error rate, total queue backlog, cost per hour, SLA violations
- Action: type, target, parameter, validity flag, raw reward
- Identity: run ID, iteration, episode index, task ID, step number, UTC timestamp

This produces a richly queryable metrics dataset that can be used to:
- Replay any episode step-by-step
- Compute custom metrics (e.g., time-to-recovery after failure)
- Diagnose reward stagnation by inspecting specific action sequences
- Generate plots (loss curves, reward curves, action distributions) post-hoc without re-running training

---

## Results: Learned Behavior

After 500 training iterations, the model demonstrated measurable improvement across all three tasks:

| Task | Baseline (NO-OP) | Trained Agent | Behavioral Observation |
|------|-----------------|---------------|----------------------|
| task-1 | $0.70$ | **$0.80+$** | Learned to scale down during idle periods — cost-aware, not just stability-seeking |
| task-2 | $0.70$ | **$0.82$** | Learned to detect failure → reroute → scale — exact SRE incident response protocol |
| task-3 | $0.21$ | **$0.94$** | Learned pre-scale from early queue signal — predictive capacity planning |

Action distribution shifted from SCALE_UP-heavy ($60\%$) to a balanced spread across all five action types, with SCALE_DOWN rising from $2\%$ to $20\%$ — the signature of an agent that manages capacity, not just adds it.

![Action type distribution comparing NO-OP baseline policy vs fully trained policy](/images/actions_taken.png)

The fine-tuned policy learned several non-trivial behaviors:

- **Cost-aware scaling down.** During low-traffic windows, the agent actively reduces capacity on non-critical nodes — it has learned the $20\times$ idle penalty and adjusts accordingly.
- **Failure detection via queue collapse.** When a node fails, its queue empties (inflow stops) but children begin starving. The agent reads this pattern rather than waiting for a FAILED status flag.
- **Pre-scaling before surges.** In Task 3, the agent raises capacity on node-1 and node-2 at the first signal of queue buildup, well before latency reaches SLA thresholds. The $5$-tick boot delay makes reactive scaling impossible — this behavior can only be learned.
- **Reroute precision.** In Task 2, the agent reroutes **from** the failed node (not to it), then scales the starved children — the exact protocol a human SRE would follow.
- **Reduced invalid actions.** The invalid action rate dropped from $\sim$15\% to $\sim$3\% as the model learned to avoid SHED_LOAD on critical nodes and generate well-formed JSON.

---

## Why This Architecture Is Superior

**It measures direction, not position.** Lyapunov drift grades the agent on whether it moved the cluster toward or away from equilibrium. Most SRE tools measure whether a threshold is crossed. One is about trajectory; the other is about a single point in time.

**It sees topology, not just nodes.** The edge imbalance term captures load distribution mismatches that flat per-node metrics fundamentally miss. A parent-child queue mismatch is invisible to per-node dashboards. AntiAtropos makes it the agent's explicit concern.

**It acts before, not after.** Smooth SLA sigmoids give gradient before violation. Boot delay creates a hard operating window where late action fails. The interaction between these design choices is deliberate: the agent can only succeed if it learns the pre-scale window. There is no fallback plan that succeeds reactively.

**It prices what matters.** The three-tier cost model separates sunk infrastructure from agent-added capacity from idle waste. The agent is never penalized for the cluster simply existing. It is penalized for choosing to leave capacity idle.

**It is formally grounded.** The Drift-Plus-Penalty framework by Neely et al. [1] provides a theoretical guarantee: minimizing this expression produces a queue-stable policy with bounded average cost. The agent is not learning opinionated heuristics. It is optimizing a mathematical objective with known stability properties.

**It transfers to production.** The same action space that controls the simulator maps directly to Kubernetes operations. The environment runs in SIMULATED mode for training, HYBRID mode for validation against live metrics, and LIVE mode for direct cluster control — all through the same interface and the same policy.

---

## Quick Start

The environment is live on Hugging Face Spaces. Connect from any OpenEnv-compatible client:

```python
pip install openenv-core
from openenv import Environment

env = Environment("https://keshav051-antiatropos.hf.space")
obs = env.reset(task_id="task-3")
for step in range(60):
    obs = env.step({"action_type": "NO_OP", "target_node_id": "node-0", "parameter": 0.0})
```


## Future Horizons: The Road to Provable Autonomy
AntiAtropos is not a static tool; it is a prototype for a new era of infrastructure. Our next steps involve expanding the state-space to include **Multi-Cluster Coordination** and implementing **Formal Safety Verification**—using the Lyapunov certificates generated during training to prove that an agent's policy will never violate safety barriers under a given traffic envelope.

We are moving toward a world where infrastructure manages itself, not through a collection of scripts, but through an inherent understanding of its own physics.

---

## References

[1] M. J. Neely et al., "Stochastic Network Optimization with Application to Communication and Queueing Systems," *Synthesis Lectures on Communication Networks*, vol. 3, no. 1, pp. 1-211, 2010.

---

## The Road Not Taken: GRPO Experimentation

### Reference Runs on the Hub

All training runs — including the canonical REINFORCE run and our GRPO experiments — are publicly available for inspection:

| Run | Loss Type | Description | Link |
|-----|-----------|-------------|------|
| **run_0011** | REINFORCE + baseline | **Reference run** — fully converged policy after 500 iterations. This trained the model discussed above. | [View on Hub](https://huggingface.co/Keshav051/antiatropos-qlora/tree/main/run_0011) |
| **grpo_run_001** | GRPO | Experimental GRPO run with K=2, comparing group-relative advantage against the running-mean baseline. | [View on Hub](https://huggingface.co/Keshav051/antiatropos-qlora/tree/main/grpo_run_001) |

The `logs/` directory at the project root also contains local copies of these artifacts for offline inspection.

### Why not GRPO?

Our initial research included an attempt at **Group Relative Policy Optimization (GRPO)**. While GRPO is excellent for relative ranking, we found that for the high-dimensional physics of a microservice cluster, **QLoRA REINFORCE** offered a more stable gradient. Computationally, QLoRA allowed us to maintain a Rank-64 adapter density that GRPO struggled to match within the same VRAM constraints. Ultimately, the REINFORCE approach yielded 1:1 performance results with a much faster iteration loop.

---

*Built for the 2026 AntiAtropos Hackathon.*
