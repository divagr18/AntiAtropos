# RL Pipeline Overhaul

## Phase 1: Simulator Physics

### Task 1.1: Exponential Latency Model
**File:** `simulator.py` line 426

Replace the linear latency formula with M/M/1 queuing theory:
```python
# Current (linear):
n.latency_ms = BASE_LATENCY_MS + (n.queue_depth * LATENCY_STEEPNESS)

# New (exponential — blows up as utilization→1):
utilization = n.incoming_request_rate / n.service_rate if n.service_rate > 0 else 1.0
if utilization >= 0.99:
    utilization = 0.99  # cap to prevent infinity
n.latency_ms = BASE_LATENCY_MS / (1.0 - utilization)
```
This creates the "hockey stick" that teaches the agent to scale *before* saturation.

### Task 1.2: Node Recovery Mechanic
**File:** `simulator.py` lines 428-441, `NodeState` dataclass

- Add `recovery_timer: int = 0` to `NodeState`
- When `queue_depth > FATAL_FAIL_THRESHOLD`, set status=FAILED but start `recovery_timer = 20` ticks
- Each tick, decrement recovery_timer. When it hits 0, set status=HEALTHY, capacity=1, queue_depth=0
- This lets the agent learn recovery strategies (reroute away, then scale up the recovering node)

### Task 1.3: Cascading Failure Pressure
**File:** `simulator.py` — new method `_cascade_failures()`

When a node fails, its peers absorb the lost capacity. If any peer's queue then exceeds `FATAL_FAIL_THRESHOLD * 1.2` within 3 ticks of the original failure, that peer also degrades. This models real cascade patterns. Called after `_update_statuses()` in `tick()`.

---

## Phase 2: Reward Shaping

### Task 2.1: Smooth SLA Penalty (Replace Binary Cliff)
**File:** `server/AntiAtropos_environment.py` line 205, `stability.py`

Replace the binary SLA violation with a smooth sigmoid that ramps up as latency approaches the threshold:
```python
# Instead of:
sla_violation_step = 1 if (avg_latency > 200.0 or error_rate > 0.05) else 0

# New:
def smooth_sla_penalty(avg_latency_norm: float, error_rate: float, 
                       threshold: float = 0.20, temperature: float = 0.03) -> float:
    """Smooth penalty in [0, 1] that ramps as latency approaches threshold."""
    lat_penalty = 1.0 / (1.0 + math.exp(-(avg_latency_norm - threshold) / temperature))
    err_penalty = 1.0 / (1.0 + math.exp(-(error_rate - 0.05) / 0.01))
    return max(lat_penalty, err_penalty)
```
This gives the agent gradient signal *before* the SLA is actually violated.

### Task 2.2: Activate the Barrier Function
**File:** `server/AntiAtropos_environment.py` lines 213-222, `stability.py`

Add `compute_barrier()` to the reward formula:
```python
raw_reward = compute_reward(
    v_prev=self._prev_lyapunov,
    v_curr=current_lyapunov,
    cost=cost,
    sla_violation_step=sla_violation_step,  # now smooth, not binary
    alpha=ALPHA,
    beta=BETA,
    gamma=GAMMA,
    barrier=compute_barrier(self._nodes_true),  # NEW
    delta=DELTA,                                 # NEW weight
)
```
Update `compute_reward()` in `stability.py` to accept and include the barrier term:
```
R_t = -(α·ΔV + β·Cost + γ·SLA_smooth + δ·Barrier)
```

### Task 2.3: Per-Node Reward Decomposition
**File:** `server/AntiAtropos_environment.py`, new method `_compute_node_rewards()`

Add per-node reward components to `ClusterObservation` so the agent can learn credit assignment:
```python
# In NodeObservation, add:
node_reward: float = 0.0  # per-node reward contribution

# Compute as:
for node in nodes_true:
    node_delta_v = importance_weight * (node_queue² - prev_node_queue²)
    node_barrier = max(0, node_queue - Q_BARRIER_MAX)²
    node.cost = node_capacity * COST_PER_CAPACITY_UNIT_PER_HOUR
    node_reward = -(ALPHA * node_delta_v + DELTA * node_barrier + BETA * node_cost)
```
This tells the agent *which* nodes improved from its actions.

---

## Phase 3: Observation + Action Space

### Task 3.1: Enrich Observations
**File:** `models.py` — `NodeObservation`, `inference.py` — `observation_for_model()`

Add to `NodeObservation`:
- `capacity: float` — current capacity units (0-5)
- `pending_capacity: float` — capacity being booted (0-5)
- `queue_delta: float` — queue depth change from last tick (-1 to +1, normalized)
- `sla_proximity: float` — how close this node is to SLA violation (0=safe, 1=violating)

Add to `ClusterObservation`:
- `reward_components: dict` — breakdown of the reward (drift, cost, sla, barrier)

Update `observation_for_model()` in `inference.py` to include `is_vip`, `importance_weight`, and the new fields.

### Task 3.2: Make SHED_LOAD and REROUTE_TRAFFIC Persistent
**File:** `simulator.py` lines 252, 270-271, 386-390

- SHED_LOAD: Instead of resetting `shed_fraction=0.0` every tick, decay it by 80% per tick (`shed_fraction *= 0.2`). The agent still needs to re-issue to maintain full effect, but the decay is gradual.
- REROUTE_TRAFFIC: Change decay from 50% to 80% per tick (`weight *= 0.2` instead of `*= 0.5`). Makes the effect last longer.

### Task 3.3: Add Action Cooldown
**File:** `control/validation.py`, `server/AntiAtropos_environment.py`

Track last action per node. If the agent issues SCALE_UP on node-0 twice within 3 ticks, the second one is rejected with "Cooldown: node-0 was scaled 2 ticks ago." This prevents thrashing and teaches the agent to wait for actions to take effect (especially important with BOOT_DELAY_TICKS=5).

---

## Phase 4: Training Loop

### Task 4.1: Episode Replay Buffer
**File:** New file `replay.py`

Store episode trajectories (obs, action, reward, done) in a rolling buffer. After each episode:
1. If `composite_score > SUCCESS_SCORE_THRESHOLD`, store the full trajectory as a "positive example"
2. If `composite_score < 0.3`, store as a "negative example"
3. Use positive examples as few-shot demonstrations in the LLM prompt

```python
class EpisodeReplayBuffer:
    def __init__(self, max_episodes: int = 50):
        self._positive: deque = deque(maxlen=max_episodes)
        self._negative: deque = deque(maxlen=max_episodes)
    
    def store(self, trajectory, score):
        if score >= 0.55:
            self._positive.append(trajectory)
        elif score < 0.3:
            self._negative.append(trajectory)
    
    def sample_demonstrations(self, n: int = 2) -> list:
        """Sample n positive episodes for few-shot prompting."""
        return random.sample(self._positive, min(n, len(self._positive)))
```

### Task 4.2: Few-Shot Prompt with Demonstrations
**File:** `inference.py` — `build_user_prompt()`, `SYSTEM_PROMPT`

Add positive trajectory examples to the prompt. After running a few episodes to populate the buffer:
```
Here is an example of a successful action sequence for a similar situation:
Step 15: {"action_type": "SCALE_UP", "target_node_id": "node-0", "parameter": 0.8} reward=0.72
Step 16: {"action_type": "NO_OP", "target_node_id": "node-0", "parameter": 0.0} reward=0.81
...
```

### Task 4.3: Multi-Episode Evaluation with Temperature Sweep
**File:** `inference.py` — `run_single_task()`, `run_all_tasks()`

- Run each task 3 times instead of once
- Sweep temperature: [0.0, 0.3, 0.7] across runs
- Report mean and std of composite score
- This gives variance estimation and lets exploration happen

### Task 4.4: Curriculum Training
**File:** New file `curriculum.py`, `inference.py`

Define progressive difficulty stages:
```python
CURRICULUM = [
    {"task": "task-1", "max_steps": 60, "difficulty": "easy",    "pass_threshold": 0.50},
    {"task": "task-1", "max_steps": 100,"difficulty": "normal",  "pass_threshold": 0.55},
    {"task": "task-2", "max_steps": 60, "difficulty": "easy",    "pass_threshold": 0.45},
    {"task": "task-3", "max_steps": 60, "difficulty": "easy",    "pass_threshold": 0.45},
    {"task": "task-2", "max_steps": 100,"difficulty": "normal",  "pass_threshold": 0.55},
    {"task": "task-3", "max_steps": 100,"difficulty": "normal",  "pass_threshold": 0.55},
]
```
The agent must pass each stage before advancing. Failed stages are retried with higher temperature.

### Task 4.5: Episode-Level Bonuses
**File:** `grader.py` — `Grade.composite`, `server/AntiAtropos_environment.py`

Add terminal bonuses to the final step's reward:
- `+0.5` if zero VIP failures throughout the episode
- `+0.3` if SLA violations < 3 for the whole episode
- `+0.2` if no barrier violations (queues never exceeded Q_BARRIER_MAX)

These reward *prevention*, not just *reaction*.

---

## Implementation Order

```
Phase 1 (Sim)  →  Phase 2 (Reward)  →  Phase 3 (Obs/Action)  →  Phase 4 (Training)
     ↓                  ↓                      ↓                        ↓
  1.1 Latency        2.1 Smooth SLA        3.1 Enrich Obs          4.1 Replay Buffer
  1.2 Recovery       2.2 Barrier            3.2 Persistent Acts     4.2 Few-Shot
  1.3 Cascade        2.3 Per-Node Reward    3.3 Cooldown            4.3 Multi-Episode
                                                                   4.4 Curriculum
                                                                   4.5 Bonuses
```

Each task is independently testable. The reward changes (Phase 2) depend on the sim changes (Phase 1) being done first. The training loop (Phase 4) benefits from all prior phases but can be developed incrementally.
