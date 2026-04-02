# AntiAtropos OpenEnv to Real-System Roadmap

This document is a build plan for a future LLM or engineer working inside this repository.
Its purpose is to convert AntiAtropos from a JSON-only simulator into a real OpenEnv-native
closed-loop SRE environment backed by Prometheus telemetry, Kubernetes actions, and Grafana
visualization, while preserving the current benchmark tasks and reward structure.

## 0. Mission

Build a system that behaves like a real site-management control loop:

1. An agent interacts only through the OpenEnv API.
2. The environment ingests real telemetry from Prometheus.
3. The environment converts telemetry into a physics-based state estimate.
4. The environment sends bounded actions to Kubernetes or a safe action executor.
5. Grafana visualizes the same live state and action history.
6. The environment remains deterministic enough for benchmark tasks 1 to 3.

The end state should feel like a literal physics engine inside site management software.

## 1. Non-Negotiable Constraints

- Preserve the OpenEnv contract: `reset()`, `step()`, and `state()`.
- Keep typed Pydantic schemas for action and observation objects.
- Keep benchmark tasks 1, 2, and 3 runnable in pure simulation mode.
- Do not expose raw Prometheus or Kubernetes APIs directly to the agent.
- Do not let the agent execute arbitrary cluster commands.
- Keep action space bounded and validated.
- Keep a safe fallback path for offline gitdemos and hackathon judging.

## 2. System Vision

The system should support three execution modes:

- `simulated`: use the current synthetic simulator only.
- `hybrid`: ingest live Prometheus telemetry, but still use the physics engine for state and reward.
- `live`: execute bounded actions on a real or sandboxed Kubernetes cluster.

The same OpenEnv observation and action schema must work in all three modes.

## 3. Target Architecture

### 3.1 Data Flow

Prometheus -> telemetry adapter -> physics/state estimator -> OpenEnv observation

OpenEnv action -> action validator -> executor -> Kubernetes or safe mock executor

Kubernetes/workloads -> Prometheus -> telemetry adapter -> physics/state estimator

Grafana reads the same telemetry and action history for human visibility.

### 3.2 Functional Layers

1. Telemetry ingestion layer
2. Physics/state estimation layer
3. Action execution layer
4. OpenEnv orchestration layer
5. Visualization layer

## 4. What to Build

### 4.1 Telemetry Ingestion

Build a Prometheus adapter that periodically queries cluster metrics and converts them into normalized node records.

Suggested module:

- `AntiAtropos/telemetry/prometheus_client.py`

Responsibilities:

- Query Prometheus on a fixed interval.
- Pull per-service and per-node metrics.
- Handle missing samples and stale samples.
- Attach labels for VIP nodes and critical services.
- Cache the latest telemetry snapshot.
- Return a normalized intermediate record, not raw PromQL output.

Important metrics to ingest if available:

- request rate
- latency percentiles
- error rate
- CPU usage
- memory usage
- pod readiness
- restart count
- queue depth proxies
- saturation indicators
- HPA or KEDA scaling state

### 4.2 Physics / State Estimation

Refactor the simulator into a state estimator that maps telemetry into a queueing and stability model.

Existing file to modify:

- `AntiAtropos/simulator.py`

Existing file to modify:

- `AntiAtropos/stability.py`

Responsibilities:

- Maintain per-node queue depth.
- Maintain service rate.
- Simulate boot delay and scaling lag.
- Simulate overload, recovery, and hysteresis.
- Simulate action efficacy degradation under thrashing.
- Weight VIP nodes more heavily than normal nodes.
- Compute Lyapunov energy and drift.
- Compute barrier violations.

Desired physics features:

- queue growth when incoming load exceeds service rate
- delayed capacity increase after scale-up
- traffic surges and ramps
- failure propagation
- controlled shedding on non-critical services only
- choke/throttle behavior when instability rises
- VIP penalty amplification

### 4.3 Action Execution

Add a safe execution layer that converts OpenEnv actions into Kubernetes-safe operations.

Suggested module:

- `AntiAtropos/control/kubernetes_executor.py`

Responsibilities:

- Scale deployments or replica sets.
- Adjust service routing or ingress weights.
- Apply safe throttles or admission control.
- Reject or penalize illegal actions.
- Return action acknowledgement metadata.

This layer must never expose arbitrary shell execution to the agent.

### 4.4 OpenEnv Orchestration

Keep OpenEnv as the single stable interface.

Existing file to modify:

- `AntiAtropos/server/AntiAtropos_environment.py`

Responsibilities:

- Own the episode state.
- On `reset()`, initialize the telemetry baseline and state estimator.
- On `step(action)`, validate the action, execute it, advance physics, pull telemetry, and compute reward.
- On `state()`, return the current latest observation.
- Keep benchmark tasks deterministic in simulation mode.

### 4.5 Visualization

Grafana should show the same world the agent sees.

Suggested dashboards:

- per-node queue depth
- latency
- error rate
- request rate
- CPU and memory
- VIP node status
- invalid action count
- action history
- choke level
- Lyapunov energy
- SLA violation count

## 5. Files to Modify

### 5.1 Core schema

Modify:

- `AntiAtropos/models.py`

Add fields such as:

- `mode`
- `is_vip`
- `importance_weight`
- `metric_timestamp`
- `data_freshness_ms`
- `action_ack_status`
- `choke_level`
- `invalid_action_count`
- `vip_failure_count`

### 5.2 Simulator and physics

Modify:

- `AntiAtropos/simulator.py`
- `AntiAtropos/stability.py`

Add or refine:

- weighted Lyapunov energy
- VIP penalties
- choke/throttle dynamics
- service lag
- failure hysteresis
- action decay
- metrics-to-state reconciliation

### 5.3 Environment orchestration

Modify:

- `AntiAtropos/server/AntiAtropos_environment.py`

Add:

- telemetry adapter integration
- action executor integration
- live/sim/hybrid mode selection
- event log tracking
- observation freshness metadata
- execution acknowledgement metadata

### 5.4 Client

Modify:

- `AntiAtropos/client.py`

Add support for:

- richer observation fields
- action acknowledgement fields
- VIP flags
- mode-aware parsing

### 5.5 Agent scripts

Modify or replace:

- `llm_agent_task1.py`
- `llm_agent_task2.py`
- `llm_agent_task3.py`

Goals:

- keep current benchmark scripts working
- add a generic runner for simulation and live modes
- add a demonstration mode for real telemetry

### 5.6 Documentation

Modify:

- `README.md`
- `instructions.md`

Add:

- architecture overview
- mode descriptions
- Prometheus and Kubernetes story
- Grafana story
- safety boundaries

## 6. Build Order

Follow this order to avoid breaking the current benchmark environment.

### Phase 1: Preserve current benchmark behavior

Goal:

- Keep tasks 1, 2, and 3 runnable exactly as they are today.

Work:

- Add new schema fields without breaking existing parsing.
- Add mode flags with a default of `simulated`.
- Keep the current simulator operational.

Acceptance:

- Task 1 still runs.
- Task 2 still runs.
- Task 3 still runs.
- Existing tests still pass or can be updated minimally.

### Phase 2: Add telemetry adapter

Goal:

- Read Prometheus data into the environment.

Work:

- Create a Prometheus client module.
- Normalize metric snapshots into per-node records.
- Map services to node IDs.
- Keep a mock telemetry source for local development.

Acceptance:

- The environment can run in `hybrid` mode with mock telemetry.
- Observation fields are populated from telemetry snapshots.

### Phase 3: Add physics reconciliation

Goal:

- Convert telemetry into queueing and stability state.

Work:

- Reconcile Prometheus signals with internal backlog estimates.
- Add lag, decay, and boot delay.
- Use weighted Lyapunov energy for VIP services.

Acceptance:

- The observation changes smoothly across ticks.
- VIP node failure creates a visibly larger score penalty.

### Phase 4: Add action executor

Goal:

- Convert actions into safe real-world operations.

Work:

- Create Kubernetes executor module.
- Map current actions to safe control operations.
- Add acknowledgement tracking.

Acceptance:

- Actions can be executed in a sandbox cluster.
- Invalid actions are rejected or penalized.

### Phase 5: Add Grafana dashboard

Goal:

- Show the same environment state visually.

Work:

- Build Grafana panels for all major metrics.
- Add action history and episode state panels.

Acceptance:

- A live demo can show observation, action, and effect in one place.

### Phase 6: Add live demo mode

Goal:

- Run AntiAtropos against real or sandbox telemetry and actions.

Work:

- Add mode selection to the environment.
- Connect telemetry, physics, and action execution.
- Keep simulation as fallback.

Acceptance:

- One OpenEnv interface works in simulated, hybrid, and live modes.

## 7. Recommended Module Layout

Suggested structure:

```text
AntiAtropos/
  models.py
  client.py
  grader.py
  simulator.py
  stability.py
  telemetry/
    __init__.py
    prometheus_client.py
    mapping.py
  control/
    __init__.py
    kubernetes_executor.py
    validation.py
  server/
    AntiAtropos_environment.py
    app.py
```

## 8. Execution Rules for Future LLMs

When implementing this roadmap, obey these rules:

1. Never break the existing OpenEnv API shape.
2. Never remove simulation mode.
3. Never let raw telemetry leak directly into the agent without normalization.
4. Never let the agent issue unbounded infrastructure commands.
5. Always keep VIP node semantics explicit.
6. Prefer additive changes before refactors.
7. Add tests for every new mode or scoring rule.
8. Keep benchmark tasks reproducible.

## 9. Testing Strategy

Add tests for:

- telemetry normalization
- VIP node weighting
- action validation
- forbidden shedding on critical nodes
- mode switching
- stale telemetry handling
- reward stability under hybrid mode
- deterministic simulation mode

Suggested test files:

- `tests/test_grader_and_reward.py`
- `tests/test_randomization_seed.py`
- `tests/test_task2_reroute.py`
- `tests/test_vip_weighting.py`
- `tests/test_prometheus_adapter.py`
- `tests/test_action_executor.py`

## 10. Safety and Demo Strategy

For the hackathon or public demo:

- Use simulation mode as the benchmark default.
- Use hybrid mode for the live demo.
- Use Kubernetes only inside a sandbox namespace.
- Use Prometheus and Grafana for visibility.
- Keep action boundaries strict.
- Keep a fallback mock telemetry source in case the live cluster is unavailable.

## 11. What Success Looks Like

The environment is successful when:

- the agent sees realistic operational telemetry
- the agent can act on a real or sandboxed system
- the physics engine makes delays, overload, and recovery meaningful
- VIP services are more important than ordinary services
- Grafana shows the agent’s decisions and consequences
- OpenEnv remains the same interface across simulation and live systems

## 12. Final Recommendation

Build this in layers.

Do not replace the current simulator.
Instead, promote it into a hybrid physics engine that can consume real telemetry while preserving benchmark reproducibility.

That gives you:

- a credible research/demo story
- a real systems story
- a usable OpenEnv benchmark
- a path from toy JSON environments to production-shaped control environments

