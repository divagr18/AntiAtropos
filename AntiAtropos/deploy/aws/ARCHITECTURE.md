# AntiAtropos Architecture Guide

A complete explanation of how AntiAtropos works across Hugging Face Spaces and AWS, written for someone who is technically strong but new to Kubernetes.

---

## The Big Picture

AntiAtropos trains AI agents to be Site Reliability Engineers (SREs). An SRE agent watches a simulated microservice cluster and decides when to scale services, reroute traffic, or shed load to keep things running smoothly.

The system is split across two platforms:

```
Hugging Face Spaces                      AWS
=====================                    ======================
The "brain"                              The "muscle"

AntiAtropos FastAPI server               EKS (Kubernetes cluster)
  - Runs the simulator                     - Runs the actual microservice pods
  - Runs the SRE agent logic               - The agent scales these pods
  - Queries Prometheus for metrics         - Prometheus Agent scrapes metrics
  - Sends scale commands to K8s            - Metrics flow to AMP
                                           - Grafana (AMG) visualizes it all
```

Why split? HF Spaces is free/cheap for running the Python server. AWS EKS is where the real infrastructure lives that the agent practices on.

---

## Kubernetes Concepts You Need

### Pod

The smallest unit in Kubernetes. A pod is one or more containers that run together. In our case, each pod runs a single nginx container that simulates a microservice (like "payments" or "checkout").

Think of it as: one running instance of a service.

### Deployment

A Deployment is a recipe that tells Kubernetes "keep N copies of this pod running at all times." If a pod dies, the Deployment automatically replaces it.

The key field is `spec.replicas` — this is the number the SRE agent changes when it scales a service up or down.

```
Deployment: payments
  replicas: 3         <-- the agent changes this number
    |
    +-- Pod: payments-abc123   (running)
    +-- Pod: payments-def456   (running)
    +-- Pod: payments-ghi789   (running)
```

**The agent scales replicas, not pods.** When it sets `replicas: 5`, Kubernetes creates 5 pods. When it sets `replicas: 2`, Kubernetes kills 3 pods.

### Service

A Service gives pods a stable network name. Instead of connecting to `payments-abc123` directly (which changes when the pod is recreated), you connect to `payments` (the Service), which routes to whichever pods are healthy.

### Namespace

A namespace is a folder for organizing resources. We use:
- `prod-sre` — where the 5 microservice Deployments live
- `monitoring` — where the Prometheus Agent pod lives
- `kube-system` — where AWS/EKS system pods live

### Node

A node is one EC2 virtual machine in the EKS cluster. Our cluster has 2-4 nodes. Each node runs multiple pods. When all nodes are full and the agent wants to scale up, Kubernetes adds more nodes (up to `maxSize: 4` in our config).

```
EKS Cluster
  Node 1 (t3.medium - 4 vCPU, 8GB RAM)
    Pod: payments-abc123
    Pod: checkout-def456
    Pod: catalog-ghi789
    Pod: prometheus-agent-xyz
  Node 2 (t3.medium - 4 vCPU, 8GB RAM)
    Pod: payments-jkl012    <-- agent scaled payments from 1 to 2
    Pod: cart-mno345
    Pod: auth-pqr678
```

### ResourceQuota

A hard limit on how many resources a namespace can use. We set one on `prod-sre` that caps total pods at 30. This is a safety net — even if the Python code cap fails, Kubernetes itself will refuse to create more than 30 pods.

---

## How the SRE Agent Works

### The Loop

Every "tick" (one step of the simulation), the agent goes through this cycle:

```
1. OBSERVE  -- Read telemetry (CPU, latency, queue depth) from Prometheus
2. DECIDE   -- Choose an action (SCALE_UP, SCALE_DOWN, REROUTE_TRAFFIC, SHED_LOAD, NO_OP)
3. ACT      -- Send the action to KubernetesExecutor
4. REWARD   -- Compute Lyapunov stability reward (was the cluster more or less stable?)
5. REPEAT
```

### How Each Action Works

| Action | What the Agent Decides | What Happens on EKS |
|---|---|---|
| `SCALE_UP` | "node-0 needs more capacity" | `KubernetesExecutor` patches `payments` Deployment: `replicas: 2 -> 5` |
| `SCALE_DOWN` | "node-3 is over-provisioned" | `KubernetesExecutor` patches `cart` Deployment: `replicas: 4 -> 1` |
| `REROUTE_TRAFFIC` | "Move traffic away from node-2" | Currently simulation-only (no live K8s ingress patching) |
| `SHED_LOAD` | "Drop 50% of traffic to node-3" | Currently simulation-only (no live K8s traffic shaping) |
| `NO_OP` | "Do nothing this tick" | Nothing changes on EKS |

### The SCALE_UP Flow in Detail

Here is exactly what happens when the agent decides to scale up `node-0` (the payments service):

```
HF Spaces                                    AWS EKS
----------                                   --------

Agent: "SCALE_UP, node-0, parameter=0.5"
  |
  v
AntiAtroposEnvironment.step()
  |
  v
KubernetesExecutor.execute_with_metadata()
  |
  v
_load_node_workload_map()
  reads: node-0 -> {"deployment": "payments", "namespace": "prod-sre"}
  |
  v
_scale_deployment("SCALE_UP", "node-0", 0.5)
  |
  +-- 1. Read current replicas: apps_v1.read_namespaced_deployment_scale("payments", "prod-sre")
  |      Current replicas = 2
  |
  +-- 2. Calculate delta: max(1, int(0.5 * 3)) = 1
  |      Desired = min(6, 2 + 1) = 3        <-- max_replicas cap from env var
  |
  +-- 3. Patch: apps_v1.patch_namespaced_deployment_scale("payments", "prod-sre",
  |         body={"spec": {"replicas": 3}})
  |
  v                                                     +---------------------------+
Returns: "Ack: SCALE_UP for node-0 -                    | K8s creates 1 new pod:    |
  deployment payments in namespace                       |   payments-newpod-xyz     |
  prod-sre scaled 2->3"                                  +---------------------------+
```

### The Telemetry Flow in Detail

How the agent reads metrics from the real cluster:

```
EKS Cluster                              AMP                          HF Spaces
-----------                              ---                          ----------

Workload pods                            AMP Workspace                AntiAtropos
(payments, checkout...)                  stores all metrics           PrometheusClient
  |                                           ^                        |
  | /metrics (scraped every 15s)              |                        |
  v                                           |                        |
Prometheus Agent                             |                        |
  |                                           |                        |
  | remote-write (SigV4 auth)                 |                        |
  +------------------------------------------->                        |
                                              |                        |
                                              |  HTTPS query           |
                                              +------------------------>
                                              (PROMETHEUS_URL env var)
                                                                       |
                                                                       v
                                                                 _fetch_real_metrics()
                                                                 runs PromQL like:
                                                                   sum(rate(http_requests_total[1m])) by (pod)
                                                                 returns: TelemetryRecord for each node
```

---

## The Three Layers of Scaling Caps

This is the most important thing to understand for cost control. There are **three** independent limits:

### Layer 1: Python Code Cap (Soft)

**Where:** `ANTIATROPOS_MAX_REPLICAS` env var on HF Spaces, read by `kubernetes_executor.py` line 18.

**How it works:** The `_scale_deployment()` method calculates `desired = min(self.max_replicas, current + delta)`. If the agent tries to scale above 6, it gets:

```
Ack: SCALE_UP for node-0 - replicas unchanged at 6 (bounds 1-6)
```

**Can it be bypassed?** Yes. A bug in the code, or someone running `kubectl scale deployment payments --replicas=50` directly.

**Set to:** `6` on HF Spaces.

### Layer 2: Kubernetes ResourceQuota (Hard)

**Where:** `k8s-workloads.yaml` — ResourceQuota on the `prod-sre` namespace.

**How it works:** Kubernetes itself refuses to schedule pods that would exceed the quota. If the namespace already has 30 pods and something tries to create a 31st:

```
Error from server (Forbidden): pods "payments-new" is forbidden:
exceeded quota: prod-sre-quota, requested: pods=1, used: pods=30, limited: pods=30
```

**Can it be bypassed?** Only by someone with cluster-admin access who deletes or edits the ResourceQuota.

**Set to:** 30 pods total, 8 CPU, 8GB RAM.

### Layer 3: EKS Node Group Max Size (Hard)

**Where:** `eksctl-cluster.yaml` — `managedNodeGroups[0].maxSize: 4`.

**How it works:** The Cluster Autoscaler will never add more than 4 nodes. Even if there are 100 pending pods, it stops at 4 nodes. Pending pods just wait.

**Can it be bypassed?** Only by someone editing the node group in the AWS console.

**Set to:** 4 nodes (4 x t3.medium = 8 vCPU, 16GB RAM max).

### How the Three Layers Work Together

```
Agent wants to scale all 5 deployments to 20 replicas each:

Layer 1 (Python cap):      6 replicas max per deployment  -> agent gets "unchanged at 6"
                           5 x 6 = 30 pods maximum

Layer 2 (ResourceQuota):   30 pods max in namespace       -> 31st pod is Forbidden

Layer 3 (Node group):      4 nodes max                     -> if 30 pods don't fit on 4 nodes,
                                                            some stay Pending (no cost)

Worst case with all caps:  30 pods on 4 nodes = ~$160/month
Without any caps:          100 pods on 25 nodes = ~$1,800/month
```

---

## The Mapping: Simulator Nodes to Real Deployments

The simulator has 5 abstract nodes (node-0 through node-4). The `ANTIATROPOS_WORKLOAD_MAP` env var tells the system which K8s Deployment each simulator node maps to:

```
Simulator Node    K8s Deployment    Namespace    Notes
-------------     ---------------   ---------    -----
node-0            payments          prod-sre     VIP (4x importance weight)
node-1            checkout          prod-sre     Critical (no SHED_LOAD)
node-2            catalog           prod-sre     Critical (no SHED_LOAD)
node-3            cart              prod-sre     Non-critical (sheddable)
node-4            auth              prod-sre     Non-critical (sheddable)
```

When the simulator says "SCALE_UP node-0 by 0.5", the system:
1. Looks up node-0 in the workload map -> `payments` in `prod-sre`
2. Calls `patch_namespaced_deployment_scale("payments", "prod-sre", ...)`
3. Kubernetes creates/destroys pods to match the new replica count

---

## What Runs Where (Complete List)

### On Hugging Face Spaces

| Component | What It Does | Port |
|---|---|---|
| FastAPI server (`server/app.py`) | HTTP API for the agent | 7860 (via NGINX) |
| Simulator (`simulator.py`) | 5-node microservice cluster simulation | Internal |
| PrometheusClient (`telemetry/prometheus_client.py`) | Queries AMP for real metrics | Outbound HTTPS |
| KubernetesExecutor (`control/kubernetes_executor.py`) | Sends scale commands to EKS | Outbound HTTPS |
| Prometheus metrics exporter | Serves `/metrics` for HF's monitoring | 8000 |
| Grafana + local Prometheus | Local dashboards (from the Dockerfile) | 3000, 9090 |

### On AWS EKS

| Component | Namespace | What It Does |
|---|---|---|
| payments Deployment | prod-sre | 2 nginx pods (scales with agent) |
| checkout Deployment | prod-sre | 1 nginx pod (scales with agent) |
| catalog Deployment | prod-sre | 1 nginx pod (scales with agent) |
| cart Deployment | prod-sre | 1 nginx pod (scales with agent) |
| auth Deployment | prod-sre | 1 nginx pod (scales with agent) |
| Prometheus Agent | monitoring | Scrapes workload pods, remote-writes to AMP |
| Cluster Autoscaler | kube-system | Adds/removes EC2 nodes based on demand |

### On AWS Managed Services

| Service | What It Does |
|---|---|
| AMP (Amazon Managed Prometheus) | Stores all metrics. Queried by HF Spaces. |
| AMG (Amazon Managed Grafana) | Visualizes metrics in dashboards. Accessed via browser. |

---

## The Simulator vs Real Cluster

AntiAtropos has three modes controlled by `ANTIATROPOS_ENV_MODE`:

### Simulated Mode (`simulated`)

Everything is fake. The simulator generates synthetic metrics (random CPU, latency, etc.). No K8s, no Prometheus. The agent practices in a safe sandbox.

This is the default on HF Spaces without AWS configured.

### Hybrid Mode (`hybrid`)

The simulator runs, but it pulls real metrics from AMP to calibrate itself. If AMP says `payments` pods have 80% CPU, the simulator adjusts its internal model to match. The agent can read real data but actions only affect the simulator, not real pods.

### Live Mode (`live`)

The real deal. The agent reads real metrics from AMP and sends real scale commands to EKS. When it says `SCALE_UP`, actual pods get created on actual EC2 instances that cost actual money.

**Set `ANTIATROPOS_ENV_MODE=live` on HF Spaces to enable this.**

---

## Cost Flow

Every pod on EKS costs money. Here is how costs flow based on the agent's actions:

```
Agent action: SCALE_UP node-0
  -> payments Deployment: replicas 2 -> 5
  -> 3 new pods created
  -> If existing nodes are full, Cluster Autoscaler adds a node
  -> New node = another t3.medium EC2 instance = ~$0.04/hr
  -> 3 pods running = 3 x (0.1 CPU + 64MB RAM) from the quota

Agent action: SCALE_DOWN node-3
  -> cart Deployment: replicas 4 -> 1
  -> 3 pods terminated
  -> If nodes are now underutilized, Cluster Autoscaler removes a node (after 10 min)
  -> One fewer EC2 instance = saves ~$0.04/hr
```

The Lyapunov reward function penalizes the agent for both instability AND cost, so a well-trained agent should learn to scale efficiently:

```
R_t = -(alpha * delta_V  +  beta * cost  +  gamma * SLA_violation)
                                  ^^^^
                           beta=0.01 penalizes over-provisioning
```

---

## Quick Reference: Key Files

| File | Purpose |
|---|---|
| `kubernetes_executor.py` | Translates agent actions to K8s API calls |
| `prometheus_client.py` | Queries AMP for real metrics |
| `simulator.py` | 5-node fluid-queue simulation |
| `stability.py` | Lyapunov reward computation |
| `deploy/aws/k8s-workloads.yaml` | The 5 Deployments + ResourceQuota on EKS |
| `deploy/aws/eksctl-cluster.yaml` | EKS cluster definition (nodes, caps) |
| `deploy/aws/prometheus-agent-values.yaml` | Helm config for Prometheus Agent |
| `deploy/aws/generate-kubeconfig.sh` | Creates kubeconfig for HF Spaces |
