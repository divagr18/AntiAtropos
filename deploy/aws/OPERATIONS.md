# AntiAtropos AWS Operations Guide

Everything you need to run the AWS infrastructure for AntiAtropos without blowing up your bill.

**Architecture: FastAPI on Hugging Face Spaces, EKS + AMP + AMG on AWS.**

---

## Table of Contents

1. [Replica Strategy & Caps](#1-replica-strategy--caps)
2. [Autoscaling Configuration](#2-autoscaling-configuration)
3. [Cost Guardrails](#3-cost-guardrails)
4. [Step-by-Step Deployment Walkthrough](#4-step-by-step-deployment-walkthrough)
5. [Configuring HF Spaces to Connect to AWS](#5-configuring-hf-spaces-to-connect-to-aws)
6. [Day-2 Operations](#6-day-2-operations)
7. [Teardown & Cost Recovery](#7-teardown--cost-recovery)

---

## 1. Replica Strategy & Caps

### What Runs Where

| Component | Where | Scaled By | Cost Impact |
|---|---|---|---|
| **AntiAtropos FastAPI server** | HF Spaces | HF auto-scales | $0-5/month (HF billing) |
| **Workload pods** (payments, checkout, etc.) | EKS | SRE agent via `KubernetesExecutor` | **HIGH** — this is where costs spiral |
| **Prometheus Agent** | EKS (monitoring ns) | Static (1 pod) | Low |
| **AMP** | AWS managed | Serverless | Pay per GB ingested |
| **AMG** | AWS managed | Serverless | Pay per editor |

### Workload Pod Replicas — Where Costs Spiral

The SRE agent's `SCALE_UP` action calls `KubernetesExecutor._scale_deployment()`, which patches `replicas` on real K8s Deployments. A bad agent can scale every deployment to the cap.

The `ANTIATROPOS_MAX_REPLICAS` env var (set on HF Spaces) is the **global** ceiling applied to all deployments. The default in `kubernetes_executor.py` is 20 — with 5 deployments, that's **100 pods** worst case. **Set it to 6.**

**Recommended caps by deployment:**

| Deployment | Min | Max Replicas | Reasoning |
|---|---|---|---|
| `payments` (node-0, VIP) | 2 | 6 | VIP node — needs redundancy, 6 is plenty for the traffic model |
| `checkout` (node-1) | 1 | 5 | Can burst but shouldn't stay high |
| `catalog` (node-2) | 1 | 5 | Same |
| `cart` (node-3) | 1 | 4 | Non-critical, sheddable |
| `auth` (node-4) | 1 | 4 | Non-critical, sheddable |

**Total worst case: 24 workload pods.**

At ~0.25 vCPU / 256MB per workload pod (nginx containers), that's ~6 vCPU and ~6GB RAM — fits on 2x t3.medium nodes with some headroom, or 3 nodes for comfort.

### How the Cap Works

The `KubernetesExecutor._scale_deployment()` method reads `ANTIATROPOS_MAX_REPLICAS` from the environment and refuses to scale above it:

```
Ack: SCALE_UP for node-0 - replicas unchanged at 6 (bounds 1-6)
```

This is enforced in code (`kubernetes_executor.py` line 115):
```python
desired = min(self.max_replicas, current + delta)
```

**Set `ANTIATROPOS_MAX_REPLICAS=6` on your HF Space.**

---

## 2. Autoscaling Configuration

### EKS Node Autoscaling

The cluster needs to grow nodes when the agent scales workloads. Install the Cluster Autoscaler:

```bash
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm repo update

helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  -f deploy/aws/cluster-autoscaler-values.yaml
```

**The node group `maxSize` in `eksctl-cluster.yaml` (4) is your ultimate cost ceiling.**

```
4 nodes x $0.0416/hr (t3.medium on-demand) = $0.1664/hr = ~$120/month max
```

With spot instances, this drops to ~$36/month max.

### What Happens When the Agent Scales Workloads

1. Agent on HF Spaces sends `SCALE_UP` action
2. `KubernetesExecutor._scale_deployment()` patches the Deployment's `spec.replicas` via EKS API server
3. Kubernetes scheduler tries to place the new pod
4. If no node has capacity -> pod is `Pending`
5. Cluster Autoscaler sees `Pending` pods -> adds a node (within `maxSize`)
6. If `maxSize` is hit -> pod stays `Pending` (agent action succeeded but pod won't schedule)

**This is why `maxSize` in the node group is your ultimate cost ceiling.**

---

## 3. Cost Guardrails

### Monthly Cost Caps by Tier

| Tier | Max Nodes | Max Workload Pods | Estimated Monthly Cost |
|---|---|---|---|
| **Dev/Testing** | 2 | 10 (2/deployment) | ~$80 |
| **Training** | 3 | 15 (3/deployment) | ~$130 |
| **Benchmark Suite** | 4 | 24 (~5/deployment) | ~$160 |
| **Unlimited (danger)** | inf | 100 (20/deployment) | $500+ |

### AWS Budgets — Get Alerts Before You Overspend

```bash
aws budgets create-budget \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget '{
    "BudgetName": "AntiAtropos-Monthly",
    "BudgetLimit": {"Amount": "150", "Unit": "USD"},
    "TimeUnit": "MONTHLY",
    "CostFilters": {
      "TagKeyValue": ["user:Project$AntiAtropos"]
    },
    "CostTypes": {
      "IncludeTax": true,
      "IncludeSubscription": true,
      "UseBlended": false
    }
  }'

# Alert at 50%
aws budgets create-notification \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget-name "AntiAtropos-Monthly" \
  --notification '{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":50}' \
  --subscribers '[{"SubscriptionType":"EMAIL","Address":"your-email@example.com"}]'

# Alert at 80%
aws budgets create-notification \
  --account-id $(aws sts get-caller-identity --query Account --output text) \
  --budget-name "AntiAtropos-Monthly" \
  --notification '{"NotificationType":"ACTUAL","ComparisonOperator":"GREATER_THAN","Threshold":80}' \
  --subscribers '[{"SubscriptionType":"EMAIL","Address":"your-email@example.com"}]'
```

### Cost-Saving Checklist

- [ ] Use **spot instances** for node groups (60-70% cheaper, OK for training)
- [ ] Set `ANTIATROPOS_MAX_REPLICAS=6` on HF Spaces (not 20) to prevent agent runaway
- [ ] Cap node group `maxSize` at 4 (in `eksctl-cluster.yaml`)
- [ ] Set AWS Budget alert at $150/month
- [ ] Scale workloads to zero between runs: `kubectl scale deployment -n prod-sre --replicas=0 --all`
- [ ] Delete the cluster for multi-day breaks: `eksctl delete cluster --name antiatropos`
- [ ] AMP free tier covers first 10GB ingest/month
- [ ] AMG free tier is 1 editor for 30 days — cancel if not needed

---

## 4. Step-by-Step Deployment Walkthrough

### Before You Start

You need:
- AWS account with billing alerts enabled
- AWS CLI v2 installed and configured (`aws configure`)
- eksctl, kubectl, helm installed
- About 20-30 minutes

### Step 1: Create the EKS Cluster (15 min)

```bash
eksctl create cluster -f deploy/aws/eksctl-cluster.yaml

# Verify
aws eks update-kubeconfig --name antiatropos --region ap-south-1
kubectl get nodes
```

### Step 2: Deploy Sample Workloads (1 min)

```bash
kubectl apply -f deploy/aws/k8s-workloads.yaml
kubectl get pods -n prod-sre
```

### Step 3: Create AMP Workspace (1 min)

```bash
aws amp create-workspace --alias antiatropos-metrics --region ap-south-1

# Note the workspace ID
aws amp list-workspaces --alias antiatropos-metrics --region ap-south-1 --query 'workspaces[0].workspaceId' --output text
```

### Step 4: Set Up IRSA (2 min)

```bash
# Prometheus agent needs to write to AMP
eksctl create iamserviceaccount \
  --cluster antiatropos \
  --namespace monitoring \
  --name prometheus-sa \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess \
  --approve
```

### Step 5: Install Prometheus Agent (2 min)

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Replace WORKSPACE_ID
helm install prometheus-agent prometheus-community/prometheus \
  --namespace monitoring --create-namespace \
  -f deploy/aws/prometheus-agent-values.yaml \
  --set "prometheus.prometheusSpec.remoteWrite[0].url=https://aps-workspaces.ap-south-1.amazonaws.com/workspaces/WORKSPACE_ID/api/v1/remote_write"
```

### Step 6: Set Up AMG (5 min)

```bash
# Create IAM role for AMG
aws iam create-role \
  --role-name AntiAtroposGrafanaRole \
  --assume-role-policy-document file://deploy/aws/grafana-trust-policy.json

aws iam attach-role-policy \
  --role-name AntiAtroposGrafanaRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusQueryAccess

# Create workspace
aws grafana create-workspace \
  --workspace-name antiatropos-dashboards \
  --account-access-type CURRENT_ACCOUNT \
  --authentication-method AWS_SSO \
  --permission-type SERVICE_MANAGED \
  --data-sources PROMETHEUS \
  --region ap-south-1
```

Then in the AMG web UI:
1. Sign in with AWS SSO
2. Configuration -> Data Sources -> Add AMP workspace
3. Dashboards -> Import -> Upload JSON from `deploy/grafana/provisioning/dashboards/json/`
4. Select AMP data source when importing

### Step 7: Install Cluster Autoscaler (2 min)

```bash
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm repo update

helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  -f deploy/aws/cluster-autoscaler-values.yaml
```

### Step 8: Generate Kubeconfig for HF Spaces (1 min)

```bash
./deploy/aws/generate-kubeconfig.sh
# Outputs: deploy/aws/kubeconfig-antiatropos.yaml
```

### Step 9: Configure HF Spaces

See [Section 5](#5-configuring-hf-spaces-to-connect-to-aws) below.

---

## 5. Configuring HF Spaces to Connect to AWS

### Secrets (HF Space Settings -> Repository secrets)

| Secret | Value |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `KUBECONFIG_CONTENT` | Base64-encoded content of `kubeconfig-antiatropos.yaml` |

To encode the kubeconfig:
```bash
cat deploy/aws/kubeconfig-antiatropos.yaml | base64 -w 0
```

### Environment Variables (HF Space Settings -> Variables)

| Variable | Value |
|---|---|
| `ANTIATROPOS_ENV_MODE` | `live` |
| `ANTIATROPOS_STRICT_REAL` | `false` |
| `PROMETHEUS_URL` | `https://aps-workspaces.ap-south-1.amazonaws.com/workspaces/WORKSPACE_ID` |
| `KUBECONFIG` | `/app/kubeconfig.yaml` |
| `ANTIATROPOS_K8S_NAMESPACE` | `prod-sre` |
| `ANTIATROPOS_DEPLOYMENT_PREFIX` | `` (empty) |
| `ANTIATROPOS_MIN_REPLICAS` | `1` |
| `ANTIATROPOS_MAX_REPLICAS` | `6` |
| `ANTIATROPOS_SCALE_STEP` | `3` |
| `ANTIATROPOS_PROM_TIMEOUT_S` | `5.0` |
| `ANTIATROPOS_METRIC_AGGREGATION` | `sum` |
| `ANTIATROPOS_WORKLOAD_MAP` | See below |

### Workload Map Value

```json
{
  "node-0": {"deployment": "payments", "namespace": "prod-sre"},
  "node-1": {"deployment": "checkout", "namespace": "prod-sre"},
  "node-2": {"deployment": "catalog", "namespace": "prod-sre"},
  "node-3": {"deployment": "cart", "namespace": "prod-sre"},
  "node-4": {"deployment": "auth", "namespace": "prod-sre"}
}
```

### Entrypoint Modification

Add this to `deploy/entrypoint.sh` before the uvicorn line, so the kubeconfig is decoded from the HF secret:

```bash
# Decode kubeconfig from HF Spaces secret
if [ -n "${KUBECONFIG_CONTENT:-}" ]; then
    echo "${KUBECONFIG_CONTENT}" | base64 -d > /app/kubeconfig.yaml
    export KUBECONFIG=/app/kubeconfig.yaml
fi
```

### Verifying the Connection

After deploying, check from HF Spaces that the server can reach AWS:

1. Check the HF Space logs for `antiatropos_step` events
2. Look for `Ack: SCALE_UP` messages (agent is reaching EKS)
3. Look for non-zero `request_rate` / `cpu_utilization` (PrometheusClient is reaching AMP)
4. If `ANTIATROPOS_STRICT_REAL=false` (recommended), failures fall back to mock silently

---

## 6. Day-2 Operations

### Scaling Workloads Manually

```bash
# Scale a specific deployment
kubectl scale deployment/payments -n prod-sre --replicas=4

# Scale all workloads down
kubectl scale deployment -n prod-sre --replicas=0 --all

# Scale all workloads back up
kubectl scale deployment payments -n prod-sre --replicas=2
kubectl scale deployment checkout -n prod-sre --replicas=1
kubectl scale deployment catalog -n prod-sre --replicas=1
kubectl scale deployment cart -n prod-sre --replicas=1
kubectl scale deployment auth -n prod-sre --replicas=1
```

### Pausing Everything (Without Deleting)

```bash
# Scale all workloads to 0
kubectl scale deployment -n prod-sre --replicas=0 --all

# Note: EKS nodes still run and cost money.
# For real savings, delete the cluster (Section 7).
```

### Monitoring Agent Behavior

Watch what the SRE agent is doing in real-time:

```bash
# Check how many workload pods the agent has created
kubectl get deployments -n prod-sre

# Check current replica counts
kubectl get hpa -A  # if any HPAs are defined

# Check node pressure
kubectl top nodes
```

### Checking Current Spend

```bash
# Current month cost by service
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '1st of this month' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics BlendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

### Regenerating Kubeconfig

If the EKS cluster is recreated or credentials expire:

```bash
./deploy/aws/generate-kubeconfig.sh
# Re-upload the base64-encoded content to HF Spaces secret KUBECONFIG_CONTENT
```

---

## 7. Teardown & Cost Recovery

### Partial Teardown (Keep Cluster, Stop Workloads)

```bash
kubectl scale deployment -n prod-sre --replicas=0 --all
# Still paying for EKS control plane ($73/month) and idle nodes
```

### Full Teardown (Stop All Charges)

```bash
# Delete workloads
kubectl delete -f deploy/aws/k8s-workloads.yaml

# Delete Prometheus agent
helm uninstall prometheus-agent -n monitoring
kubectl delete namespace monitoring

# Delete AMP workspace
AMP_WS_ID=$(aws amp list-workspaces --alias antiatropos-metrics --region ap-south-1 --query 'workspaces[0].workspaceId' --output text)
aws amp delete-workspace --workspace-id $AMP_WS_ID --region ap-south-1

# Delete AMG workspace
AMG_WS_ID=$(aws grafana list-workspaces --region ap-south-1 --query 'workspaces[0].id' --output text)
aws grafana delete-workspace --workspace-id $AMG_WS_ID

# Delete IAM role for Grafana
aws iam detach-role-policy --role-name AntiAtroposGrafanaRole --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusQueryAccess
aws iam detach-role-policy --role-name AntiAtroposGrafanaRole --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess
aws iam delete-role --role-name AntiAtroposGrafanaRole

# Delete the EKS cluster (10-15 min)
eksctl delete cluster --name antiatropos --region ap-south-1

# Verify nothing is left
aws eks list-clusters --region ap-south-1
aws amp list-workspaces --region ap-south-1
```

Also remove the `KUBECONFIG_CONTENT` secret and reset `PROMETHEUS_URL` to `mock` in your HF Space.

---

## Quick Reference Card

| Task | Command |
|---|---|
| Deploy AWS infra | `./deploy/aws/deploy.sh` |
| Check workloads | `kubectl get pods -n prod-sre` |
| Check monitoring | `kubectl get pods -n monitoring` |
| Scale a workload | `kubectl scale deployment/payments -n prod-sre --replicas=N` |
| Pause all workloads | `kubectl scale deployment -n prod-sre --replicas=0 --all` |
| Check AMP data | `awscurl --service aps "https://aps-workspaces.ap-south-1.amazonaws.com/workspaces/WS_ID/api/v1/query?query=up" --region ap-south-1` |
| Generate kubeconfig | `./deploy/aws/generate-kubeconfig.sh` |
| Nuke everything | `eksctl delete cluster --name antiatropos --region ap-south-1` |

