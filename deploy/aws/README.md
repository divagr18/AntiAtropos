# AntiAtropos AWS Deployment Guide

Deploy the AWS infrastructure (EKS + AMP + AMG) that AntiAtropos on Hugging Face Spaces connects to.

## Architecture

```
Hugging Face Spaces                    AWS Region (us-east-1)
=====================                  ======================
                                       ┌─────────────────────────┐
                                       │ EKS Cluster             │
┌─────────────────┐                    │  ├── Workload pods      │
│ AntiAtropos     │  PROMETHEUS_URL    │  │   (payments, checkout │
│ FastAPI Server  │───────────────────>│  │    catalog, cart, auth)│
│ (port 7860)     │  (HTTPS + SigV4)   │  └── Prometheus Agent    │
│                 │                    │      (scrapes workloads, │
│                 │  KUBECONFIG        │       remote-writes to   │
│                 │───────────────────>│       AMP)               │
│                 │  (EKS API server)  └─────────────────────────┘
│                 │                    ┌─────────────────────────┐
│                 │                    │ Amazon Managed          │
│                 │                    │ Prometheus (AMP)        │
│                 │                    │  Workspace: antiatropos │
│                 │                    └─────────────────────────┘
│                 │                    ┌─────────────────────────┐
│                 │                    │ Amazon Managed Grafana  │
│                 │                    │  Dashboards: overview   │
│                 │                    │  + live                 │
└─────────────────┘                    └─────────────────────────┘
```

**Key principle: FastAPI runs on HF Spaces. AWS runs K8s workloads + AMP + AMG only.**

---

## Phase 0: Prerequisites

```bash
# AWS CLI v2
curl "https://awscli.amazonaws.com/AWSCLIV2.msi" -o "AWSCLIV2.msi"
msiexec /i AWSCLIV2.msi

# eksctl
choco install eksctl

# kubectl
choco install kubernetes-cli

# Helm
choco install kubernetes-helm

# Authenticate
aws configure
```

---

## Phase 1: Create the EKS Cluster (15 min)

```bash
eksctl create cluster -f deploy/aws/eksctl-cluster.yaml

# Verify
aws eks update-kubeconfig --name antiatropos --region us-east-1
kubectl get nodes
```

---

## Phase 2: Deploy Sample Workloads on EKS

These are the microservice deployments the SRE agent will scale up/down:

```bash
kubectl apply -f deploy/aws/k8s-workloads.yaml
```

This creates 5 deployments in the `prod-sre` namespace:
- `payments` (node-0, VIP) — 2 replicas
- `checkout` (node-1) — 1 replica
- `catalog` (node-2) — 1 replica
- `cart` (node-3) — 1 replica
- `auth` (node-4) — 1 replica

Verify:
```bash
kubectl get pods -n prod-sre
```

---

## Phase 3: Set Up Amazon Managed Prometheus (AMP)

### Create AMP Workspace

```bash
aws amp create-workspace \
  --alias antiatropos-metrics \
  --region us-east-1

# Note the workspace ID
aws amp list-workspaces --alias antiatropos-metrics --region us-east-1
```

### Set Up IRSA for Prometheus Agent

```bash
eksctl create iamserviceaccount \
  --cluster antiatropos \
  --namespace monitoring \
  --name prometheus-sa \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess \
  --approve \
  --override-existing-serviceaccounts
```

### Install Prometheus Agent on EKS

The agent scrapes workload pods and remote-writes metrics to AMP:

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Replace WORKSPACE_ID with your AMP workspace ID
helm install prometheus-agent prometheus-community/prometheus \
  --namespace monitoring --create-namespace \
  -f deploy/aws/prometheus-agent-values.yaml \
  --set prometheus.prometheusSpec.remoteWrite[0].url="https://aps-workspaces.us-east-1.amazonaws.com/workspaces/WORKSPACE_ID/api/v1/remote_write"
```

### Verify AMP is Receiving Data

```bash
pip install awscurl
awscurl --service aps "https://aps-workspaces.us-east-1.amazonaws.com/workspaces/WORKSPACE_ID/api/v1/query?query=up" --region us-east-1
```

---

## Phase 4: Set Up Amazon Managed Grafana (AMG)

### Create IAM Role for AMG

```bash
aws iam create-role \
  --role-name AntiAtroposGrafanaRole \
  --assume-role-policy-document file://deploy/aws/grafana-trust-policy.json

aws iam attach-role-policy \
  --role-name AntiAtroposGrafanaRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusQueryAccess

aws iam attach-role-policy \
  --role-name AntiAtroposGrafanaRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess
```

### Create AMG Workspace

```bash
aws grafana create-workspace \
  --workspace-name antiatropos-dashboards \
  --account-access-type CURRENT_ACCOUNT \
  --authentication-method AWS_SSO \
  --permission-type SERVICE_MANAGED \
  --data-sources PROMETHEUS \
  --region us-east-1

# Get the URL
aws grafana list-workspaces --region us-east-1 --query 'workspaces[0].endpoint' --output text
```

### Add AMP Data Source and Import Dashboards

1. Open the AMG workspace URL in your browser
2. Sign in with AWS SSO
3. Go to Configuration → Data Sources → Add Prometheus → select the `antiatropos-metrics` workspace
4. Go to Dashboards → Import → Upload JSON files from `deploy/grafana/provisioning/dashboards/json/`
5. When importing, select the AMP data source (not the default "Prometheus" with UID `PBFA97CFB590B2093`)

---

## Phase 5: Generate Kubeconfig for HF Spaces

The AntiAtropos server on HF Spaces needs a kubeconfig to talk to EKS:

```bash
./deploy/aws/generate-kubeconfig.sh
```

This outputs `deploy/aws/kubeconfig-antiatropos.yaml`. You'll set this as a secret on HF Spaces.

---

## Phase 6: Configure HF Spaces Environment Variables

Set these in your HF Space (Settings → Repository secrets and Variables):

### Secrets

| Secret | Value |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key |
| `KUBECONFIG_CONTENT` | Full content of `kubeconfig-antiatropos.yaml`, base64-encoded |

### Environment Variables

| Variable | Value |
|---|---|
| `ANTIATROPOS_ENV_MODE` | `live` |
| `ANTIATROPOS_STRICT_REAL` | `false` |
| `PROMETHEUS_URL` | `https://aps-workspaces.us-east-1.amazonaws.com/workspaces/WORKSPACE_ID` |
| `KUBECONFIG` | `/app/kubeconfig.yaml` |
| `ANTIATROPOS_K8S_NAMESPACE` | `prod-sre` |
| `ANTIATROPOS_MAX_REPLICAS` | `6` |
| `ANTIATROPOS_MIN_REPLICAS` | `1` |
| `ANTIATROPOS_SCALE_STEP` | `3` |
| `ANTIATROPOS_PROM_TIMEOUT_S` | `5.0` |
| `ANTIATROPOS_METRIC_AGGREGATION` | `sum` |
| `ANTIATROPOS_WORKLOAD_MAP` | See below |

### Workload Map

```json
{
  "node-0": {"deployment": "payments", "namespace": "prod-sre"},
  "node-1": {"deployment": "checkout", "namespace": "prod-sre"},
  "node-2": {"deployment": "catalog", "namespace": "prod-sre"},
  "node-3": {"deployment": "cart", "namespace": "prod-sre"},
  "node-4": {"deployment": "auth", "namespace": "prod-sre"}
}
```

### Entrypoint Addition

Add this to `deploy/entrypoint.sh` before starting uvicorn, so the kubeconfig is decoded from the HF secret:

```bash
# Decode kubeconfig from HF Spaces secret
if [ -n "${KUBECONFIG_CONTENT:-}" ]; then
    echo "${KUBECONFIG_CONTENT}" | base64 -d > /app/kubeconfig.yaml
    export KUBECONFIG=/app/kubeconfig.yaml
fi
```

---

## Phase 7: Install Cluster Autoscaler

So EKS can add nodes when the agent scales workloads:

```bash
helm repo add autoscaler https://kubernetes.github.io/autoscaler
helm repo update

helm install cluster-autoscaler autoscaler/cluster-autoscaler \
  --namespace kube-system \
  -f deploy/aws/cluster-autoscaler-values.yaml
```

The node group `maxSize: 4` in `eksctl-cluster.yaml` caps your compute cost.

---

## Cost Estimates

| Resource | Config | Monthly Cost (approx) |
|---|---|---|
| EKS Control Plane | 1 cluster | $73 |
| EKS Nodes | 2x t3.medium | $60 |
| AMP | <10GB ingest | ~$3-5 |
| AMG | 1 editor | Free tier or ~$9 |
| **Total** | | **~$135-150/month** |
| HF Spaces | Free tier or $5/mo | (separate billing) |

No ECR, no ALB, no server pods on AWS — cheaper than running everything on AWS.

### Cost-Saving Tips

- Use spot instances for node groups (60-70% cheaper)
- Scale workloads to zero between runs: `kubectl scale deployment -n prod-sre --replicas=0 --all`
- Delete the cluster between training runs: `eksctl delete cluster --name antiatropos`
- AMP free tier covers first 10GB ingest/month
- AMG free tier is 1 editor for 30 days

---

## Teardown

```bash
# Delete workloads
kubectl delete -f deploy/aws/k8s-workloads.yaml

# Delete Prometheus agent
helm uninstall prometheus-agent -n monitoring
kubectl delete namespace monitoring

# Delete AMP workspace
AMP_WS_ID=$(aws amp list-workspaces --alias antiatropos-metrics --region us-east-1 --query 'workspaces[0].workspaceId' --output text)
aws amp delete-workspace --workspace-id $AMP_WS_ID --region us-east-1

# Delete AMG workspace
AMG_WS_ID=$(aws grafana list-workspaces --region us-east-1 --query 'workspaces[0].id' --output text)
aws grafana delete-workspace --workspace-id $AMG_WS_ID

# Delete the EKS cluster (10-15 min)
eksctl delete cluster --name antiatropos --region us-east-1
```

---

## Troubleshooting

### HF Spaces can't reach AMP
- Verify `PROMETHEUS_URL` includes the full workspace path
- AMP requires SigV4 auth — ensure `requests-aws4auth` is in your dependencies
- Set `ANTIATROPOS_PROM_TIMEOUT_S=5.0` (cross-network latency)

### HF Spaces can't reach EKS
- Verify `KUBECONFIG` path and the file is decoded properly
- Check the EKS API server endpoint is public (default)
- Verify the IAM user in the kubeconfig has EKS access
- Test locally: `kubectl --kubeconfig=kubeconfig-antiatropos.yaml get nodes`

### AMP not receiving metrics
```bash
kubectl logs -n monitoring -l app.kubernetes.io/name=prometheus
```

### Grafana shows no data
1. Verify the AMG data source points to the correct AMP workspace
2. Check time range (AMP default retention is 30 days)
3. Verify PromQL queries match your metric names
