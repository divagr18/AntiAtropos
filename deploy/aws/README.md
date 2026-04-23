# AntiAtropos AWS Deployment Guide

Deploy the AWS infrastructure (EKS + AMP) that AntiAtropos on Hugging Face Spaces connects to.

For FastAPI wiring with `aws` mode and laptop Grafana, see [deploy/aws/FASTAPI_AWS_MODE_GUIDE.md](deploy/aws/FASTAPI_AWS_MODE_GUIDE.md).

## Architecture

```
Hugging Face Spaces                    AWS Region (ap-south-1)
=====================                  ======================
                                       ┌─────────────────────────┐
                                       │ EKS Cluster             │
┌─────────────────┐                    │  ├── Workload pods      │
│ AntiAtropos     │  PROMETHEUS_URL    │  │   (payments, checkout │
│ FastAPI Server  │───────────────────>│  │    catalog, cart, auth)│
│ (port 7860)     │  (HTTPS + SigV4)   │  ├── Prometheus Agent    │
│                 │                    │  │   (scrapes workloads, │
│                 │  KUBECONFIG        │  │    remote-writes AMP) │
│                 │───────────────────>│  ├── Grafana            │
│                 │  (EKS API server)  │  │   (self-hosted,       │
│                 │                    │  │    dashboards)        │
│                 │                    │  └── Monitoring ns       │
│                 │                    └─────────────────────────┘
│                 │                    ┌─────────────────────────┐
│                 │                    │ Amazon Managed          │
│                 │                    │ Prometheus (AMP)        │
│                 │                    │  Workspace: antiatropos │
│                 │                    └─────────────────────────┘
└─────────────────┘
```

**Key principle: FastAPI runs on HF Spaces. AWS runs K8s workloads + AMP + self-hosted Grafana.**

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
aws eks update-kubeconfig --name antiatropos --region ap-south-1
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
  --region ap-south-1

# Note the workspace ID
aws amp list-workspaces --alias antiatropos-metrics --region ap-south-1
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
  --set prometheus.prometheusSpec.remoteWrite[0].url="https://aps-workspaces.ap-south-1.amazonaws.com/workspaces/WORKSPACE_ID/api/v1/remote_write"
```

### Verify AMP is Receiving Data

```bash
pip install awscurl
awscurl --service aps "https://aps-workspaces.ap-south-1.amazonaws.com/workspaces/WORKSPACE_ID/api/v1/query?query=up" --region ap-south-1
```

---

## Phase 4 (Optional): Set Up Self-Hosted Grafana on EKS

If you are on free-tier nodes, skip this section and run Grafana locally on your laptop.

### Install Grafana

```bash
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm install grafana grafana/grafana \
  --namespace monitoring \
  -f deploy/aws/grafana-values.yaml
```

### Create Dashboard Secret

```bash
kubectl create secret generic antiatropos-grafana-dashboards \
  --from-file=antiatropos-overview.json=deploy/grafana/provisioning/dashboards/json/antiatropos-overview.json \
  --from-file=antiatropos-live.json=deploy/grafana/provisioning/dashboards/json/antiatropos-live.json \
  --namespace monitoring \
  --dry-run=client -o yaml | kubectl apply -f -
```

### Access Grafana

```bash
kubectl port-forward svc/grafana 3000 -n monitoring
```

Open `http://localhost:3000` in your browser:
- Username: `admin`
- Password: `antiatropos`

The data source `AMP-Local` is pre-configured to use the local Prometheus agent, and dashboards are auto-imported from the secret.

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
| `ANTIATROPOS_ENV_MODE` | `aws` |
| `ANTIATROPOS_STRICT_REAL` | `false` |
| `PROMETHEUS_URL` | `https://aps-workspaces.ap-south-1.amazonaws.com/workspaces/WORKSPACE_ID` |
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

### FastAPI Reset Mode

Use `mode="aws"` on environment reset for AWS-backed execution. If omitted, the server will use `ANTIATROPOS_ENV_MODE`.

---

## Local Grafana (Recommended on Free Tier)

Grafana is only for observability dashboards. Agent action execution stays in FastAPI + Kubernetes executor.

Start Grafana locally:

```bash
docker run -d --name antiatropos-grafana -p 3000:3000 grafana/grafana:latest
```

Then in Grafana:

1. Add Prometheus datasource using AMP workspace URL:
  - `https://aps-workspaces.<region>.amazonaws.com/workspaces/<WORKSPACE_ID>`
2. Enable SigV4 auth and set the same AWS region.
3. Import dashboards:
  - [deploy/grafana/provisioning/dashboards/json/antiatropos-overview.json](deploy/grafana/provisioning/dashboards/json/antiatropos-overview.json)
  - [deploy/grafana/provisioning/dashboards/json/antiatropos-live.json](deploy/grafana/provisioning/dashboards/json/antiatropos-live.json)

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
| EBS Volume (Grafana) | 5Gi | ~$0.50 |
| **Total** | | **~$135-145/month** |
| HF Spaces | Free tier or $5/mo | (separate billing) |

No ECR, no ALB, no server pods on AWS — cheaper than running everything on AWS.

### Cost-Saving Tips

- Use spot instances for node groups (60-70% cheaper)
- Scale workloads to zero between runs: `kubectl scale deployment -n prod-sre --replicas=0 --all`
- Delete the cluster between training runs: `eksctl delete cluster --name antiatropos`
- AMP free tier covers first 10GB ingest/month
- Grafana is self-hosted (free, runs on EKS)

---

## Teardown

```bash
# Delete workloads
kubectl delete -f deploy/aws/k8s-workloads.yaml

# Delete Grafana
helm uninstall grafana -n monitoring

# Delete Prometheus agent
helm uninstall prometheus-agent -n monitoring
kubectl delete namespace monitoring

# Delete dashboard secret
kubectl delete secret antiatropos-grafana-dashboards -n monitoring 2>/dev/null || true

# Delete AMP workspace
AMP_WS_ID=$(aws amp list-workspaces --alias antiatropos-metrics --region ap-south-1 --query 'workspaces[0].workspaceId' --output text)
aws amp delete-workspace --workspace-id $AMP_WS_ID --region ap-south-1

# Delete the EKS cluster (10-15 min)
eksctl delete cluster --name antiatropos --region ap-south-1
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
1. Verify the `AMP-Local` data source is configured: `http://prometheus-agent-server.monitoring.svc.cluster.local:80`
2. Check time range (AMP default retention is 30 days)
3. Verify PromQL queries match your metric names
4. Check Grafana logs: `kubectl logs -n monitoring -l app.kubernetes.io/name=grafana`
5. Verify dashboards secret exists: `kubectl get secret antiatropos-grafana-dashboards -n monitoring`

