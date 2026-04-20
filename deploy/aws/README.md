# AntiAtropos AWS Deployment Guide

Complete guide for deploying AntiAtropos on AWS with EKS, Amazon Managed Prometheus (AMP), and Amazon Managed Grafana (AMG).

## Architecture

```
AWS Region (us-east-1)
├── EKS Cluster
│   ├── AntiAtropos FastAPI pod
│   ├── Prometheus Agent pod (remote-writes to AMP)
│   └── Sample workload pods (optional, for live mode)
├── Amazon Managed Prometheus (AMP)
│   └── Workspace: antiatropos-metrics
├── Amazon Managed Grafana (AMG)
│   └── Workspace: antiatropos-dashboards
├── ALB (Application Load Balancer)
│   └── / → FastAPI, /grafana → AMG
└── ECR (Container Registry)
    └── antiatropos:latest
```

---

## Phase 0: Prerequisites

```bash
# Install CLI tools (if not already installed)
# AWS CLI v2
curl "https://awscli.amazonaws.com/AWSCLIV2.msi" -o "AWSCLIV2.msi"
msiexec /i AWSCLIV2.msi

# eksctl (EKS management)
choco install eksctl   # or: winget install --id=FluxCD.eksctl

# kubectl
choco install kubernetes-cli

# Helm
choco install kubernetes-helm

# Authenticate AWS
aws configure
# Enter: Access Key ID, Secret Access Key, Region (us-east-1), Output (json)
```

---

## Phase 1: Create the EKS Cluster

### Option A: eksctl (recommended, fastest)

Create file `deploy/aws/eksctl-cluster.yaml` then run:

```bash
eksctl create cluster -f deploy/aws/eksctl-cluster.yaml
```

### Option B: AWS Console

1. Go to EKS → Create Cluster
2. Name: `antiatropos`, Kubernetes 1.30
3. Cluster service role: Create new (let EKS create it)
4. Networking: Default VPC, all AZs
5. Add node group: `linux-nodes`, t3.medium, 2-4 nodes
6. Create and wait ~15 minutes

### Verify

```bash
aws eks update-kubeconfig --name antiatropos --region us-east-1
kubectl get nodes
```

---

## Phase 2: Set Up Amazon Managed Prometheus (AMP)

### Create AMP Workspace

```bash
aws amp create-workspace \
  --alias antiatropos-metrics \
  --region us-east-1

# Note the workspace ARN and ID from the output
aws amp list-workspaces --alias antiatropos-metrics --region us-east-1
```

### Install Prometheus Agent on EKS (remote-writes to AMP)

```bash
# Add the Prometheus Community Helm repo
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install prometheus agent with AMP remote write
# Replace WORKSPACE_ID with your AMP workspace ID
helm install prometheus-agent prometheus-community/prometheus \
  --namespace monitoring --create-namespace \
  -f deploy/aws/prometheus-agent-values.yaml \
  --set prometheus.prometheusSpec.remoteWrite[0].url="https://aps-workspaces.us-east-1.amazonaws.com/workspaces/WORKSPACE_ID/api/v1/remote_write"
```

### Verify AMP is Receiving Data

```bash
# Port-forward to query AMP directly
aws amp query-status --workspace-id WORKSPACE_ID --region us-east-1

# Or use awscurl for instant queries
pip install awscurl
awscurl --service aps "https://aps-workspaces.us-east-1.amazonaws.com/workspaces/WORKSPACE_ID/api/v1/query?query=up" --region us-east-1
```

---

## Phase 3: Set Up Amazon Managed Grafana (AMG)

### Create AMG Workspace

```bash
# First, create the IAM role for Grafana (allows it to read AMP)
aws iam create-role \
  --role-name AntiAtroposGrafanaRole \
  --assume-role-policy-document file://deploy/aws/grafana-trust-policy.json

aws iam attach-role-policy \
  --role-name AntiAtroposGrafanaRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusQueryAccess

aws iam attach-role-policy \
  --role-name AntiAtroposGrafanaRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess

# Create the Grafana workspace
aws grafana create-workspace \
  --workspace-name antiatropos-dashboards \
  --account-access-type CURRENT_ACCOUNT \
  --authentication-method AWS_SSO \
  --permission-type SERVICE_MANAGED \
  --data-sources PROMETHEUS \
  --region us-east-1

# Note the workspace URL from the output
aws grafana list-workspaces --region us-east-1
```

### Add AMP as a Data Source in AMG

1. Open the AMG workspace URL in your browser
2. Sign in with AWS SSO
3. Go to Configuration → Data Sources
4. AMP should auto-discover if in same account/region
5. Select the `antiatropos-metrics` workspace

### Import AntiAtropos Dashboards

```bash
# Use the Grafana API to import dashboards
# Replace GRAFANA_URL and API_KEY
GRAFANA_URL="https://YOUR-WORKSPACE-id.grafana.us-east-1.amazonaws.com"
API_KEY="YOUR-API-KEY"

# Import the overview dashboard
curl -X POST "$GRAFANA_URL/api/dashboards/db" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d @deploy/aws/grafana-dashboard-overview.json

# Import the live dashboard
curl -X POST "$GRAFANA_URL/api/dashboards/db" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d @deploy/aws/grafana-dashboard-live.json
```

---

## Phase 4: Build and Push the Docker Image

```bash
# Create ECR repository
aws ecr create-repository \
  --repository-name antiatropos \
  --region us-east-1

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  $(aws sts get-caller-identity --query Account --output text).dkr.ecr.us-east-1.amazonaws.com

# Build and push
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI=$ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/antiatropos

docker build -t antiatropos:latest .
docker tag antiatropos:latest $ECR_URI:latest
docker push $ECR_URI:latest
```

---

## Phase 5: Deploy AntiAtropos to EKS

```bash
# Apply Kubernetes manifests
kubectl apply -f deploy/aws/k8s-namespace.yaml
kubectl apply -f deploy/aws/k8s-configmap.yaml
kubectl apply -f deploy/aws/k8s-deployment.yaml
kubectl apply -f deploy/aws/k8s-service.yaml

# If using AWS Load Balancer Controller (recommended)
kubectl apply -f deploy/aws/k8s-ingress.yaml

# Check rollout
kubectl rollout status deployment/antiatropos -n antiatropos
kubectl get pods -n antiatropos
kubectl logs -f deployment/antiatropos -n antiatropos
```

### Environment Variables for Live Mode

The deployment manifest sets these to connect AntiAtropos to real infrastructure:

```yaml
env:
  - name: ANTIATROPOS_ENV_MODE
    value: "live"
  - name: PROMETHEUS_URL
    value: "https://aps-workspaces.us-east-1.amazonaws.com/workspaces/WORKSPACE_ID"
  - name: KUBECONFIG
    value: ""  # Empty = use in-cluster config
  - name: ANTIATROPOS_WORKLOAD_MAP
    value: '{"node-0":{"deployment":"payments","namespace":"prod-sre"},"node-1":{"deployment":"checkout","namespace":"prod-sre"}}'
```

---

## Phase 6: Access Your Deployment

### Get the ALB URL

```bash
kubectl get ingress -n antiatropos
# Copy the ADDRESS column
```

### Endpoints

| Endpoint | URL |
|---|---|
| Landing Page | `http://ALB_ADDRESS/` |
| API Health | `http://ALB_ADDRESS/health` |
| Prometheus Metrics | `http://ALB_ADDRESS/metrics` |
| Grafana Dashboards | AMG workspace URL (separate) |

### Port-Forward for Local Debugging

```bash
# FastAPI
kubectl port-forward -n antiatropos deployment/antiatropos 8000:8000

# Direct pod metrics
curl http://localhost:8000/metrics
```

---

## Phase 7: IRSA (IAM Roles for Service Accounts)

This lets the AntiAtropos pod authenticate with AMP without hardcoded credentials.

```bash
# Create OIDC provider for the EKS cluster
eksctl utils associate-iam-oidc-provider \
  --cluster antiatropos --region us-east-1 --approve

# Create IAM role for the AntiAtropos service account
eksctl create iamserviceaccount \
  --cluster antiatropos \
  --namespace antiatropos \
  --name antiatropos-sa \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonPrometheusQueryAccess \
  --attach-policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess \
  --approve \
  --override-existing-serviceaccounts

# Redeploy to pick up the new service account
kubectl rollout restart deployment/antiatropos -n antiatropos
```

---

## Cost Estimates

| Resource | Config | Monthly Cost (approx) |
|---|---|---|
| EKS Control Plane | 1 cluster | $73 |
| EKS Nodes | 2x t3.medium | $60 |
| AMP | <10GB ingest | ~$3-5 |
| AMG | 1 editor + viewers | Free tier or ~$9 |
| ALB | 1 load balancer | $16 |
| ECR | <1GB storage | <$1 |
| **Total** | | **~$150-160/month** |

### Cost-Saving Tips

- Use `t3.spot` for node groups (60-70% cheaper)
- Scale nodes to 0 when not training: `kubectl cordon` + drain
- Use Fargate profiles for the AntiAtropos pod (pay-per-pod-second)
- Delete the cluster between training runs with `eksctl delete cluster`

---

## Teardown

```bash
# Delete everything in reverse order
kubectl delete -f deploy/aws/k8s-ingress.yaml
kubectl delete -f deploy/aws/k8s-service.yaml
kubectl delete -f deploy/aws/k8s-deployment.yaml
kubectl delete -f deploy/aws/k8s-configmap.yaml
kubectl delete -f deploy/aws/k8s-namespace.yaml

aws grafana delete-workspace --workspace-id AMG_WORKSPACE_ID
aws amp delete-workspace --workspace-id AMP_WORKSPACE_ID
aws ecr delete-repository --repository-name antiatropos --force

eksctl delete cluster --name antiatropos --region us-east-1
```

---

## Troubleshooting

### Pods not starting
```bash
kubectl describe pod -n antiatropos -l app=antiatropos
kubectl logs -n antiatropos -l app=antiatropos --previous
```

### AMP not receiving metrics
```bash
# Check the prometheus agent logs
kubectl logs -n monitoring -l app.kubernetes.io/name=prometheus

# Verify remote-write endpoint
aws amp describe-workspace --workspace-id WORKSPACE_ID
```

### Can't reach AMP from pod
```bash
# Verify IRSA is attached
kubectl get pod -n antiatropos -o yaml | grep -A5 serviceAccount

# Check pod can reach AMP
kubectl exec -n antiatropos deployment/antiatropos -- \
  curl -s "https://aps-workspaces.us-east-1.amazonaws.com/workspaces/WORKSPACE_ID/api/v1/query?query=up"
```

### Grafana dashboard shows no data
1. Verify the data source URL in AMG points to the correct AMP workspace
2. Check time range (AMP has a retention period; default 30 days)
3. Verify the PromQL queries in dashboards match your metric names
