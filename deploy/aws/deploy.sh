#!/usr/bin/env bash
# AntiAtropos AWS Quick Deploy Script
# 
# Prerequisites: aws cli, eksctl, kubectl, helm, docker
# 
# Usage:
#   chmod +x deploy/aws/deploy.sh
#   ./deploy/aws/deploy.sh
#
# This script creates all AWS resources and deploys AntiAtropos to EKS.
# Set these environment variables before running:
#   OPENAI_API_KEY     - Your OpenAI API key (required)
#   AWS_REGION         - AWS region (default: us-east-1)
#   CLUSTER_NAME       - EKS cluster name (default: antiatropos)

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-antiatropos}"
AWS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== AntiAtropos AWS Deployment ==="
echo "Region:      $REGION"
echo "Cluster:     $CLUSTER_NAME"
echo ""

# --- Check prerequisites ---
for cmd in aws eksctl kubectl helm docker; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd is not installed. Please install it first."
        exit 1
    fi
done

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set."
    exit 1
fi

# --- Phase 1: Create EKS Cluster ---
echo ""
echo ">>> Phase 1: Creating EKS cluster..."
if eksctl get cluster --name "$CLUSTER_NAME" --region "$REGION" &>/dev/null; then
    echo "Cluster $CLUSTER_NAME already exists, skipping creation."
else
    eksctl create cluster -f "$AWS_DIR/eksctl-cluster.yaml"
    echo "Cluster created."
fi

aws eks update-kubeconfig --name "$CLUSTER_NAME" --region "$REGION"
echo "kubeconfig updated."

# --- Phase 2: Create AMP Workspace ---
echo ""
echo ">>> Phase 2: Creating Amazon Managed Prometheus workspace..."
AMP_WS_ID=$(aws amp list-workspaces --alias antiatropos-metrics --region "$REGION" --query 'workspaces[0].workspaceId' --output text 2>/dev/null || echo "")

if [ -z "$AMP_WS_ID" ] || [ "$AMP_WS_ID" = "None" ]; then
    AMP_WS_ID=$(aws amp create-workspace \
        --alias antiatropos-metrics \
        --region "$REGION" \
        --query 'workspaceId' \
        --output text)
    echo "AMP workspace created: $AMP_WS_ID"
else
    echo "AMP workspace already exists: $AMP_WS_ID"
fi

AMP_URL="https://aps-workspaces.$REGION.amazonaws.com/workspaces/$AMP_WS_ID"
echo "AMP URL: $AMP_URL"

# --- Phase 3: Set up IAM Roles for Service Accounts (IRSA) ---
echo ""
echo ">>> Phase 3: Setting up IRSA..."
CLUSTER_OIDC=$(aws eks describe-cluster --name "$CLUSTER_NAME" --region "$REGION" --query 'cluster.identity.oidc.issuer' --output text | sed 's|https://||')
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Prometheus service account
if kubectl get serviceaccount prometheus-sa -n monitoring &>/dev/null; then
    echo "prometheus-sa already exists."
else
    eksctl create iamserviceaccount \
        --cluster "$CLUSTER_NAME" \
        --namespace monitoring \
        --name prometheus-sa \
        --attach-policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess \
        --approve \
        --override-existing-serviceaccounts
    echo "prometheus-sa created."
fi

# AntiAtropos service account
if kubectl get serviceaccount antiatropos-sa -n antiatropos &>/dev/null; then
    echo "antiatropos-sa already exists."
else
    eksctl create iamserviceaccount \
        --cluster "$CLUSTER_NAME" \
        --namespace antiatropos \
        --name antiatropos-sa \
        --attach-policy-arn arn:aws:iam::aws:policy/AmazonPrometheusQueryAccess \
        --approve \
        --override-existing-serviceaccounts
    echo "antiatropos-sa created."
fi

# --- Phase 4: Install Prometheus Agent ---
echo ""
echo ">>> Phase 4: Installing Prometheus Agent (remote-writes to AMP)..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update

if helm status prometheus-agent -n monitoring &>/dev/null; then
    echo "prometheus-agent already installed, upgrading..."
    helm upgrade prometheus-agent prometheus-community/prometheus \
        --namespace monitoring \
        -f "$AWS_DIR/prometheus-agent-values.yaml" \
        --set "prometheus.prometheusSpec.remoteWrite[0].url=$AMP_URL/api/v1/remote_write"
else
    helm install prometheus-agent prometheus-community/prometheus \
        --namespace monitoring --create-namespace \
        -f "$AWS_DIR/prometheus-agent-values.yaml" \
        --set "prometheus.prometheusSpec.remoteWrite[0].url=$AMP_URL/api/v1/remote_write"
    echo "prometheus-agent installed."
fi

# --- Phase 5: Build and Push Docker Image ---
echo ""
echo ">>> Phase 5: Building and pushing Docker image to ECR..."
ECR_URI="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/antiatropos"

if aws ecr describe-repositories --repository-name antiatropos --region "$REGION" &>/dev/null; then
    echo "ECR repository already exists."
else
    aws ecr create-repository --repository-name antiatropos --region "$REGION"
    echo "ECR repository created."
fi

aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin \
    "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

docker build -t antiatropos:latest "$AWS_DIR/../.."
docker tag antiatropos:latest "$ECR_URI:latest"
docker push "$ECR_URI:latest"
echo "Image pushed to $ECR_URI:latest"

# --- Phase 6: Deploy AntiAtropos ---
echo ""
echo ">>> Phase 6: Deploying AntiAtropos to EKS..."

# Apply namespace first
kubectl apply -f "$AWS_DIR/k8s-namespace.yaml"

# Create/update configmap with the real AMP URL
kubectl create configmap antiatropos-config \
    --from-literal=ANTIATROPOS_ENV_MODE=live \
    --from-literal=ANTIATROPOS_STRICT_REAL=false \
    --from-literal="PROMETHEUS_URL=$AMP_URL" \
    --from-literal=KUBECONFIG="" \
    --from-literal=ANTIATROPOS_K8S_NAMESPACE=default \
    --from-literal=ANTIATROPOS_DEPLOYMENT_PREFIX="" \
    --from-literal=ANTIATROPOS_MIN_REPLICAS=1 \
    --from-literal=ANTIATROPOS_MAX_REPLICAS=20 \
    --from-literal=ANTIATROPOS_SCALE_STEP=3 \
    --from-literal=ANTIATROPOS_WORKLOAD_MAP='{}' \
    --from-literal=ANTIATROPOS_NODE_DEPLOYMENT_MAP='{}' \
    --from-literal=ANTIATROPOS_PROM_TIMEOUT_S=5.0 \
    --from-literal=ANTIATROPOS_METRIC_AGGREGATION=sum \
    -n antiatropos \
    --dry-run=client -o yaml | kubectl apply -f -

# Create secret with OpenAI API key
kubectl create secret generic antiatropos-secrets \
    --from-literal=openai-api-key="$OPENAI_API_KEY" \
    -n antiatropos \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply deployment with the correct ECR image
sed "s|ACCOUNT_ID|$ACCOUNT_ID|g; s|us-east-1|$REGION|g" "$AWS_DIR/k8s-deployment.yaml" | kubectl apply -f -
kubectl apply -f "$AWS_DIR/k8s-service.yaml"

# Wait for rollout
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/antiatropos -n antiatropos --timeout=300s

# --- Done ---
echo ""
echo "=========================================="
echo "   AntiAtropos AWS Deployment Complete!"
echo "=========================================="
echo ""
echo "AMP Workspace ID: $AMP_WS_ID"
echo "AMP URL:          $AMP_URL"
echo ""
echo "Next steps:"
echo "  1. Create an Amazon Managed Grafana workspace:"
echo "     aws grafana create-workspace --workspace-name antiatropos-dashboards \\"
echo "       --account-access-type CURRENT_ACCOUNT --authentication-method AWS_SSO \\"
echo "       --permission-type SERVICE_MANAGED --data-sources PROMETHEUS --region $REGION"
echo ""
echo "  2. Add AMP as a data source in AMG and import dashboards from:"
echo "     deploy/grafana/provisioning/dashboards/json/"
echo ""
echo "  3. Get the AntiAtropos service URL:"
echo "     kubectl get svc -n antiatropos antiatropos"
echo ""
echo "  4. Port-forward for local testing:"
echo "     kubectl port-forward -n antiatropos deployment/antiatropos 8000:8000"
