#!/usr/bin/env bash
# AntiAtropos AWS Infrastructure Deploy Script
#
# Deploys: EKS cluster, sample workloads, AMP workspace, Prometheus Agent,
#          AMG workspace, Cluster Autoscaler, and generates kubeconfig for HF Spaces.
#
# The AntiAtropos FastAPI server runs on Hugging Face Spaces, NOT on AWS.
# This script only sets up the infrastructure that HF Spaces connects to.
#
# Prerequisites: aws cli, eksctl, kubectl, helm
#
# Usage:
#   chmod +x deploy/aws/deploy.sh
#   ./deploy/aws/deploy.sh
#
# Environment variables:
#   AWS_REGION     - AWS region (default: us-east-1)
#   CLUSTER_NAME   - EKS cluster name (default: antiatropos)

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-antiatropos}"
AWS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== AntiAtropos AWS Infrastructure Deployment ==="
echo "Region:      $REGION"
echo "Cluster:     $CLUSTER_NAME"
echo "FastAPI:     Runs on HF Spaces (not deployed here)"
echo ""

# --- Check prerequisites ---
for cmd in aws eksctl kubectl helm; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: $cmd is not installed. Please install it first."
        exit 1
    fi
done

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

# --- Phase 2: Deploy Sample Workloads ---
echo ""
echo ">>> Phase 2: Deploying sample workloads (payments, checkout, catalog, cart, auth)..."
kubectl apply -f "$AWS_DIR/k8s-workloads.yaml"
echo "Workloads deployed."
kubectl get pods -n prod-sre

# --- Phase 3: Create AMP Workspace ---
echo ""
echo ">>> Phase 3: Creating Amazon Managed Prometheus workspace..."
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

# --- Phase 4: Set up IRSA for Prometheus Agent ---
echo ""
echo ">>> Phase 4: Setting up IRSA for Prometheus Agent..."
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

# --- Phase 5: Install Prometheus Agent ---
echo ""
echo ">>> Phase 5: Installing Prometheus Agent (remote-writes to AMP)..."
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

# --- Phase 6: Create AMG Workspace ---
echo ""
echo ">>> Phase 6: Creating Amazon Managed Grafana workspace..."

# Create IAM role for Grafana if it doesn't exist
if aws iam get-role --role-name AntiAtroposGrafanaRole &>/dev/null; then
    echo "AntiAtroposGrafanaRole already exists."
else
    aws iam create-role \
        --role-name AntiAtroposGrafanaRole \
        --assume-role-policy-document file://"$AWS_DIR/grafana-trust-policy.json"
    aws iam attach-role-policy \
        --role-name AntiAtroposGrafanaRole \
        --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusQueryAccess
    aws iam attach-role-policy \
        --role-name AntiAtroposGrafanaRole \
        --policy-arn arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess
    echo "AntiAtroposGrafanaRole created."
fi

AMG_WS_ID=$(aws grafana list-workspaces --region "$REGION" --query 'workspaces[0].id' --output text 2>/dev/null || echo "")
if [ -z "$AMG_WS_ID" ] || [ "$AMG_WS_ID" = "None" ]; then
    AMG_WS_ID=$(aws grafana create-workspace \
        --workspace-name antiatropos-dashboards \
        --account-access-type CURRENT_ACCOUNT \
        --authentication-method AWS_SSO \
        --permission-type SERVICE_MANAGED \
        --data-sources PROMETHEUS \
        --region "$REGION" \
        --query 'workspaceId' \
        --output text)
    echo "AMG workspace created: $AMG_WS_ID"
else
    echo "AMG workspace already exists: $AMG_WS_ID"
fi

AMG_ENDPOINT=$(aws grafana list-workspaces --region "$REGION" --query 'workspaces[0].endpoint' --output text)
echo "AMG URL: https://$AMG_ENDPOINT"

# --- Phase 7: Install Cluster Autoscaler ---
echo ""
echo ">>> Phase 7: Installing Cluster Autoscaler..."
helm repo add autoscaler https://kubernetes.github.io/autoscaler 2>/dev/null || true
helm repo update

if helm status cluster-autoscaler -n kube-system &>/dev/null; then
    echo "cluster-autoscaler already installed, upgrading..."
    helm upgrade cluster-autoscaler autoscaler/cluster-autoscaler \
        --namespace kube-system \
        -f "$AWS_DIR/cluster-autoscaler-values.yaml"
else
    helm install cluster-autoscaler autoscaler/cluster-autoscaler \
        --namespace kube-system \
        -f "$AWS_DIR/cluster-autoscaler-values.yaml"
    echo "cluster-autoscaler installed."
fi

# --- Phase 8: Generate Kubeconfig for HF Spaces ---
echo ""
echo ">>> Phase 8: Generating kubeconfig for HF Spaces..."
"$AWS_DIR/generate-kubeconfig.sh"

# --- Done ---
echo ""
echo "=========================================="
echo "   AntiAtropos AWS Infrastructure Ready!"
echo "=========================================="
echo ""
echo "AMP Workspace ID:  $AMP_WS_ID"
echo "AMP URL:           $AMP_URL"
echo "AMG Workspace ID:  $AMG_WS_ID"
echo "AMG URL:           https://$AMG_ENDPOINT"
echo ""
echo "Kubeconfig saved:  $AWS_DIR/kubeconfig-antiatropos.yaml"
echo ""
echo "Next steps — configure your HF Space:"
echo "  1. Set secret KUBECONFIG_CONTENT = base64 of kubeconfig-antiatropos.yaml"
echo "  2. Set env var PROMETHEUS_URL = $AMP_URL"
echo "  3. Set env var KUBECONFIG = /app/kubeconfig.yaml"
echo "  4. Set env var ANTIATROPOS_ENV_MODE = live"
echo "  5. Set env var ANTIATROPOS_MAX_REPLICAS = 6"
echo "  6. Set env var ANTIATROPOS_WORKLOAD_MAP = (see OPERATIONS.md)"
echo "  7. Add kubeconfig decode to deploy/entrypoint.sh (see OPERATIONS.md)"
echo ""
echo "In AMG, add AMP as a data source and import dashboards from:"
echo "  deploy/grafana/provisioning/dashboards/json/"
