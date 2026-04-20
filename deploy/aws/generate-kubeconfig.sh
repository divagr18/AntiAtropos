#!/usr/bin/env bash
# Generate a kubeconfig for HF Spaces to connect to the EKS cluster.
#
# This creates a kubeconfig that uses AWS IAM authenticator,
# which works from outside the cluster (like from HF Spaces).
#
# Prerequisites:
#   - aws cli
#   - kubectl
#   - eksctl
#   - The EKS cluster must already exist
#
# Usage:
#   ./generate-kubeconfig.sh
#
# Output:
#   deploy/aws/kubeconfig-antiatropos.yaml
#
# Then on HF Spaces:
#   1. base64 encode: cat kubeconfig-antiatropos.yaml | base64 -w 0
#   2. Set as HF Space secret: KUBECONFIG_CONTENT = <base64 output>
#   3. Set env var: KUBECONFIG = /app/kubeconfig.yaml
#   4. Add to deploy/entrypoint.sh:
#        if [ -n "${KUBECONFIG_CONTENT:-}" ]; then
#            echo "${KUBECONFIG_CONTENT}" | base64 -d > /app/kubeconfig.yaml
#            export KUBECONFIG=/app/kubeconfig.yaml
#        fi

set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME="${CLUSTER_NAME:-antiatropos}"
AWS_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT="$AWS_DIR/kubeconfig-antiatropos.yaml"

echo "=== Generating kubeconfig for HF Spaces ==="
echo "Cluster: $CLUSTER_NAME"
echo "Region:  $REGION"
echo ""

# Verify cluster exists
if ! eksctl get cluster --name "$CLUSTER_NAME" --region "$REGION" &>/dev/null; then
    echo "ERROR: Cluster $CLUSTER_NAME not found. Create it first with eksctl."
    exit 1
fi

# Get cluster details
CLUSTER_ENDPOINT=$(aws eks describe-cluster \
    --name "$CLUSTER_NAME" \
    --region "$REGION" \
    --query 'cluster.endpoint' \
    --output text)

CLUSTER_CA=$(aws eks describe-cluster \
    --name "$CLUSTER_NAME" \
    --region "$REGION" \
    --query 'cluster.certificateAuthority.data' \
    --output text)

# Get the current AWS identity for the kubeconfig
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_ARN=$(aws sts get-caller-identity --query Arn --output text)

echo "Cluster endpoint: $CLUSTER_ENDPOINT"
echo "AWS identity:     $AWS_ARN"
echo ""

# Generate the kubeconfig
cat > "$OUTPUT" <<EOF
# Kubeconfig for AntiAtropos on Hugging Face Spaces
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# Cluster:   $CLUSTER_NAME
# Region:    $REGION
#
# This kubeconfig uses AWS IAM authenticator.
# The HF Space container must have aws-cli and aws-iam-authenticator available,
# OR the kubernetes Python client must be configured with AWS credentials.
#
# To use this on HF Spaces:
#   1. base64 encode this file: cat kubeconfig-antiatropos.yaml | base64 -w 0
#   2. Set as HF secret: KUBECONFIG_CONTENT = <base64>
#   3. Set env var: KUBECONFIG = /app/kubeconfig.yaml
#   4. Decode in entrypoint.sh before uvicorn starts

apiVersion: v1
kind: Config
clusters:
  - cluster:
      certificate-authority-data: $CLUSTER_CA
      server: $CLUSTER_ENDPOINT
    name: $CLUSTER_NAME

contexts:
  - context:
      cluster: $CLUSTER_NAME
      user: antiatropos-hf-user
    name: $CLUSTER_NAME

current-context: $CLUSTER_NAME

preferences: {}

users:
  - name: antiatropos-hf-user
    user:
      exec:
        apiVersion: client.authentication.k8s.io/v1beta1
        command: aws
        args:
          - eks
          - token
          - --region
          - $REGION
          - --cluster-name
          - $CLUSTER_NAME
        env:
          - name: AWS_STS_REGIONAL_ENDPOINTS
            value: regional
          - name: AWS_DEFAULT_REGION
            value: $REGION
        interactiveMode: IfAvailable
EOF

echo "Kubeconfig written to: $OUTPUT"
echo ""
echo "IMPORTANT: The HF Space container needs the AWS CLI and credentials"
echo "to authenticate with EKS. You have two options:"
echo ""
echo "Option A: Include aws-cli in your Docker image and set AWS_ACCESS_KEY_ID /"
echo "          AWS_SECRET_ACCESS_KEY as HF Space secrets."
echo ""
echo "Option B: Use the kubernetes Python client with AWS SDK (boto3)."
echo "          The kubernetes_executor.py already supports this via"
echo "          load_kube_config() which uses the Python client's auth plugins."
echo ""
echo "To encode for HF Spaces secret:"
echo "  cat $OUTPUT | base64 -w 0"
