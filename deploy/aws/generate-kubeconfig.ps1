# Generate a kubeconfig for HF Spaces to connect to the EKS cluster.
#
# This creates a kubeconfig that uses AWS IAM authenticator,
# which works from outside the cluster (like from HF Spaces).
#
# Prerequisites: aws cli, kubectl, eksctl
#
# Usage:
#   .\deploy\aws\generate-kubeconfig.ps1
#
# Output:
#   deploy/aws/kubeconfig-antiatropos.yaml
#
# Then on HF Spaces:
#   1. base64 encode: $b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes('deploy\aws\kubeconfig-antiatropos.yaml'))
#   2. Set as HF Space secret: KUBECONFIG_CONTENT = <base64 output>
#   3. Set env var: KUBECONFIG = /app/kubeconfig.yaml
#   4. Add to deploy/entrypoint.sh:
#        if [ -n "${KUBECONFIG_CONTENT:-}" ]; then
#            echo "${KUBECONFIG_CONTENT}" | base64 -d > /app/kubeconfig.yaml
#            export KUBECONFIG=/app/kubeconfig.yaml
#        fi

$ErrorActionPreference = "Stop"

$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "ap-southeast-1" }
$ClusterName = if ($env:CLUSTER_NAME) { $env:CLUSTER_NAME } else { "antiatropos" }
$AwsDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Output = Join-Path $AwsDir "kubeconfig-antiatropos.yaml"

Write-Host ""
Write-Host "=== Generating kubeconfig for HF Spaces ===" -ForegroundColor Cyan
Write-Host "Cluster: $ClusterName"
Write-Host "Region:  $Region"
Write-Host ""

# Verify cluster exists
$clusterExists = $false
try {
    eksctl get cluster --name $ClusterName --region $Region 2>$null | Out-Null
    $clusterExists = $true
} catch {}

if (-not $clusterExists) {
    Write-Host "ERROR: Cluster $ClusterName not found. Create it first with eksctl." -ForegroundColor Red
    exit 1
}

# Get cluster details
$ClusterEndpoint = aws eks describe-cluster --name $ClusterName --region $Region --query 'cluster.endpoint' --output text
$ClusterCa = aws eks describe-cluster --name $ClusterName --region $Region --query 'cluster.certificateAuthority.data' --output text
$AwsArn = aws sts get-caller-identity --query Arn --output text
$Timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

Write-Host "Cluster endpoint: $ClusterEndpoint"
Write-Host "AWS identity:     $AwsArn"
Write-Host ""

# Generate the kubeconfig
$kubeconfig = @"
# Kubeconfig for AntiAtropos on Hugging Face Spaces
# Generated: $Timestamp
# Cluster:   $ClusterName
# Region:    $Region
#
# This kubeconfig uses AWS IAM authenticator.
# The HF Space container must have aws-cli and aws-iam-authenticator available,
# OR the kubernetes Python client must be configured with AWS credentials.
#
# To use this on HF Spaces:
#   1. base64 encode this file
#   2. Set as HF secret: KUBECONFIG_CONTENT = <base64>
#   3. Set env var: KUBECONFIG = /app/kubeconfig.yaml
#   4. Decode in entrypoint.sh before uvicorn starts

apiVersion: v1
kind: Config
clusters:
  - cluster:
      certificate-authority-data: $ClusterCa
      server: $ClusterEndpoint
    name: $ClusterName

contexts:
  - context:
      cluster: $ClusterName
      user: antiatropos-hf-user
    name: $ClusterName

current-context: $ClusterName

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
          - $Region
          - --cluster-name
          - $ClusterName
        env:
          - name: AWS_STS_REGIONAL_ENDPOINTS
            value: regional
          - name: AWS_DEFAULT_REGION
            value: $Region
        interactiveMode: IfAvailable
"@

$kubeconfig | Out-File -FilePath $Output -Encoding utf8 -Force

Write-Host "Kubeconfig written to: $Output" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: The HF Space container needs the AWS CLI and credentials" -ForegroundColor Yellow
Write-Host "to authenticate with EKS. You have two options:"
Write-Host ""
Write-Host "Option A: Include aws-cli in your Docker image and set AWS_ACCESS_KEY_ID /"
Write-Host "          AWS_SECRET_ACCESS_KEY as HF Space secrets."
Write-Host ""
Write-Host "Option B: Use the kubernetes Python client with AWS SDK (boto3)."
Write-Host "          The kubernetes_executor.py already supports this via"
Write-Host "          load_kube_config() which uses the Python client's auth plugins."
Write-Host ""
Write-Host "To encode for HF Spaces secret:" -ForegroundColor Yellow
Write-Host "  [Convert]::ToBase64String([IO.File]::ReadAllBytes('$Output'))"
