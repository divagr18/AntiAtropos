# AntiAtropos AWS Infrastructure Deploy Script (PowerShell)
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
#   .\deploy\aws\deploy.ps1
#
# Environment variables:
#   $env:AWS_REGION     - AWS region (default: ap-south-1)
#   $env:CLUSTER_NAME   - EKS cluster name (default: antiatropos)

$ErrorActionPreference = "Stop"

$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "ap-south-1" }
$ClusterName = if ($env:CLUSTER_NAME) { $env:CLUSTER_NAME } else { "antiatropos" }
$AwsDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host ""
Write-Host "=== AntiAtropos AWS Infrastructure Deployment ===" -ForegroundColor Cyan
Write-Host "Region:      $Region"
Write-Host "Cluster:     $ClusterName"
Write-Host "FastAPI:     Runs on HF Spaces (not deployed here)"
Write-Host ""

# --- Check prerequisites ---
$missing = @()
foreach ($cmd in @("aws", "eksctl", "kubectl", "helm")) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        $missing += $cmd
    }
}
if ($missing.Count -gt 0) {
    Write-Host "ERROR: Missing prerequisites: $($missing -join ', ')" -ForegroundColor Red
    Write-Host "Install them first:" -ForegroundColor Yellow
    Write-Host "  choco install awscli eksctl kubernetes-cli kubernetes-helm -y" -ForegroundColor Yellow
    exit 1
}

# --- Phase 1: Create EKS Cluster ---
Write-Host ""
Write-Host ">>> Phase 1: Creating EKS cluster (without nodegroup)..." -ForegroundColor Yellow

$clusterExists = $false
try {
    eksctl get cluster --name $ClusterName --region $Region 2>$null | Out-Null
    $clusterExists = $true
} catch {}

if ($clusterExists) {
    Write-Host "Cluster $ClusterName already exists, skipping creation."
} else {
    # Create cluster without nodegroup first (faster, avoids timeout)
    $TempClusterConfig = Join-Path $AwsDir "eksctl-cluster-only.yaml"
    $ClusterYaml = Get-Content (Join-Path $AwsDir "eksctl-cluster.yaml") -Raw
    # Remove nodegroups section for initial cluster creation
    $ClusterOnlyYaml = $ClusterYaml -replace '(?s)(managedNodeGroups:.*)', ''
    $ClusterOnlyYaml | Out-File -FilePath $TempClusterConfig -Encoding utf8
    eksctl create cluster -f $TempClusterConfig
    Remove-Item $TempClusterConfig -Force
    Write-Host "Cluster created." -ForegroundColor Green
}

aws eks update-kubeconfig --name $ClusterName --region $Region
Write-Host "kubeconfig updated."

# --- Phase 1b: Create Nodegroup Separately ---
Write-Host ""
Write-Host ">>> Phase 1b: Creating nodegroup (separate step to avoid timeout)..." -ForegroundColor Yellow

$nodegroupExists = $false
try {
    eksctl get nodegroup --cluster $ClusterName --region $Region 2>$null | Select-String "linux-nodes" | Out-Null
    $nodegroupExists = $true
} catch {}

if ($nodegroupExists) {
    Write-Host "Nodegroup already exists, skipping creation."
} else {
    # Create nodegroup separately (better error handling, can retry)
    eksctl create nodegroup --config-file (Join-Path $AwsDir "eksctl-cluster.yaml")
    Write-Host "Nodegroup created." -ForegroundColor Green
}

# Verify nodes are ready
Write-Host "Waiting for nodes to be ready..."
$nodesReady = $false
for ($i = 0; $i -lt 30; $i++) {
    $nodes = kubectl get nodes --no-headers 2>$null
    if ($nodes) {
        Write-Host "Nodes ready:" -ForegroundColor Green
        kubectl get nodes
        $nodesReady = $true
        break
    }
    Start-Sleep -Seconds 10
}
if (-not $nodesReady) {
    Write-Host "WARNING: Nodes not ready yet. Check with: kubectl get nodes" -ForegroundColor Yellow
}

Write-Host "Enabling Prefix Delegation on VPC CNI..."
kubectl set env daemonset aws-node -n kube-system ENABLE_PREFIX_DELEGATION=true
Write-Host "Prefix Delegation enabled."

# --- Phase 2: Deploy Sample Workloads ---
Write-Host ""
Write-Host ">>> Phase 2: Deploying sample workloads (payments, checkout, catalog, cart, auth)..." -ForegroundColor Yellow
kubectl apply -f (Join-Path $AwsDir "k8s-workloads.yaml")
Write-Host "Workloads deployed." -ForegroundColor Green
kubectl get pods -n prod-sre

# --- Phase 3: Create AMP Workspace ---
Write-Host ""
Write-Host ">>> Phase 3: Creating Amazon Managed Prometheus workspace..." -ForegroundColor Yellow

$AmpWsId = $null
try {
    $AmpWsId = aws amp list-workspaces --alias antiatropos-metrics --region $Region --query 'workspaces[0].workspaceId' --output text 2>$null
    if ($AmpWsId -eq "None") { $AmpWsId = $null }
} catch {}

if ([string]::IsNullOrWhiteSpace($AmpWsId)) {
    $AmpWsId = aws amp create-workspace `
        --alias antiatropos-metrics `
        --region $Region `
        --query 'workspaceId' `
        --output text
    Write-Host "AMP workspace created: $AmpWsId" -ForegroundColor Green
} else {
    Write-Host "AMP workspace already exists: $AmpWsId"
}

$AmpUrl = "https://aps-workspaces.$Region.amazonaws.com/workspaces/$AmpWsId"
Write-Host "AMP URL: $AmpUrl"

# --- Phase 4: Set up IRSA for Prometheus Agent ---
Write-Host ""
Write-Host ">>> Phase 4: Setting up IRSA for Prometheus Agent..." -ForegroundColor Yellow

$saExists = $false
try {
    kubectl get serviceaccount prometheus-sa -n monitoring 2>$null | Out-Null
    $saExists = $true
} catch {}

if ($saExists) {
    Write-Host "prometheus-sa already exists."
} else {
    eksctl create iamserviceaccount `
        --cluster $ClusterName `
        --namespace monitoring `
        --name prometheus-sa `
        --attach-policy-arn "arn:aws:iam::aws:policy/AmazonPrometheusRemoteWriteAccess" `
        --approve `
        --override-existing-serviceaccounts
    Write-Host "prometheus-sa created." -ForegroundColor Green
}

# --- Phase 5: Install Prometheus Agent ---
Write-Host ""
Write-Host ">>> Phase 5: Installing Prometheus Agent (remote-writes to AMP)..." -ForegroundColor Yellow

helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>$null
helm repo update

$agentInstalled = $false
try {
    helm status prometheus-agent -n monitoring 2>$null | Out-Null
    $agentInstalled = $true
} catch {}

$promValuesYaml = Join-Path $AwsDir "prometheus-agent-values.yaml"
$remoteWriteUrl = "$AmpUrl/api/v1/remote_write"

if ($agentInstalled) {
    Write-Host "prometheus-agent already installed, upgrading..."
    helm upgrade prometheus-agent prometheus-community/prometheus `
        --namespace monitoring `
        -f $promValuesYaml `
        --set "prometheus.prometheusSpec.remoteWrite[0].url=$remoteWriteUrl"
} else {
    helm install prometheus-agent prometheus-community/prometheus `
        --namespace monitoring --create-namespace `
        -f $promValuesYaml `
        --set "prometheus.prometheusSpec.remoteWrite[0].url=$remoteWriteUrl"
    Write-Host "prometheus-agent installed." -ForegroundColor Green
}

# --- Phase 6: Install Self-Hosted Grafana on EKS ---
Write-Host ""
Write-Host ">>> Phase 6: Installing self-hosted Grafana on EKS..." -ForegroundColor Yellow

# Add Grafana Helm repo
helm repo add grafana https://grafana.github.io/helm-charts 2>$null
helm repo update

# Create a secret with the dashboard JSON files for Grafana to import
$DashboardsDir = Join-Path $PSScriptRoot "..\..\grafana\provisioning\dashboards\json"
if (Test-Path $DashboardsDir) {
    Write-Host "Creating dashboard secret from $DashboardsDir..."
    kubectl create secret generic antiatropos-grafana-dashboards `
        --from-file=antiatropos-overview.json=$(Join-Path $DashboardsDir "antiatropos-overview.json") `
        --from-file=antiatropos-live.json=$(Join-Path $DashboardsDir "antiatropos-live.json") `
        --namespace monitoring `
        --dry-run=client -o yaml | kubectl apply -f -
    Write-Host "Dashboard secret created." -ForegroundColor Green
} else {
    Write-Host "Dashboard JSON directory not found at $DashboardsDir, skipping."
}

# Install Grafana
$GrafanaValuesYaml = Join-Path $AwsDir "grafana-values.yaml"

if (helm status grafana -n monitoring 2>$null) {
    Write-Host "Grafana already installed, upgrading..."
    helm upgrade grafana grafana/grafana --namespace monitoring -f $GrafanaValuesYaml
} else {
    helm install grafana grafana/grafana --namespace monitoring -f $GrafanaValuesYaml
    Write-Host "Grafana installed." -ForegroundColor Green
}

# Wait for Grafana pod to be ready
Write-Host "Waiting for Grafana pod to be ready..."
kubectl rollout status deployment/grafana --namespace monitoring --timeout=120s 2>$null | Out-Null

$GrafanaPod = kubectl get pods -n monitoring -l app.kubernetes.io/name=grafana -o jsonpath='{.items[0].metadata.name}' 2>$null
Write-Host "Grafana pod: $GrafanaPod"
Write-Host "To access Grafana: kubectl port-forward svc/grafana 3000 -n monitoring" -ForegroundColor Yellow
Write-Host "Login: admin / antiatropos"

# --- Phase 7: Install Cluster Autoscaler ---
Write-Host ""
Write-Host ">>> Phase 7: Installing Cluster Autoscaler..." -ForegroundColor Yellow

helm repo add autoscaler https://kubernetes.github.io/autoscaler 2>$null
helm repo update

$autoscalerInstalled = $false
try {
    helm status cluster-autoscaler -n kube-system 2>$null | Out-Null
    $autoscalerInstalled = $true
} catch {}

$autoscalerValues = Join-Path $AwsDir "cluster-autoscaler-values.yaml"

if ($autoscalerInstalled) {
    Write-Host "cluster-autoscaler already installed, upgrading..."
    helm upgrade cluster-autoscaler autoscaler/cluster-autoscaler `
        --namespace kube-system `
        -f $autoscalerValues
} else {
    helm install cluster-autoscaler autoscaler/cluster-autoscaler `
        --namespace kube-system `
        -f $autoscalerValues
    Write-Host "cluster-autoscaler installed." -ForegroundColor Green
}

# --- Phase 8: Generate Kubeconfig for HF Spaces ---
Write-Host ""
Write-Host ">>> Phase 8: Generating kubeconfig for HF Spaces..." -ForegroundColor Yellow

$generateScript = Join-Path $AwsDir "generate-kubeconfig.ps1"
if (Test-Path $generateScript) {
    & $generateScript
} else {
    # Inline kubeconfig generation if the .ps1 version doesn't exist yet
    $output = Join-Path $AwsDir "kubeconfig-antiatropos.yaml"

    # Verify cluster exists
    $clusterCheck = $false
    try {
        eksctl get cluster --name $ClusterName --region $Region 2>$null | Out-Null
        $clusterCheck = $true
    } catch {}
    if (-not $clusterCheck) {
        Write-Host "ERROR: Cluster $ClusterName not found." -ForegroundColor Red
        exit 1
    }

    $ClusterEndpoint = aws eks describe-cluster --name $ClusterName --region $Region --query 'cluster.endpoint' --output text
    $ClusterCa = aws eks describe-cluster --name $ClusterName --region $Region --query 'cluster.certificateAuthority.data' --output text
    $Timestamp = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")

    $kubeconfig = @"
# Kubeconfig for AntiAtropos on Hugging Face Spaces
# Generated: $Timestamp
# Cluster:   $ClusterName
# Region:    $Region
#
# This kubeconfig uses AWS IAM authenticator.
# The HF Space container must have aws-cli available,
# OR the kubernetes Python client must be configured with AWS credentials.

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
          - get-token
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

    $kubeconfig | Out-File -FilePath $output -Encoding utf8 -Force
    Write-Host "Kubeconfig written to: $output" -ForegroundColor Green
    Write-Host ""
    Write-Host "To encode for HF Spaces secret:" -ForegroundColor Yellow
    Write-Host "  [Convert]::ToBase64String([System.IO.File]::ReadAllBytes('$output'))"
}

# --- Done ---
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   AntiAtropos AWS Infrastructure Ready!" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "AMP Workspace ID:  $AmpWsId"
Write-Host "AMP URL:           $AmpUrl"
Write-Host ""
Write-Host "Grafana: Self-hosted on EKS (monitoring namespace)"
Write-Host "  Access: kubectl port-forward svc/grafana 3000 -n monitoring"
Write-Host "  Login: admin / antiatropos"
Write-Host "  URL: http://localhost:3000"
Write-Host ""
Write-Host "Kubeconfig saved:  $(Join-Path $AwsDir 'kubeconfig-antiatropos.yaml')"
Write-Host ""
Write-Host "Next steps - configure your HF Space:" -ForegroundColor Yellow
Write-Host "  1. Set secret KUBECONFIG_CONTENT = base64 of kubeconfig-antiatropos.yaml"
Write-Host "  2. Set env var PROMETHEUS_URL = $AmpUrl"
Write-Host "  3. Set env var KUBECONFIG = /app/kubeconfig.yaml"
Write-Host "  4. Set env var ANTIATROPOS_ENV_MODE = live"
Write-Host "  5. Set env var ANTIATROPOS_MAX_REPLICAS = 6"
Write-Host "  6. Set env var ANTIATROPOS_WORKLOAD_MAP = (see OPERATIONS.md)"
Write-Host "  7. Add kubeconfig decode to deploy/entrypoint.sh (see OPERATIONS.md)"
