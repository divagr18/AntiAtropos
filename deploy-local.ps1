# AntiAtropos Local Cluster Deploy
# Deploys workloads, Prometheus, and Grafana on the Kind cluster.
# Grafana port-forward starts automatically at the end.

param(
    [switch]$SkipPortForward,
    [int]$GrafanaPort = 3000
)

Write-Host "=== AntiAtropos Local Deploy ===" -ForegroundColor Cyan
Write-Host ""

# --- 1. Check cluster ---
Write-Host "[1/5] Checking Kind cluster..." -ForegroundColor Yellow
$cluster = kubectl config current-context 2>$null
if ($cluster -notmatch "antiatropos") {
    Write-Host "WARNING: Current context is '$cluster', expected 'kind-antiatropos-local'. Proceed anyway? [Y/n]"
    $r = Read-Host
    if ($r -eq 'n') { exit 1 }
}

# --- 2. Deploy workload pods ---
Write-Host "[2/5] Deploying workload pods..." -ForegroundColor Yellow
kubectl create ns prod-sre 2>&1 | Out-Null
kubectl create ns monitoring 2>&1 | Out-Null
kubectl apply -f "$PSScriptRoot\deploy\local-laptop.yaml"
Write-Host "  Waiting for workloads to be ready..."
kubectl wait --for=condition=ready pod -l app --all -n prod-sre --timeout=120s 2>$null
Write-Host "  Workloads ready."

# --- 3. Deploy Prometheus ---
Write-Host "[3/5] Deploying Prometheus..." -ForegroundColor Yellow
$promRelease = helm list -n monitoring -q 2>$null | Select-String "prometheus"
if ($promRelease) {
    helm upgrade prometheus prometheus-community/prometheus -n monitoring -f "$PSScriptRoot\deploy\prometheus-helm-values.yaml"
} else {
    helm install prometheus prometheus-community/prometheus -n monitoring -f "$PSScriptRoot\deploy\prometheus-helm-values.yaml"
}
Write-Host "  Waiting for Prometheus server..."
kubectl wait --for=condition=ready pod -l "app.kubernetes.io/name=prometheus" -n monitoring --timeout=120s 2>$null
Write-Host "  Prometheus ready."

# --- 4. Deploy Grafana ---
Write-Host "[4/5] Deploying Grafana..." -ForegroundColor Yellow
# Update dashboard ConfigMap
kubectl delete configmap grafana-dashboards -n monitoring 2>$null
kubectl create configmap grafana-dashboards -n monitoring --from-file="$PSScriptRoot\deploy\grafana\provisioning\dashboards\json\"

$grafRelease = helm list -n monitoring -q 2>$null | Select-String "grafana"
if ($grafRelease) {
    helm upgrade grafana grafana/grafana -n monitoring -f "$PSScriptRoot\deploy\grafana-helm-values.yaml"
} else {
    helm install grafana grafana/grafana -n monitoring -f "$PSScriptRoot\deploy\grafana-helm-values.yaml"
}
Write-Host "  Waiting for Grafana..."
kubectl wait --for=condition=ready pod -l "app.kubernetes.io/name=grafana" -n monitoring --timeout=120s 2>$null
Write-Host "  Grafana ready."

# --- 5. Start Grafana port-forward ---
Write-Host "[5/5] Grafana port-forward..." -ForegroundColor Yellow
if (-not $SkipPortForward) {
    # Kill any existing port-forward on the same port
    $existing = Get-NetTCPConnection -LocalPort $GrafanaPort -ErrorAction SilentlyContinue 2>$null
    if ($existing) {
        $pid = $existing.OwningProcess
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue 2>$null
        Start-Sleep -Seconds 1
    }

    Write-Host "  Starting port-forward on localhost:$GrafanaPort..."
    $proc = Start-Process -PassThru -NoNewWindow kubectl -ArgumentList "port-forward","-n","monitoring","svc/grafana","${GrafanaPort}:80"

    Start-Sleep -Seconds 2
    # Verify the port-forward is alive
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:$GrafanaPort/api/health" -UseBasicParsing -TimeoutSec 5
        Write-Host ""
        Write-Host "=== Deploy Complete ===" -ForegroundColor Green
        Write-Host "  Grafana:  http://localhost:$GrafanaPort  (admin / antiatropos)"
        Write-Host "  Dashboards: AntiAtropos Overview, AntiAtropos Live Control Plane"
        Write-Host "  Port-forward PID: $($proc.Id)"
        Write-Host ""
        Write-Host "To stop port-forward: Stop-Process -Id $($proc.Id)"
    } catch {
        Write-Host "WARNING: Port-forward started but Grafana not reachable yet. Try: kubectl port-forward -n monitoring svc/grafana ${GrafanaPort}:80"
    }
} else {
    Write-Host ""
    Write-Host "=== Deploy Complete ===" -ForegroundColor Green
    Write-Host "  To access Grafana: kubectl port-forward -n monitoring svc/grafana ${GrafanaPort}:80"
}
