# AntiAtropos Local Cluster Teardown
# Removes workloads, Prometheus, and Grafana. Stops port-forwards.

Write-Host "=== AntiAtropos Local Teardown ===" -ForegroundColor Cyan

# --- Stop port-forward ---
Write-Host "[1/3] Stopping port-forwards..." -ForegroundColor Yellow
$connections = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
if ($connections) {
    $connections | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Write-Host "  Stopped port-forward on :3000"
}

# --- Uninstall Helm releases ---
Write-Host "[2/3] Uninstalling Helm releases..." -ForegroundColor Yellow
helm uninstall grafana -n monitoring 2>&1 | Out-Null
Write-Host "  Grafana uninstalled."
helm uninstall prometheus -n monitoring 2>&1 | Out-Null
Write-Host "  Prometheus uninstalled."

# --- Delete namespaces ---
Write-Host "[3/3] Deleting namespaces..." -ForegroundColor Yellow
kubectl delete ns prod-sre monitoring 2>&1 | Out-Null

Start-Sleep -Seconds 3
Write-Host ""
Write-Host "=== Teardown Complete ===" -ForegroundColor Green
kubectl get ns
