# Start Grafana port-forward from Kind cluster
# Run 'deploy-local.ps1' first to ensure Grafana is deployed.

$port = 3000

Write-Host "Starting Grafana port-forward on localhost:$port..."

# Kill any existing on that port
$existing = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
if ($existing) {
    Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

kubectl port-forward -n monitoring svc/grafana ${port}:80
