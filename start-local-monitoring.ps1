# Start local Prometheus + Grafana for AntiAtropos control-plane metrics
# This scrapes localhost:8000/metrics (the local OpenEnv server)
#
# Requirements: Docker Desktop with WSL2 backend

$ErrorActionPreference = "Stop"

Write-Host "=== AntiAtropos Local Monitoring ===" -ForegroundColor Cyan
Write-Host ""

# Check Docker
$docker = Get-Command docker -ErrorAction SilentlyContinue
if (-not $docker) {
    Write-Host "ERROR: Docker not found. Install Docker Desktop first." -ForegroundColor Red
    exit 1
}

Write-Host "Starting local Prometheus + Grafana..." -ForegroundColor Yellow

# Ensure we're in the repo root
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $repoRoot

# Start the stack
docker compose -f docker-compose.local-monitoring.yml up -d

Write-Host ""
Write-Host "Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Health checks
$promReady = $false
$grafReady = $false
$retries = 10

for ($i = 0; $i -lt $retries; $i++) {
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 2
        $promReady = $true
    } catch {}

    try {
        $null = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 2
        $grafReady = $true
    } catch {}

    if ($promReady -and $grafReady) { break }
    Start-Sleep -Seconds 2
}

Write-Host ""
Write-Host "=== Local Monitoring Ready ===" -ForegroundColor Green

if ($promReady) {
    Write-Host "  Prometheus:  http://localhost:9090" -ForegroundColor Green
    Write-Host "  Targets:     http://localhost:9090/targets" -ForegroundColor Gray
} else {
    Write-Host "  Prometheus:  NOT READY (check docker logs antiatropos-prometheus-local)" -ForegroundColor Red
}

if ($grafReady) {
    Write-Host "  Grafana:     http://localhost:3000  (admin / antiatropos)" -ForegroundColor Green
    Write-Host "  Dashboards:  AntiAtropos Overview, AntiAtropos Live Control Plane" -ForegroundColor Gray
} else {
    Write-Host "  Grafana:     NOT READY (check docker logs antiatropos-grafana-local)" -ForegroundColor Red
}

Write-Host ""
Write-Host "NOTE: This shows reward/lyapunov/actions from your LOCAL OpenEnv server." -ForegroundColor Cyan
Write-Host "      For workload pod metrics (CPU/requests/queue), use VM Grafana at:" -ForegroundColor Cyan
Write-Host "      http://206.189.136.21:30000" -ForegroundColor Cyan
Write-Host ""
Write-Host "To stop:  docker compose -f docker-compose.local-monitoring.yml down" -ForegroundColor Gray
