# Observability Setup Guide

This guide covers end-to-end setup for:
- AntiAtropos server metrics (`/metrics`)
- Prometheus scraping
- Grafana dashboards
- Kubernetes metrics via `kube-state-metrics`

It is written for Windows + Docker Desktop Kubernetes.

## 1) Prerequisites

- Docker Desktop running
- Kubernetes enabled in Docker Desktop
- `kubectl` available in PATH
- Project root: `D:\Anti-Atropos`

Quick checks:

```powershell
kubectl config current-context
kubectl get nodes
```

Expected context: `docker-desktop` and at least one `Ready` node.

## 2) Create Demo Namespace and Workloads

Create namespace and deployments used by live mapping:

```powershell
kubectl create namespace sre-sandbox

kubectl -n sre-sandbox create deployment payments-api --image=nginx --replicas=1
kubectl -n sre-sandbox create deployment checkout-api --image=nginx --replicas=1
kubectl -n sre-sandbox create deployment inventory-api --image=nginx --replicas=1
kubectl -n sre-sandbox create deployment search-api --image=nginx --replicas=1
kubectl -n sre-sandbox create deployment gateway-api --image=nginx --replicas=1
```

Verify:

```powershell
kubectl -n sre-sandbox get deploy,pod
```

## 3) Start AntiAtropos Server with Live Config

Run in the same PowerShell session:

```powershell
cd D:\Anti-Atropos

$env:KUBECONFIG="$HOME\.kube\config"
$env:ANTIATROPOS_K8S_NAMESPACE="sre-sandbox"
$env:ANTIATROPOS_WORKLOAD_MAP='{"node-0":{"deployment":"payments-api","namespace":"sre-sandbox"},"node-1":{"deployment":"checkout-api","namespace":"sre-sandbox"},"node-2":{"deployment":"inventory-api","namespace":"sre-sandbox"},"node-3":{"deployment":"search-api","namespace":"sre-sandbox"},"node-4":{"deployment":"gateway-api","namespace":"sre-sandbox"}}'

python -m uvicorn AntiAtropos.server.app:app --host 0.0.0.0 --port 8000 --log-level info
```

Check server metrics endpoint:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content
```

## 4) Start Prometheus + Grafana

In another terminal:

```powershell
cd D:\Anti-Atropos
docker compose -f docker-compose.observability.yml up -d
```

Open:
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000` (`admin` / `admin`)

## 5) Install kube-state-metrics (one-time)

If not installed:

```powershell
kubectl apply -f https://raw.githubusercontent.com/kubernetes/kube-state-metrics/main/examples/standard/standard.yaml
```

Verify:

```powershell
kubectl -n kube-system get deploy,svc,pod | Select-String kube-state-metrics
```

## 6) Forward kube-state-metrics to Host

Keep this running in a dedicated terminal:

```powershell
kubectl -n kube-system port-forward svc/kube-state-metrics 8081:8080 --address 0.0.0.0
```

Verify host can read it:

```powershell
(Invoke-WebRequest http://127.0.0.1:8081/metrics).Content | Select-String "kube_deployment_spec_replicas|kube_pod_status_phase"
```

## 7) Verify Prometheus Scrapes Everything

Open `http://127.0.0.1:9090/targets`

Expected jobs:
- `antiatropos` = `UP`
- `kube-state-metrics` = `UP`

If `kube-state-metrics` is `DOWN`, restart Prometheus:

```powershell
docker compose -f docker-compose.observability.yml restart prometheus
```

## 8) Grafana Dashboards

Preprovisioned dashboards:
- `AntiAtropos Live Control Plane`
- `Kubernetes Overview (sre-sandbox)`

If a dashboard is missing:

```powershell
docker compose -f docker-compose.observability.yml restart grafana
```

## 9) Useful Prometheus Queries

AntiAtropos control-plane:

```promql
sum(rate(antiatropos_actions_total[1m])) by (action_type, ack_class)
antiatropos_reward
histogram_quantile(0.95, sum(rate(antiatropos_executor_latency_ms_bucket[2m])) by (le, mode))
sum(rate(antiatropos_executor_errors_total[5m])) by (error_code)
```

Kubernetes:

```promql
kube_deployment_spec_replicas{namespace="sre-sandbox"}
kube_deployment_status_replicas_available{namespace="sre-sandbox"}
sum by (phase) (kube_pod_status_phase{namespace="sre-sandbox"} == 1)
sum(increase(kube_pod_container_status_restarts_total{namespace="sre-sandbox"}[10m]))
```

## 10) Troubleshooting

### A) `MODE_UNSUPPORTED` keeps increasing

Cause: server running without real live env vars or old process still bound to `:8000`.

Fix:

```powershell
$pid8000 = (Get-NetTCPConnection -LocalPort 8000 -State Listen).OwningProcess
Stop-Process -Id $pid8000 -Force
```

Then restart server with `KUBECONFIG`, `ANTIATROPOS_K8S_NAMESPACE`, and `ANTIATROPOS_WORKLOAD_MAP` set.

### B) Prometheus fails to start with duplicate job name

Cause: duplicate `job_name` in `observability/prometheus/prometheus.yml`.

Fix: keep only one `kube-state-metrics` block, then:

```powershell
docker compose -f docker-compose.observability.yml restart prometheus
```

### C) Grafana K8s dashboard shows “No data”

Check in order:
1. `kubectl port-forward ... --address 0.0.0.0` is running
2. `http://127.0.0.1:8081/metrics` returns kube metrics
3. Prometheus `targets` page shows `kube-state-metrics` as `UP`

### D) Dashboard missing

```powershell
docker compose -f docker-compose.observability.yml restart grafana
```

## 11) Run the Agent

In a new terminal:

```powershell
cd D:\Anti-Atropos
python .\llm_agent_task1.py --env-url http://127.0.0.1:8000
```

Optional live scale watch:

```powershell
kubectl -n sre-sandbox get deploy payments-api -w
```
