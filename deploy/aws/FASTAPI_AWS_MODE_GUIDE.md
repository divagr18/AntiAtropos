# FastAPI AWS Mode + Local Grafana Guide

This setup keeps Kubernetes + AMP in AWS, while Grafana runs on your laptop.

## 1) Environment file

Use [../../.env.example](../../.env.example) as template. A starter [../../.env](../../.env) is already created.

Important keys:

- `ANTIATROPOS_ENV_MODE=aws`
- `KUBECONFIG=.../deploy/aws/kubeconfig-antiatropos.yaml`
- `PROMETHEUS_URL=https://aps-workspaces.<region>.amazonaws.com/workspaces/<workspace-id>`
- `ANTIATROPOS_WORKLOAD_MAP=...`
- `ANTIATROPOS_GRAFANA_MODE=external`

## 2) Load .env in PowerShell

From workspace root:

```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
  $name, $value = $_ -split '=', 2
  [System.Environment]::SetEnvironmentVariable($name, $value, 'Process')
}
```

## 3) Start FastAPI server

```powershell
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## 4) Verify runtime wiring

Check runtime endpoint:

- [server/app.py](../../server/app.py) exposes `GET /config/runtime`
- Example URL: `http://localhost:8000/config/runtime`

You should see:

- `env_mode: "aws"`
- `prometheus_url_configured: true`
- `kubeconfig_configured: true`
- `workload_map_configured: true`

## 5) Reset environment in AWS mode

Use reset with `mode="aws"`, or omit mode and rely on `ANTIATROPOS_ENV_MODE=aws`.

## 6) Run Grafana locally (not in EKS)

```powershell
docker run -d --name antiatropos-grafana -p 3000:3000 grafana/grafana:latest
```

Open `http://localhost:3000` and add AMP as Prometheus datasource:

- URL: `https://aps-workspaces.<region>.amazonaws.com/workspaces/<workspace-id>`
- Auth: SigV4 enabled
- Region: your AWS region (for example `ap-south-1`)

Import dashboards:

- [../grafana/provisioning/dashboards/json/antiatropos-overview.json](../grafana/provisioning/dashboards/json/antiatropos-overview.json)
- [../grafana/provisioning/dashboards/json/antiatropos-live.json](../grafana/provisioning/dashboards/json/antiatropos-live.json)

## Notes

Grafana is observability-only. Agent control runs via FastAPI + Kubernetes executor.
