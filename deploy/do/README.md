# DigitalOcean Droplet one-shot deploy

This deploy flow is for a single Ubuntu Droplet running:
- k3s (single-node Kubernetes)
- AntiAtropos sample workloads (`prod-sre`)
- Prometheus + Grafana (`monitoring`)
- FastAPI control server (`antiatropos-fastapi` systemd service)

## Run

From repository root on the Droplet:

```bash
sudo bash deploy/do/deploy-droplet-one-shot.sh
```

Optional overrides:

```bash
sudo REPO_DIR=/opt/AntiAtropos FASTAPI_PORT=8010 MAX_REPLICAS=200 bash deploy/do/deploy-droplet-one-shot.sh
```

## What the script configures

- k3s kubelet with `max-pods=250`
- Prometheus service exposed on NodePort `30090`
- Prometheus scrape job for annotated pods in namespace `prod-sre`
- Env file at `.env.droplet` with:
  - `ANTIATROPOS_ENV_MODE=live`
  - `KUBECONFIG=/etc/rancher/k3s/k3s.yaml`
  - `ANTIATROPOS_WORKLOAD_MAP` for `node-0`..`node-4`
- Systemd service:
  - Name: `antiatropos-fastapi`
  - Exec: `uvicorn server.app:app --host 0.0.0.0 --port 8000`

## Verify

```bash
systemctl status antiatropos-fastapi --no-pager
curl http://127.0.0.1:8000/config/runtime
kubectl get deploy -n prod-sre
kubectl get pods -n monitoring
curl http://127.0.0.1:30090/api/v1/targets
kubectl -n monitoring port-forward svc/grafana 3000:80
```

If your local simulator FastAPI should use VM telemetry, set local `.env`:

```env
PROMETHEUS_URL=http://<droplet-ip>:30090
```

## Agent call example

```bash
curl -X POST http://127.0.0.1:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"SCALE_UP","target_node_id":"node-3","parameter":0.6}'
```
