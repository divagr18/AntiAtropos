# DigitalOcean Droplet one-shot deploy

This deploy flow is for a single Ubuntu Droplet running:
- k3s (single-node Kubernetes)
- AntiAtropos sample workloads (`prod-sre`)
- Prometheus + Grafana (`monitoring`)
- lightweight control-plane API (`antiatropos-control` on port `8010`)

The OpenEnv runtime (`server.app`) is intentionally **not** run on the droplet.
The only supported split is:
- local machine: OpenEnv server + inference loop
- droplet: Kubernetes executor API + observability stack

## Run

From repository root on the Droplet:

```bash
sudo bash deploy/do/deploy-droplet-one-shot.sh
```

Optional overrides:

```bash
sudo REPO_DIR=/opt/AntiAtropos CONTROL_PORT=8010 MAX_REPLICAS=200 bash deploy/do/deploy-droplet-one-shot.sh
```

## What the script configures

- k3s kubelet with `max-pods=250`
- Prometheus service exposed on NodePort `30090`
- Prometheus scrape job for annotated pods in namespace `prod-sre`
- Env file at `.env.droplet` with:
  - `KUBECONFIG=/etc/rancher/k3s/k3s.yaml`
  - `ANTIATROPOS_WORKLOAD_MAP` for `node-0`..`node-4`
- Systemd service:
  - Name: `antiatropos-control`
  - Exec: `uvicorn server.local_laptop_control:app --host 0.0.0.0 --port 8010`
- Legacy cleanup:
  - `antiatropos-fastapi` (VM OpenEnv service) is disabled/removed by default deploy path

## Verify

```bash
systemctl status antiatropos-control --no-pager
curl http://127.0.0.1:8010/health
kubectl get deploy -n prod-sre
kubectl get pods -n monitoring
curl http://127.0.0.1:30090/api/v1/targets
kubectl -n monitoring port-forward svc/grafana 3000:80
```

Set local `.env` to use this consolidated path:

```env
ENV_URL=http://localhost:8000
ANTIATROPOS_CONTROL_PLANE_URL=http://<droplet-ip>:8010
PROMETHEUS_URL=http://<droplet-ip>:30090
```

## Deterministic remote-scaling proof

On droplet, watch desired replicas:

```bash
watch -n 1 'kubectl -n prod-sre get deploy -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,READY:.status.readyReplicas,AVAILABLE:.status.availableReplicas'
```

From local machine, send one control action:

```bash
curl -X POST http://<droplet-ip>:8010/step \
  -H "Content-Type: application/json" \
  -d '{"action_type":"SCALE_UP","target_node_id":"node-0","parameter":1.0}'
```

If `payments` desired replicas increase, scaling is happening on droplet.

## Troubleshooting

- **Pods do not move during inference**
  - Verify local env points to droplet control API:
    - `ANTIATROPOS_CONTROL_PLANE_URL=http://<droplet-ip>:8010`
  - Check droplet control health:
    - `curl http://127.0.0.1:8010/health`
  - Check service status:
    - `systemctl status antiatropos-control --no-pager`
- **Connection refused from local to droplet:8010**
  - Service not running or firewall closed.
  - Start service and open firewall if needed.
- **Need to remove legacy VM OpenEnv service**
  - `sudo bash deploy/do/uninstall-legacy-openenv.sh`
