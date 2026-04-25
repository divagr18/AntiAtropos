# Local Laptop Kubernetes Control with FastAPI

This guide uses your local manifest [deploy/local-laptop.yaml](deploy/local-laptop.yaml) and a lightweight server [server/local_laptop_control.py](server/local_laptop_control.py).

## 1) Deploy local workloads

```powershell
kubectl apply -f deploy/local-laptop.yaml
kubectl get deploy -n prod-sre
```

Expected deployments:
- `auth`
- `cart`
- `catalog`
- `checkout`
- `payments`

## 2) Set required environment variables

The controller requires `KUBECONFIG` and `ANTIATROPOS_WORKLOAD_MAP`.

```powershell
$env:KUBECONFIG = "$HOME/.kube/config"
$env:ANTIATROPOS_K8S_NAMESPACE = "prod-sre"
$env:ANTIATROPOS_MIN_REPLICAS = "1"
$env:ANTIATROPOS_MAX_REPLICAS = "6"
$env:ANTIATROPOS_SCALE_STEP = "3"
$env:ANTIATROPOS_WORKLOAD_MAP = '{"node-0":{"deployment":"payments","namespace":"prod-sre"},"node-1":{"deployment":"checkout","namespace":"prod-sre"},"node-2":{"deployment":"catalog","namespace":"prod-sre"},"node-3":{"deployment":"cart","namespace":"prod-sre"},"node-4":{"deployment":"auth","namespace":"prod-sre"}}'
```

If you already have these in [.env](.env), load them first.

## 3) Start lightweight FastAPI server

```powershell
uvicorn server.local_laptop_control:app --host 0.0.0.0 --port 8010
```

## 4) Validate server health

```powershell
Invoke-RestMethod http://localhost:8010/health
```

Check:
- `is_mock` should be `False`
- `mapped_targets` should include `node-0`..`node-4`

## 5) Let your agent execute actions

The server accepts `POST /step` with:
- `action_type`: `NO_OP` | `SCALE_UP` | `SCALE_DOWN`
- `target_node_id`: `node-*`
- `parameter`: float

Example:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8010/step -ContentType "application/json" -Body '{"action_type":"SCALE_UP","target_node_id":"node-3","parameter":0.6}'
```

## 6) Verify Kubernetes effect

```powershell
kubectl get deploy cart -n prod-sre
kubectl get deploy -n prod-sre
```

## Notes

- This controller is intentionally minimal and does not provide simulator rewards.
- It is suitable for direct action execution tests from your agent.
- If you need OpenEnv-compatible `/reset` + `/step` + reward loop, use [server/app.py](server/app.py) in `aws` mode.
