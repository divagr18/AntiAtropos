#!/usr/bin/env bash
set -euo pipefail

# One-shot deploy for a single DigitalOcean Droplet:
# - Installs k3s with kubelet max-pods=250
# - Deploys workloads + Prometheus + Grafana
# - Creates env file for live Kubernetes scaling
# - Starts lightweight control-plane API via systemd (antiatropos-control)

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo bash deploy/do/deploy-droplet-one-shot.sh"
  exit 1
fi

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-/etc/rancher/k3s/k3s.yaml}"
CONTROL_PORT="${CONTROL_PORT:-8010}"
CONTROL_HOST="${CONTROL_HOST:-0.0.0.0}"
K8S_NAMESPACE="${K8S_NAMESPACE:-prod-sre}"
MONITORING_NAMESPACE="${MONITORING_NAMESPACE:-monitoring}"
PY_VENV_DIR="${PY_VENV_DIR:-${REPO_DIR}/.venv-droplet}"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env.droplet}"
MIN_REPLICAS="${MIN_REPLICAS:-1}"
MAX_REPLICAS="${MAX_REPLICAS:-250}"
SCALE_STEP="${SCALE_STEP:-3}"
WORKLOAD_MAP="${WORKLOAD_MAP:-{\"node-0\":{\"deployment\":\"payments\",\"namespace\":\"prod-sre\"},\"node-1\":{\"deployment\":\"checkout\",\"namespace\":\"prod-sre\"},\"node-2\":{\"deployment\":\"catalog\",\"namespace\":\"prod-sre\"},\"node-3\":{\"deployment\":\"cart\",\"namespace\":\"prod-sre\"},\"node-4\":{\"deployment\":\"auth\",\"namespace\":\"prod-sre\"}}}"

echo "=== AntiAtropos Droplet One-Shot Deploy ==="
echo "Repo:        ${REPO_DIR}"
echo "Kubeconfig:  ${KUBECONFIG_PATH}"
echo "Control API: ${CONTROL_HOST}:${CONTROL_PORT}"
echo ""

if [[ ! -f "${REPO_DIR}/deploy/local-laptop.yaml" ]]; then
  echo "ERROR: deploy/local-laptop.yaml not found. Run from AntiAtropos checkout."
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y curl ca-certificates gnupg lsb-release python3 python3-venv python3-pip

if ! command -v kubectl >/dev/null 2>&1; then
  echo "Installing k3s (includes kubectl)..."
  curl -sfL https://get.k3s.io | sh -s - --write-kubeconfig-mode 644 --kubelet-arg=max-pods=250
else
  echo "k3s/kubectl already present; skipping k3s install."
fi

if ! command -v helm >/dev/null 2>&1; then
  echo "Installing Helm..."
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
fi

export KUBECONFIG="${KUBECONFIG_PATH}"

echo "Waiting for Kubernetes node to be Ready..."
kubectl wait --for=condition=Ready node --all --timeout=180s

kubectl create ns "${K8S_NAMESPACE}" >/dev/null 2>&1 || true
kubectl create ns "${MONITORING_NAMESPACE}" >/dev/null 2>&1 || true

echo "Deploying AntiAtropos workloads..."
kubectl apply -f "${REPO_DIR}/deploy/local-laptop.yaml"

echo "Installing/upgrading Prometheus + Grafana..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null 2>&1 || true
helm repo add grafana https://grafana.github.io/helm-charts >/dev/null 2>&1 || true
helm repo update

helm upgrade --install prometheus prometheus-community/prometheus \
  -n "${MONITORING_NAMESPACE}" \
  -f "${REPO_DIR}/deploy/prometheus-helm-values.yaml"

if [[ -d "${REPO_DIR}/deploy/grafana/provisioning/dashboards/json" ]]; then
  kubectl delete configmap grafana-dashboards -n "${MONITORING_NAMESPACE}" >/dev/null 2>&1 || true
  kubectl create configmap grafana-dashboards \
    -n "${MONITORING_NAMESPACE}" \
    --from-file="${REPO_DIR}/deploy/grafana/provisioning/dashboards/json/"
fi

helm upgrade --install grafana grafana/grafana \
  -n "${MONITORING_NAMESPACE}" \
  -f "${REPO_DIR}/deploy/grafana-helm-values.yaml"

if [[ ! -f "${ENV_FILE}" ]]; then
  cat > "${ENV_FILE}" <<EOF
KUBECONFIG=/etc/rancher/k3s/k3s.yaml
ANTIATROPOS_K8S_NAMESPACE=prod-sre
ANTIATROPOS_MIN_REPLICAS=${MIN_REPLICAS}
ANTIATROPOS_MAX_REPLICAS=${MAX_REPLICAS}
ANTIATROPOS_SCALE_STEP=${SCALE_STEP}
ANTIATROPOS_WORKLOAD_MAP=${WORKLOAD_MAP}
EOF
  echo "Created ${ENV_FILE}"
else
  echo "Using existing ${ENV_FILE}"
fi

echo "Preparing Python environment..."
python3 -m venv "${PY_VENV_DIR}"
"${PY_VENV_DIR}/bin/python" -m pip install --upgrade pip
if [[ -f "${REPO_DIR}/pyproject.toml" ]]; then
  # Prefer project metadata (uses openenv-core, not legacy openenv package name).
  "${PY_VENV_DIR}/bin/pip" install -e "${REPO_DIR}"
else
  "${PY_VENV_DIR}/bin/pip" install -r "${REPO_DIR}/server/requirements.txt"
fi

# Hard cleanup: remove legacy VM OpenEnv service if it exists.
if systemctl list-unit-files | grep -q '^antiatropos-fastapi\.service'; then
  echo "Disabling legacy service antiatropos-fastapi..."
  systemctl disable --now antiatropos-fastapi >/dev/null 2>&1 || true
  rm -f /etc/systemd/system/antiatropos-fastapi.service
fi

cat > /etc/systemd/system/antiatropos-control.service <<EOF
[Unit]
Description=AntiAtropos Droplet Control API
After=network-online.target k3s.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${REPO_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${PY_VENV_DIR}/bin/uvicorn server.local_laptop_control:app --host ${CONTROL_HOST} --port ${CONTROL_PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now antiatropos-control

echo ""
echo "Waiting for control API readiness..."
for _ in {1..30}; do
  if curl -fsS "http://127.0.0.1:${CONTROL_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

PUBLIC_IP="$(curl -fsS https://api.ipify.org 2>/dev/null || true)"
if [[ -z "${PUBLIC_IP}" ]]; then
  PUBLIC_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
fi
PROM_URL_DISPLAY="http://${PUBLIC_IP:-<droplet-ip>}:30090"

echo ""
echo "=== Deploy Complete ==="
echo "Control health:  http://127.0.0.1:${CONTROL_PORT}/health"
echo "Control step:    http://127.0.0.1:${CONTROL_PORT}/step"
echo "Prometheus svc:  kubectl -n ${MONITORING_NAMESPACE} get svc prometheus-server"
echo "Prometheus URL:  ${PROM_URL_DISPLAY}"
echo "Grafana access:  kubectl -n ${MONITORING_NAMESPACE} port-forward svc/grafana 3000:80"
echo ""
echo "Service status command:"
echo "  systemctl status antiatropos-control --no-pager"
echo ""
echo "If needed, edit env and restart control service:"
echo "  ${ENV_FILE}"
echo "  systemctl restart antiatropos-control"
echo ""
echo "Verify remote scaling path:"
echo "  watch -n 1 'kubectl -n prod-sre get deploy -o custom-columns=NAME:.metadata.name,DESIRED:.spec.replicas,READY:.status.readyReplicas'"
