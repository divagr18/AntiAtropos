#!/usr/bin/env bash
set -euo pipefail

FASTAPI_PID=""
PROMETHEUS_PID=""
GRAFANA_PID=""
NGINX_PID=""
MONITOR_PID=""

cleanup() {
    for pid in "${MONITOR_PID}" "${NGINX_PID}" "${GRAFANA_PID}" "${PROMETHEUS_PID}" "${FASTAPI_PID}"; do
        if [[ -n "${pid}" ]]; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup INT TERM EXIT

cd /app

# Source HF Spaces live-mode config if present (overrides Dockerfile defaults)
if [[ -f /app/.env.hf ]]; then
  echo "Loading .env.hf..."
  set -a
  # shellcheck source=/dev/null
  source /app/.env.hf
  set +a
fi

uvicorn server.app:app --host 127.0.0.1 --port 8000 &
FASTAPI_PID=$!

/opt/prometheus/prometheus \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/tmp/prometheus-data \
    --web.listen-address=127.0.0.1:9090 \
    --web.route-prefix=/prometheus \
    &
PROMETHEUS_PID=$!

/opt/grafana/bin/grafana-server \
    --homepath /opt/grafana \
    --config /etc/grafana/grafana.ini \
    cfg:default.paths.data=/var/lib/grafana \
    cfg:default.paths.logs=/var/log/grafana \
    cfg:default.paths.plugins=/var/lib/grafana/plugins \
    cfg:default.paths.provisioning=/etc/grafana/provisioning \
    &
GRAFANA_PID=$!

nginx -g "daemon off;" &
NGINX_PID=$!

monitor_children() {
    while true; do
        for pid in "${FASTAPI_PID}" "${PROMETHEUS_PID}" "${GRAFANA_PID}"; do
            if ! kill -0 "${pid}" 2>/dev/null; then
                echo "A backend service exited unexpectedly." >&2
                kill "${NGINX_PID}" 2>/dev/null || true
                exit 1
            fi
        done
        sleep 2
    done
}

monitor_children &
MONITOR_PID=$!

wait "${NGINX_PID}"
