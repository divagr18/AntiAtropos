FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROMETHEUS_VERSION=3.5.1 \
    GRAFANA_VERSION=12.3.1 \
    PROMETHEUS_ARCH=linux-amd64 \
    GRAFANA_ARCH=linux-amd64 \
    ANTIATROPOS_ENV_MODE=live \
    ANTIATROPOS_REWARD_OUTPUT_MODE=normalized \
    ANTIATROPOS_CONTROL_TIMEOUT_S=8.0 \
    ANTIATROPOS_PROM_TIMEOUT_S=5.0 \
    ANTIATROPOS_STRICT_REAL=false \
    ANTIATROPOS_METRIC_AGGREGATION=sum \
    ANTIATROPOS_K8S_NAMESPACE=prod-sre \
    ANTIATROPOS_MIN_REPLICAS=1 \
    ANTIATROPOS_SCALE_STEP=3 \
    ANTIATROPOS_CONTROL_PLANE_URL=http://206.189.136.21:8010 \
    PROMETHEUS_URL=http://206.189.136.21:30090 \
    ANTIATROPOS_WORKLOAD_MAP={"node-0":{"deployment":"payments","namespace":"prod-sre"},"node-1":{"deployment":"checkout","namespace":"prod-sre"},"node-2":{"deployment":"catalog","namespace":"prod-sre"},"node-3":{"deployment":"cart","namespace":"prod-sre"},"node-4":{"deployment":"auth","namespace":"prod-sre"}} \
    ANTIATROPOS_LABEL_NODE_MAP={"payments":"node-0","checkout":"node-1","catalog":"node-2","cart":"node-3","auth":"node-4"} \
    ANTIATROPOS_PROM_QUERY_REQUEST_RATE="sum(rate(http_requests_total[1m])) by (node_id)" \
    ANTIATROPOS_PROM_QUERY_LATENCY_MS="histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (node_id, le)) * 1000" \
    ANTIATROPOS_PROM_QUERY_ERROR_RATE="sum(rate(http_requests_total{status=~\"5..\"}[1m])) by (node_id) / clamp_min(sum(rate(http_requests_total[1m])) by (node_id), 1)" \
    ANTIATROPOS_PROM_QUERY_CPU="avg(rate(container_cpu_usage_seconds_total[1m])) by (node_id)" \
    ANTIATROPOS_PROM_QUERY_QUEUE_DEPTH="avg(queue_depth) by (node_id)" 

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    fontconfig \
    libfontconfig1 \
    libfreetype6 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    nginx \
    tar \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip setuptools wheel \
    && pip install -e .

RUN set -eux; \
    mkdir -p /opt/prometheus /opt/grafana /var/lib/grafana /var/log/grafana /var/lib/prometheus /etc/grafana /etc/prometheus; \
    curl -fsSL -o /tmp/prometheus.tar.gz "https://github.com/prometheus/prometheus/releases/download/v${PROMETHEUS_VERSION}/prometheus-${PROMETHEUS_VERSION}.${PROMETHEUS_ARCH}.tar.gz"; \
    tar -xzf /tmp/prometheus.tar.gz -C /tmp; \
    PROMETHEUS_DIR="$(find /tmp -maxdepth 1 -type d -name "prometheus-*" | head -n 1)"; \
    cp -a "${PROMETHEUS_DIR}/." /opt/prometheus/; \
    curl -fsSL -o /tmp/grafana.tar.gz "https://dl.grafana.com/oss/release/grafana-${GRAFANA_VERSION}.linux-amd64.tar.gz"; \
    tar -xzf /tmp/grafana.tar.gz -C /tmp; \
    GRAFANA_DIR="$(find /tmp -maxdepth 1 -type d -name "grafana-*" | head -n 1)"; \
    cp -a "${GRAFANA_DIR}/." /opt/grafana/; \
    rm -rf /tmp/prometheus* /tmp/grafana*; \
    chmod +x /app/deploy/entrypoint.sh

COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY deploy/prometheus.yml /etc/prometheus/prometheus.yml
COPY deploy/index.html /var/www/html/index.html
COPY deploy/grafana/grafana.ini /etc/grafana/grafana.ini
COPY deploy/grafana/provisioning /etc/grafana/provisioning

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:7860/health || exit 1

CMD ["/app/deploy/entrypoint.sh"]
