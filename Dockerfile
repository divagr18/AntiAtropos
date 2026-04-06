FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROMETHEUS_VERSION=3.5.1 \
    GRAFANA_VERSION=12.3.1 \
    PROMETHEUS_ARCH=linux-amd64 \
    GRAFANA_ARCH=linux-amd64

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
