#!/usr/bin/env bash
set -euo pipefail

# Removes legacy VM OpenEnv service path.
# This keeps droplet runtime focused on control API + observability only.

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run as root: sudo bash deploy/do/uninstall-legacy-openenv.sh"
  exit 1
fi

if systemctl list-unit-files | grep -q '^antiatropos-fastapi\.service'; then
  echo "Stopping and disabling antiatropos-fastapi..."
  systemctl disable --now antiatropos-fastapi >/dev/null 2>&1 || true
else
  echo "antiatropos-fastapi service not registered."
fi

if [[ -f /etc/systemd/system/antiatropos-fastapi.service ]]; then
  rm -f /etc/systemd/system/antiatropos-fastapi.service
  echo "Removed /etc/systemd/system/antiatropos-fastapi.service"
fi

systemctl daemon-reload
echo "Legacy VM OpenEnv service cleanup complete."
