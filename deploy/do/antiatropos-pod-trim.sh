#!/usr/bin/env bash
# antiatropos-pod-trim.sh
# Resets all prod-sre deployments to their minimum replica count
# AND deletes completed/failed/evicted pods to prevent accumulation.
# Installed as a cron job to prevent pod stacking across episodes.
set -euo pipefail

KUBECONFIG="${KUBECONFIG:-/etc/rancher/k3s/k3s.yaml}"
export KUBECONFIG
NAMESPACE="${1:-prod-sre}"
MIN_REPLICAS="${2:-1}"

trimmed=0
while IFS= read -r deploy; do
    current=$(kubectl get deploy "$deploy" -n "$NAMESPACE" -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "0")
    if [[ "$current" -gt "$MIN_REPLICAS" ]]; then
        kubectl scale deploy "$deploy" -n "$NAMESPACE" --replicas="$MIN_REPLICAS" >/dev/null 2>&1
        trimmed=$((trimmed + 1))
    fi
done < <(kubectl get deploy -n "$NAMESPACE" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

# Delete completed (Succeeded), failed, and evicted pods across the namespace.
# These accumulate across episodes and can exhaust node resources
# even after deployments are scaled back down.
deleted=0
for phase in Succeeded Failed; do
    while IFS= read -r pod; do
        [[ -z "$pod" ]] && continue
        kubectl delete pod "$pod" -n "$NAMESPACE" --force --grace-period=0 >/dev/null 2>&1 && deleted=$((deleted + 1))
    done < <(kubectl get pods -n "$NAMESPACE" --field-selector=status.phase=$phase -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)
done

# Also nuke evicted pods (reason=Evicted, phase=Failed is often covered
# above, but some k3s versions keep evicted pods in a weird state).
while IFS= read -r pod; do
    [[ -z "$pod" ]] && continue
    kubectl delete pod "$pod" -n "$NAMESPACE" --force --grace-period=0 >/dev/null 2>&1 && deleted=$((deleted + 1))
done < <(kubectl get pods -n "$NAMESPACE" -o json | \
    grep -l '"reason": "Evicted"' >/dev/null 2>&1 && \
    kubectl get pods -n "$NAMESPACE" -o jsonpath='{range .items[?(@.status.reason=="Evicted")]}{.metadata.name}{"\n"}{end}' 2>/dev/null || true)

if [[ "$trimmed" -gt 0 || "$deleted" -gt 0 ]]; then
    echo "$(date -Iseconds) Trimmed $trimmed deployments to $MIN_REPLICAS replicas, deleted $deleted stale pods in $NAMESPACE"
fi
