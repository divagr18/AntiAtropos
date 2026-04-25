import os
import json
import time
import logging
import requests
from uuid import uuid4
from typing import Optional

logger = logging.getLogger("kubernetes_executor")

class KubernetesExecutor:
    """
    Executes high-level SRE actions on a Kubernetes cluster.
    Provides a safe layer between SREAgent and actual infrastructure.
    """
    def __init__(self, kubeconfig: Optional[str] = None):
        # Use provided path or env var, defaulting to mock if neither is found
        self.kubeconfig = kubeconfig or os.getenv("KUBECONFIG")
        self.remote_control_url = os.getenv("ANTIATROPOS_CONTROL_PLANE_URL", "").strip().rstrip("/")
        self.remote_timeout_s = float(os.getenv("ANTIATROPOS_CONTROL_TIMEOUT_S", "5.0"))
        self.remote_retry_count = int(os.getenv("ANTIATROPOS_CONTROL_RETRY_COUNT", "2"))
        self.remote_retry_backoff_s = float(os.getenv("ANTIATROPOS_CONTROL_RETRY_BACKOFF_S", "0.25"))
        self.is_mock = (
            not self.remote_control_url
            and (not self.kubeconfig or self.kubeconfig.lower() == "mock")
        )
        self.namespace = os.getenv("ANTIATROPOS_K8S_NAMESPACE", "default")
        self.min_replicas = int(os.getenv("ANTIATROPOS_MIN_REPLICAS", "1"))
        self.max_replicas = self._parse_max_replicas(os.getenv("ANTIATROPOS_MAX_REPLICAS"))
        self.scale_step = int(os.getenv("ANTIATROPOS_SCALE_STEP", "3"))
        self._apps_v1_api = None
        self._node_workload_map = self._load_node_workload_map()
        self._live_supported_actions = {"NO_OP", "SCALE_UP", "SCALE_DOWN", "REROUTE_TRAFFIC", "SHED_LOAD"}
        self.k8s_retry_count = int(os.getenv("ANTIATROPOS_K8S_RETRY_COUNT", "2"))
        self.k8s_retry_backoff_s = float(os.getenv("ANTIATROPOS_K8S_RETRY_BACKOFF_S", "0.2"))

    @staticmethod
    def _parse_max_replicas(raw: Optional[str]) -> Optional[int]:
        """
        Parse optional max replicas.

        Returns:
          - int when a positive explicit cap is provided
          - None when scale-up should be unbounded
        """
        if raw is None:
            return None
        value = str(raw).strip().lower()
        if value in ("", "none", "unbounded", "inf", "infinite"):
            return None
        try:
            parsed = int(value)
        except ValueError:
            return None
        if parsed <= 0:
            return None
        return parsed

    @staticmethod
    def _normalize_action_type(action_type) -> str:
        if hasattr(action_type, "value"):
            return str(action_type.value)
        return str(action_type)
        
    def execute(self, action_type: str, target: str, parameter: float) -> str:
        """
        Translates SRE actions to Kube requests (ScaleDeployment, PatchIngress, etc.)
        """
        return self.execute_with_metadata(action_type, target, parameter)["ack_status"]

    def execute_with_metadata(self, action_type: str, target: str, parameter: float) -> dict:
        """
        Execute action and return acknowledgement plus executor metadata.
        """
        action_id = str(uuid4())
        started = time.perf_counter()
        ack_status = ""
        error_code = ""

        if self.is_mock:
            ack_status = self._mock_execution(action_type, target, parameter)
        else:
            try:
                ack_status = self._real_execution(action_type, target, parameter)
            except Exception as e:
                logger.error(f"Execution failed for {action_type} on {target}: {str(e)}")
                ack_status = f"Error: Failed to execute {action_type} on {target}: {str(e)}"
                error_code = "EXECUTION_ERROR"

        if ack_status.startswith("Rejected:") and not error_code:
            error_code = "REJECTED_ACTION"
        elif ack_status.startswith("Error:") and not error_code:
            error_code = "EXECUTION_ERROR"

        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "action_id": action_id,
            "ack_status": ack_status,
            "executor_latency_ms": latency_ms,
            "executor_error_code": error_code,
        }

    def live_enabled_actions(self) -> set[str]:
        """Action types that are actually executable in real live mode."""
        if self.is_mock:
            return {"NO_OP"}
        return set(self._live_supported_actions)

    def live_capability_error(self, action_type: str) -> Optional[str]:
        """Returns reason when action is not runnable in live mode, else None."""
        action = self._normalize_action_type(action_type)
        if action not in self.live_enabled_actions():
            if self.is_mock:
                return (
                    f"Live mode rejected {action}: no real Kubernetes executor is configured "
                    "(set KUBECONFIG and ANTIATROPOS_WORKLOAD_MAP)."
                )
            return f"Live mode rejected {action}: no executor is enabled for this action."
        return None

    def _real_execution(self, action_type: str, target: str, parameter: float) -> str:
        """Execute bounded actions on a Kubernetes cluster."""
        action = self._normalize_action_type(action_type)

        if self.remote_control_url:
            return self._remote_execution(action, target, parameter)

        if action == "NO_OP":
            return "Ack: NO_OP - no cluster mutation"

        if action in ("SCALE_UP", "SCALE_DOWN"):
            return self._scale_deployment(action, target, parameter)

        if action == "REROUTE_TRAFFIC":
            return self._reroute_traffic(target, parameter)

        if action == "SHED_LOAD":
            return self._shed_load(target, parameter)

        return f"Rejected: {action} is not enabled for live Kubernetes execution"

    def _mock_execution(self, action_type: str, target: str, parameter: float) -> str:
        """Returns mock acknowledgement for actions."""
        # TODO: Add realistic latency simulation for K8s control plane
        action = self._normalize_action_type(action_type)
        return f"Ack: {action} for {target} with value {parameter} - Status: Applied"

    def _scale_deployment(self, action_type: str, target: str, parameter: float) -> str:
        namespace, deployment_name = self._resolve_workload_target(target)
        apps_v1 = self._get_apps_v1_api()

        scale_obj = apps_v1.read_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
        )

        current = int(scale_obj.spec.replicas or self.min_replicas)
        delta = max(1, int(float(parameter) * self.scale_step))
        if action_type == "SCALE_UP":
            if self.max_replicas is None:
                desired = current + delta
            else:
                desired = min(self.max_replicas, current + delta)
        else:
            desired = max(self.min_replicas, current - delta)

        if desired == current:
            upper = "unbounded" if self.max_replicas is None else str(self.max_replicas)
            return (
                f"Ack: {action_type} for {target} - replicas unchanged at {current} "
                f"(bounds {self.min_replicas}-{upper})"
            )

        self._patch_deployment_scale_with_retry(
            apps_v1=apps_v1,
            deployment_name=deployment_name,
            namespace=namespace,
            desired=desired,
        )

        return (
            f"Ack: {action_type} for {target} - deployment {deployment_name} "
            f"in namespace {namespace} scaled {current}->{desired}"
        )

    def _reroute_traffic(self, target: str, parameter: float) -> str:
        """
        Live implementation of REROUTE_TRAFFIC.

        Shifts capacity away from the target node onto healthy peers by:
          1. Scaling DOWN the target deployment by parameter * current_replicas
             (min: min_replicas, so at least 1 replica remains).
          2. Distributing the shed replicas equally across all other healthy
             deployments as a SCALE_UP (best-effort, capped at max_replicas).

        This reuses the same patch_namespaced_deployment_scale mechanism as
        SCALE_UP/SCALE_DOWN, ensuring observable cluster mutations.
        """
        namespace, deployment_name = self._resolve_workload_target(target)
        apps_v1 = self._get_apps_v1_api()

        scale_obj = apps_v1.read_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
        )
        current_target = int(scale_obj.spec.replicas or self.min_replicas)

        frac = min(1.0, max(0.0, float(parameter)))
        delta = max(1, int(current_target * frac))
        new_target = max(self.min_replicas, current_target - delta)

        messages: list[str] = []

        if new_target != current_target:
            self._patch_deployment_scale_with_retry(
                apps_v1=apps_v1,
                deployment_name=deployment_name,
                namespace=namespace,
                desired=new_target,
            )
            messages.append(
                f"target {deployment_name} scaled {current_target}->{new_target}"
            )
        else:
            messages.append(
                f"target {deployment_name} unchanged at {current_target} (already at min)"
            )

        # Redistribute shed replicas across healthy peers (best-effort)
        healthy_peers = [
            (peer_id, peer_info)
            for peer_id, peer_info in self._node_workload_map.items()
            if peer_id != target
        ]

        if healthy_peers and delta > 0:
            peer_delta = max(1, delta // len(healthy_peers))
            scaled_peers = 0
            for peer_id, peer_info in healthy_peers:
                peer_deployment = peer_info["deployment"]
                peer_ns = peer_info.get("namespace", self.namespace)
                try:
                    peer_scale = apps_v1.read_namespaced_deployment_scale(
                        name=peer_deployment, namespace=peer_ns,
                    )
                    peer_current = int(peer_scale.spec.replicas or self.min_replicas)
                    peer_desired = peer_current + peer_delta
                    if self.max_replicas is not None:
                        peer_desired = min(self.max_replicas, peer_desired)
                    if peer_desired != peer_current:
                        self._patch_deployment_scale_with_retry(
                            apps_v1=apps_v1,
                            deployment_name=peer_deployment,
                            namespace=peer_ns,
                            desired=peer_desired,
                        )
                        scaled_peers += 1
                except Exception:
                    pass  # best-effort for peers

            if scaled_peers:
                messages.append(
                    f"redistributed +{peer_delta} replicas to {scaled_peers} peer(s)"
                )

        return (
            f"Ack: REROUTE_TRAFFIC for {target} (frac={frac:.2f}) - "
            + "; ".join(messages)
        )

    def _shed_load(self, target: str, parameter: float) -> str:
        """
        Live implementation of SHED_LOAD.

        Drops a fraction of capacity from the target node by scaling DOWN
        its deployment.  The shed fraction decays over time in the simulator,
        but in live mode the replica reduction is permanent until the agent
        explicitly scales back up.

        Critical nodes (node-0, node-1, node-2) are guarded by validation
        before this method is ever called.
        """
        namespace, deployment_name = self._resolve_workload_target(target)
        apps_v1 = self._get_apps_v1_api()

        scale_obj = apps_v1.read_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
        )
        current = int(scale_obj.spec.replicas or self.min_replicas)

        frac = min(1.0, max(0.0, float(parameter)))
        delta = max(1, int(current * frac))
        desired = max(self.min_replicas, current - delta)

        if desired == current:
            return (
                f"Ack: SHED_LOAD for {target} - replicas unchanged at {current} "
                f"(already at min_replicas={self.min_replicas})"
            )

        self._patch_deployment_scale_with_retry(
            apps_v1=apps_v1,
            deployment_name=deployment_name,
            namespace=namespace,
            desired=desired,
        )

        return (
            f"Ack: SHED_LOAD for {target} - deployment {deployment_name} "
            f"in namespace {namespace} scaled {current}->{desired} "
            f"(shed {delta} replicas, frac={frac:.2f})"
        )

    def _patch_deployment_scale_with_retry(self, apps_v1, deployment_name: str, namespace: str, desired: int) -> None:
        """
        Patch deployment replicas with retries for transient API server errors.
        """
        from kubernetes.client.rest import ApiException

        max_attempts = max(1, self.k8s_retry_count + 1)
        for attempt in range(1, max_attempts + 1):
            try:
                apps_v1.patch_namespaced_deployment_scale(
                    name=deployment_name,
                    namespace=namespace,
                    body={"spec": {"replicas": desired}},
                )
                return
            except ApiException as exc:
                retryable = exc.status in (409, 429, 500, 502, 503, 504)
                if (not retryable) or attempt >= max_attempts:
                    raise
                sleep_s = self.k8s_retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Retrying deployment scale patch after ApiException status=%s attempt=%s/%s",
                    exc.status,
                    attempt,
                    max_attempts,
                )
                time.sleep(sleep_s)

    def _remote_execution(self, action: str, target: str, parameter: float) -> str:
        """
        Delegate action execution to a remote FastAPI control plane.

        Expected remote endpoint contract:
          - POST /step
          - Request: {action_type, target_node_id, parameter}
          - Success response includes ack_status and starts with "Ack:"

        This contract matches server.local_laptop_control and is the only
        supported remote control-plane format.
        """
        if not self.remote_control_url:
            raise ValueError("ANTIATROPOS_CONTROL_PLANE_URL is not configured")

        endpoint = f"{self.remote_control_url}/step"
        action_payload = {
            "action_type": action,
            "target_node_id": target,
            "parameter": float(parameter),
        }
        payload = action_payload

        response = self._post_with_retry(endpoint=endpoint, payload=payload)

        if response.status_code >= 400:
            detail = ""
            try:
                body = response.json()
                detail = str(body.get("detail", body))
            except Exception:
                detail = response.text.strip()
            if response.status_code == 422 and "action" in detail:
                detail = (
                    f"{detail}. Expected lightweight control-plane contract at "
                    f"{endpoint}: "
                    '{"action_type":"SCALE_UP","target_node_id":"node-0","parameter":1.0}'
                )
            raise RuntimeError(
                f"Remote control-plane rejected action ({response.status_code}): {detail}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise RuntimeError("Remote control-plane returned non-JSON response") from exc

        ack = str(data.get("ack_status", "")).strip()
        if not ack:
            action_id = str(data.get("action_id", "")).strip() or "remote"
            return f"Ack: {action} for {target} via remote control-plane ({action_id})"
        return ack

    def _post_with_retry(self, endpoint: str, payload: dict) -> requests.Response:
        """
        POST helper with retries for transient HTTP/network failures.
        """
        max_attempts = max(1, self.remote_retry_count + 1)
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.post(endpoint, json=payload, timeout=self.remote_timeout_s)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    break
                sleep_s = self.remote_retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Retrying remote control-plane POST after network error attempt=%s/%s: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
                time.sleep(sleep_s)
                continue

            if response.status_code >= 500 and attempt < max_attempts:
                sleep_s = self.remote_retry_backoff_s * (2 ** (attempt - 1))
                logger.warning(
                    "Retrying remote control-plane POST after HTTP %s attempt=%s/%s",
                    response.status_code,
                    attempt,
                    max_attempts,
                )
                time.sleep(sleep_s)
                continue

            return response

        if last_exc is not None:
            raise RuntimeError(f"Remote control-plane request failed: {last_exc}") from last_exc
        raise RuntimeError("Remote control-plane request failed after retries")

    def _get_apps_v1_api(self):
        if self._apps_v1_api is not None:
            return self._apps_v1_api

        from kubernetes import client, config

        if self.kubeconfig and self.kubeconfig.lower() not in ("mock", ""):
            config.load_kube_config(config_file=self.kubeconfig)
        else:
            config.load_incluster_config()

        self._apps_v1_api = client.AppsV1Api()
        return self._apps_v1_api

    def _load_node_workload_map(self) -> dict[str, dict[str, str]]:
        """
        Load node->workload mapping.

        Preferred format (ANTIATROPOS_WORKLOAD_MAP):
        {
          "node-0": {"deployment": "payments", "namespace": "prod-sre"},
          "node-1": {"deployment": "checkout"}
        }

        Legacy fallback (ANTIATROPOS_NODE_DEPLOYMENT_MAP):
        {
          "node-0": "payments",
          "node-1": "checkout"
        }
        """
        raw = os.getenv("ANTIATROPOS_WORKLOAD_MAP", "")
        if raw:
            parsed = self._parse_json_mapping(raw)
            if parsed is not None:
                return parsed

        legacy_raw = os.getenv("ANTIATROPOS_NODE_DEPLOYMENT_MAP", "")
        if legacy_raw:
            legacy = self._parse_legacy_mapping(legacy_raw)
            if legacy is not None:
                return legacy

        return {}

    def _parse_json_mapping(self, raw: str) -> Optional[dict[str, dict[str, str]]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        out: dict[str, dict[str, str]] = {}
        for node_id, workload in data.items():
            if not isinstance(workload, dict):
                return None
            deployment = workload.get("deployment")
            if not deployment:
                return None
            namespace = workload.get("namespace", self.namespace)
            out[str(node_id)] = {
                "deployment": str(deployment),
                "namespace": str(namespace),
            }
        return out

    def _parse_legacy_mapping(self, raw: str) -> Optional[dict[str, dict[str, str]]]:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        out: dict[str, dict[str, str]] = {}
        for node_id, deployment in data.items():
            if not deployment:
                return None
            out[str(node_id)] = {
                "deployment": str(deployment),
                "namespace": self.namespace,
            }
        return out

    def _resolve_workload_target(self, target: str) -> tuple[str, str]:
        if target not in self._node_workload_map:
            raise ValueError(
                f"Missing workload mapping for target '{target}'. "
                "Set ANTIATROPOS_WORKLOAD_MAP with node->deployment bindings."
            )

        workload = self._node_workload_map[target]
        return workload["namespace"], workload["deployment"]
