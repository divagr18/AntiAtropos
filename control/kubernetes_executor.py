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
        self._live_supported_actions = {"NO_OP", "SCALE_UP", "SCALE_DOWN"}

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

        apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=namespace,
            body={"spec": {"replicas": desired}},
        )

        return (
            f"Ack: {action_type} for {target} - deployment {deployment_name} "
            f"in namespace {namespace} scaled {current}->{desired}"
        )

    def _remote_execution(self, action: str, target: str, parameter: float) -> str:
        """
        Delegate action execution to a remote FastAPI control plane.

        Expected remote endpoint contract:
          - POST /step
          - Request: {action_type, target_node_id, parameter}
          - Success response includes ack_status and starts with "Ack:"
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

        try:
            response = requests.post(endpoint, json=payload, timeout=self.remote_timeout_s)
        except requests.RequestException as exc:
            raise RuntimeError(f"Remote control-plane request failed: {exc}") from exc

        if response.status_code == 422:
            # OpenEnv server.app expects {"action": {...}} shape on /step.
            try:
                body = response.json()
                detail = str(body.get("detail", body))
            except Exception:
                detail = response.text.strip()
            if "body" in detail and "action" in detail:
                try:
                    response = requests.post(
                        endpoint,
                        json={"action": action_payload},
                        timeout=self.remote_timeout_s,
                    )
                except requests.RequestException as exc:
                    raise RuntimeError(f"Remote control-plane retry failed: {exc}") from exc

        if response.status_code >= 400:
            detail = ""
            try:
                body = response.json()
                detail = str(body.get("detail", body))
            except Exception:
                detail = response.text.strip()
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
