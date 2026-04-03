import os
import json
from typing import Optional

class KubernetesExecutor:
    """
    Executes high-level SRE actions on a Kubernetes cluster.
    Provides a safe layer between SREAgent and actual infrastructure.
    """
    def __init__(self, kubeconfig: Optional[str] = None):
        # Use provided path or env var, defaulting to mock if neither is found
        self.kubeconfig = kubeconfig or os.getenv("KUBECONFIG")
        self.is_mock = not self.kubeconfig or self.kubeconfig.lower() == "mock"
        self.namespace = os.getenv("ANTIATROPOS_K8S_NAMESPACE", "default")
        self.deployment_prefix = os.getenv("ANTIATROPOS_DEPLOYMENT_PREFIX", "")
        self.min_replicas = int(os.getenv("ANTIATROPOS_MIN_REPLICAS", "1"))
        self.max_replicas = int(os.getenv("ANTIATROPOS_MAX_REPLICAS", "20"))
        self.scale_step = int(os.getenv("ANTIATROPOS_SCALE_STEP", "3"))
        self._apps_v1_api = None
        self._node_deployment_map = self._load_node_deployment_map()
        
    def execute(self, action_type: str, target: str, parameter: float) -> str:
        """
        Translates SRE actions to Kube requests (ScaleDeployment, PatchIngress, etc.)
        """
        if self.is_mock:
            return self._mock_execution(action_type, target, parameter)
            
        try:
            return self._real_execution(action_type, target, parameter)
        except Exception as e:
            return f"Error: Failed to execute {action_type} on {target}: {str(e)}"

    def _real_execution(self, action_type: str, target: str, parameter: float) -> str:
        """Execute bounded actions on a Kubernetes cluster."""
        if action_type == "NO_OP":
            return "Ack: NO_OP - no cluster mutation"

        if action_type in ("SCALE_UP", "SCALE_DOWN"):
            return self._scale_deployment(action_type, target, parameter)

        return f"Rejected: {action_type} is not enabled for live Kubernetes execution"

    def _mock_execution(self, action_type: str, target: str, parameter: float) -> str:
        """Returns mock acknowledgement for actions."""
        # TODO: Add realistic latency simulation for K8s control plane
        return f"Ack: {action_type} for {target} with value {parameter} - Status: Applied"

    def _scale_deployment(self, action_type: str, target: str, parameter: float) -> str:
        deployment_name = self._resolve_deployment_name(target)
        apps_v1 = self._get_apps_v1_api()

        scale_obj = apps_v1.read_namespaced_deployment_scale(
            name=deployment_name,
            namespace=self.namespace,
        )

        current = int(scale_obj.spec.replicas or self.min_replicas)
        delta = max(1, int(float(parameter) * self.scale_step))
        if action_type == "SCALE_UP":
            desired = min(self.max_replicas, current + delta)
        else:
            desired = max(self.min_replicas, current - delta)

        if desired == current:
            return (
                f"Ack: {action_type} for {target} - replicas unchanged at {current} "
                f"(bounds {self.min_replicas}-{self.max_replicas})"
            )

        apps_v1.patch_namespaced_deployment_scale(
            name=deployment_name,
            namespace=self.namespace,
            body={"spec": {"replicas": desired}},
        )

        return f"Ack: {action_type} for {target} - deployment {deployment_name} scaled {current}->{desired}"

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

    def _load_node_deployment_map(self) -> dict[str, str]:
        raw = os.getenv("ANTIATROPOS_NODE_DEPLOYMENT_MAP", "")
        if not raw:
            return {}
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except json.JSONDecodeError:
            return {}
        return {}

    def _resolve_deployment_name(self, target: str) -> str:
        if target in self._node_deployment_map:
            return self._node_deployment_map[target]
        return f"{self.deployment_prefix}{target}"
