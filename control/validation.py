from typing import List, Optional

class ActionValidator:
    """
    Validates SRE actions to ensure they stay within safety boundaries.
    Prevents destructive operations like 100% shedding on critical nodes.
    """
    def __init__(self, critical_nodes: Optional[List[str]] = None):
        self.critical_nodes = critical_nodes or ["node-0", "node-1", "node-2"]

    def validate(self, action_type: str, target: str, parameter: float, valid_targets: Optional[List[str]] = None) -> (bool, str):
        """
        Returns (is_valid, error_message).
        """
        if hasattr(action_type, "value"):
            action = str(action_type.value)
        else:
            action = str(action_type)

        if valid_targets is not None and target not in valid_targets:
            return False, f"Unknown target node: {target}"

        if action == "SHED_LOAD" and target in self.critical_nodes:
            return False, f"Forbidden: Load shedding on critical node {target}."

        if action in ["SCALE_UP", "SCALE_DOWN"]:
            if parameter < 0.0:
                return False, "Negative scaling parameters are not allowed."
            if parameter > 10.0:
                return False, "Scaling parameter must be <= 10.0."

        if action in ["REROUTE_TRAFFIC", "SHED_LOAD"] and not (0.0 <= parameter <= 1.0):
            return False, f"{action} parameter must be in [0.0, 1.0]."

        if action == "NO_OP" and parameter != 0.0:
            return False, "NO_OP requires parameter=0.0."
            
        return True, "Success"
