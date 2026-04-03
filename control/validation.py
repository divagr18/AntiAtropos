from typing import List, Optional
from pydantic import BaseModel

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
        if valid_targets is not None and target not in valid_targets:
            return False, f"Unknown target node: {target}"

        if action_type == "SHED_LOAD" and target in self.critical_nodes:
            return False, f"Forbidden: Load shedding on critical node {target}."
        
        if action_type in ["SCALE_UP", "SCALE_DOWN"] and parameter < 0.0:
            return False, "Negative scaling parameters are not allowed."
            
        return True, "Success"
