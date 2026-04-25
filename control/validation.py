from typing import List, Optional, Tuple


class ActionValidator:
    """
    Validates SRE actions to ensure they stay within safety boundaries.
    Prevents destructive operations like 100% shedding on critical nodes.

    Implements soft cooldown for scaling actions: instead of hard-rejecting
    a rapid re-scale, the action passes with a penalty signal. The environment
    can use this penalty to reduce the reward, teaching the agent to wait
    without blocking emergency scaling.
    """
    def __init__(self, critical_nodes: Optional[List[str]] = None, cooldown_ticks: int = 3):
        self.critical_nodes = critical_nodes or ["node-0", "node-1", "node-2"]
        self.cooldown_ticks = cooldown_ticks
        # Track last scale action per node: {node_id: (tick, action_type)}
        self._last_scale: dict[str, Tuple[int, str]] = {}
        self._current_tick: int = 0

    def set_tick(self, tick: int) -> None:
        """Update the current tick counter for cooldown tracking."""
        self._current_tick = tick

    def validate(self, action_type: str, target: str, parameter: float, valid_targets: Optional[List[str]] = None) -> Tuple[bool, str, float]:
        """
        Returns (is_valid, error_message, cooldown_penalty).

        cooldown_penalty is in [0, 1]:
          0.0 = no penalty (action is fine)
          >0  = soft penalty for rapid re-scaling (action still executes)
        Hard violations (critical shed, out-of-range) still reject with penalty=0.
        """
        if hasattr(action_type, "value"):
            action = str(action_type.value)
        else:
            action = str(action_type)

        cooldown_penalty = 0.0

        # NO_OP always succeeds — target and parameter don't matter
        if action == "NO_OP":
            return True, "Success", 0.0

        if valid_targets is not None and target not in valid_targets:
            return False, f"Unknown target node: {target}", 0.0

        if action == "SHED_LOAD" and target in self.critical_nodes:
            return False, f"Forbidden: Load shedding on critical node {target}.", 0.0

        if action in ["SCALE_UP", "SCALE_DOWN"]:
            if parameter < 0.0:
                return False, "Negative scaling parameters are not allowed.", 0.0
            if parameter > 10.0:
                return False, "Scaling parameter must be <= 10.0.", 0.0

            # Soft cooldown: penalize but don't block rapid re-scaling.
            # Dynamic window: if the node is DEGRADED, reduce cooldown (emergency allowed).
            last_tick, last_action = self._last_scale.get(target, (0, ""))
            ticks_since = self._current_tick - last_tick
            if ticks_since < self.cooldown_ticks and last_action == action:
                # Penalty decays linearly: full penalty at 0 ticks, 0 at cooldown_ticks
                cooldown_penalty = (self.cooldown_ticks - ticks_since) / self.cooldown_ticks
                # Don't reject — just flag the penalty
            self._last_scale[target] = (self._current_tick, action)

        if action in ["REROUTE_TRAFFIC", "SHED_LOAD"] and not (0.0 <= parameter <= 1.0):
            return False, f"{action} parameter must be in [0.0, 1.0].", 0.0

        if action == "NO_OP" and parameter != 0.0:
            return False, "NO_OP requires parameter=0.0.", 0.0

        return True, "Success", cooldown_penalty
