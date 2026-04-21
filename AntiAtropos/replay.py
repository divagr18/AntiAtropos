"""
AntiAtropos Episode Replay Buffer.

Stores episode trajectories for few-shot demonstrations during inference.
Uses summarization/compression to keep context window manageable:
- Only stores key transition windows (action, reward spike, SLA violation)
- Compresses long stable stretches into single summary lines
- Caps total demonstration size to avoid LLM context overflow
"""

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Transition:
    """A single step in an episode trajectory."""
    step: int
    action_type: str
    target_node_id: str
    parameter: float
    reward: float
    avg_latency_norm: float
    error_rate: float
    queue_backlog_norm: float
    sla_violation: bool


@dataclass
class EpisodeTrajectory:
    """A compressed episode trajectory for few-shot prompting."""
    task_id: str
    score: float
    # Full trajectory is NOT stored — only key transitions
    key_transitions: List[Transition] = field(default_factory=list)
    total_steps: int = 0
    final_sla_violations: int = 0
    final_invalid_actions: int = 0

    def to_prompt_lines(self, max_lines: int = 8) -> List[str]:
        """Convert to concise prompt lines, capped at max_lines.

        Summarization strategy:
        1. Always include first action (shows opening strategy)
        2. Always include highest-reward action (shows what worked)
        3. Always include last action (shows closing strategy)
        4. Fill remaining with transitions near SLA violations
        5. If still under max_lines, add evenly-spaced transitions
        """
        if not self.key_transitions:
            return []

        lines: List[str] = []
        selected: List[Transition] = []

        # Always take first
        selected.append(self.key_transitions[0])

        # Always take highest-reward
        best = max(self.key_transitions, key=lambda t: t.reward)
        if best not in selected:
            selected.append(best)

        # Always take last
        last = self.key_transitions[-1]
        if last not in selected:
            selected.append(last)

        # Add transitions near SLA violations (up to 2)
        violation_trans = [t for t in self.key_transitions if t.sla_violation and t not in selected]
        for vt in violation_trans[:2]:
            selected.append(vt)

        # Fill with evenly-spaced transitions
        remaining = max_lines - len(selected)
        if remaining > 0 and len(self.key_transitions) > len(selected):
            stride = max(1, len(self.key_transitions) // (remaining + 1))
            for i in range(stride, len(self.key_transitions), stride):
                if self.key_transitions[i] not in selected and remaining > 0:
                    selected.append(self.key_transitions[i])
                    remaining -= 1

        # Sort by step and format
        selected.sort(key=lambda t: t.step)
        for t in selected[:max_lines]:
            action_str = f'{{"action_type":"{t.action_type}","target_node_id":"{t.target_node_id}","parameter":{t.parameter:.2f}}}'
            lines.append(f"Step {t.step}: {action_str} reward={t.reward:.2f}")

        # Add summary
        lines.append(
            f"[Episode summary: score={self.score:.2f}, "
            f"steps={self.total_steps}, "
            f"SLA_violations={self.final_sla_violations}]"
        )
        return lines


class EpisodeReplayBuffer:
    """
    Rolling buffer of episode trajectories for few-shot learning.

    Addresses context explosion by:
    1. Storing only compressed trajectories (key transitions, not full)
    2. Capping demonstration size at MAX_DEMO_LINES per prompt inclusion
    3. Sampling at most MAX_DEMOS_PER_PROMPT trajectories
    """

    MAX_DEMO_LINES: int = 8  # Max lines per trajectory in prompt
    MAX_DEMOS_PER_PROMPT: int = 2  # Max trajectories included in prompt

    def __init__(self, max_episodes: int = 50):
        self._positive: deque[EpisodeTrajectory] = deque(maxlen=max_episodes)
        self._negative: deque[EpisodeTrajectory] = deque(maxlen=max_episodes)

    def store(self, trajectory: EpisodeTrajectory, score: float) -> None:
        """Store an episode trajectory, categorized by score."""
        if score >= 0.55:
            self._positive.append(trajectory)
        elif score < 0.3:
            self._negative.append(trajectory)

    def sample_demonstrations(self, n: Optional[int] = None) -> List[EpisodeTrajectory]:
        """Sample n positive episodes for few-shot prompting."""
        if n is None:
            n = self.MAX_DEMOS_PER_PROMPT
        if not self._positive:
            return []
        return random.sample(list(self._positive), min(n, len(self._positive)))

    def format_demonstrations(self) -> str:
        """Format sampled demonstrations into a prompt-ready string.

        Returns empty string if no demonstrations available.
        Total output is bounded by MAX_DEMO_LINES * MAX_DEMOS_PER_PROMPT.
        """
        demos = self.sample_demonstrations()
        if not demos:
            return ""

        parts = []
        for i, demo in enumerate(demos):
            lines = demo.to_prompt_lines(max_lines=self.MAX_DEMO_LINES)
            if lines:
                parts.append(f"Example {i+1} (task={demo.task_id}):")
                parts.extend(lines)

        if not parts:
            return ""

        return "Successful episode examples:\n" + "\n".join(parts)


def compress_trajectory(
    steps: List[dict],
    task_id: str,
    score: float,
    total_steps: int,
    final_sla_violations: int = 0,
    final_invalid_actions: int = 0,
) -> EpisodeTrajectory:
    """Compress a raw step list into a trajectory with only key transitions.

    Raw steps are dicts with keys:
        step, action_type, target_node_id, parameter, reward,
        avg_latency_norm, error_rate, queue_backlog_norm, sla_violation

    Key transition selection:
    - First step
    - Last step
    - Steps with SLA violations
    - Steps with highest/lowest reward
    - Steps where action changed direction (e.g. SCALE_UP then SCALE_DOWN)
    """
    if not steps:
        return EpisodeTrajectory(
            task_id=task_id,
            score=score,
            total_steps=total_steps,
            final_sla_violations=final_sla_violations,
            final_invalid_actions=final_invalid_actions,
        )

    # Always include first and last
    key_indices = {0, len(steps) - 1}

    # Include SLA violations
    for i, s in enumerate(steps):
        if s.get("sla_violation"):
            key_indices.add(i)

    # Include reward extremes
    if len(steps) > 2:
        best_idx = max(range(len(steps)), key=lambda i: steps[i].get("reward", 0))
        worst_idx = min(range(len(steps)), key=lambda i: steps[i].get("reward", 0))
        key_indices.add(best_idx)
        key_indices.add(worst_idx)

    # Include action direction changes
    for i in range(1, len(steps)):
        prev_action = steps[i - 1].get("action_type", "")
        curr_action = steps[i].get("action_type", "")
        if prev_action != curr_action:
            key_indices.add(i)

    # Build compressed transitions (sorted)
    key_transitions = []
    for i in sorted(key_indices):
        s = steps[i]
        key_transitions.append(Transition(
            step=s.get("step", i),
            action_type=s.get("action_type", "NO_OP"),
            target_node_id=s.get("target_node_id", "node-0"),
            parameter=s.get("parameter", 0.0),
            reward=s.get("reward", 0.0),
            avg_latency_norm=s.get("avg_latency_norm", 0.0),
            error_rate=s.get("error_rate", 0.0),
            queue_backlog_norm=s.get("queue_backlog_norm", 0.0),
            sla_violation=s.get("sla_violation", False),
        ))

    return EpisodeTrajectory(
        task_id=task_id,
        score=score,
        key_transitions=key_transitions,
        total_steps=total_steps,
        final_sla_violations=final_sla_violations,
        final_invalid_actions=final_invalid_actions,
    )
