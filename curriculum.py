"""
AntiAtropos Curriculum Training.

Defines progressive difficulty stages that the agent must pass before advancing.
Failed stages are retried with higher temperature for exploration.

Each stage specifies:
- task: Which task to run
- max_steps: Episode length (shorter = easier)
- pass_threshold: Minimum composite score to advance
- temperature: Suggest LLM temperature for this stage
- description: Human-readable label
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CurriculumStage:
    """A single stage in the training curriculum."""
    task: str
    max_steps: int
    pass_threshold: float
    temperature: float = 0.0
    description: str = ""
    retries: int = 0  # Number of failed attempts so far
    max_retries: int = 3  # Max retries before advancing anyway

    @property
    def retry_temperature(self) -> float:
        """Temperature increases with retries to encourage exploration."""
        if self.retries == 0:
            return self.temperature
        # 0.3, 0.6, 0.9 on retries
        return min(1.0, self.temperature + self.retries * 0.3)

    @property
    def should_skip(self) -> bool:
        """Skip this stage if too many retries."""
        return self.retries >= self.max_retries


# Progressive curriculum: start easy, add complexity
CURRICULUM: List[CurriculumStage] = [
    CurriculumStage(
        task="task-1", max_steps=40, pass_threshold=0.40,
        temperature=0.0, description="Short ramp — learn basic scaling",
    ),
    CurriculumStage(
        task="task-1", max_steps=60, pass_threshold=0.50,
        temperature=0.0, description="Standard ramp — scale proactively",
    ),
    CurriculumStage(
        task="task-1", max_steps=100, pass_threshold=0.55,
        temperature=0.0, description="Full ramp — cost-aware scaling",
    ),
    CurriculumStage(
        task="task-2", max_steps=40, pass_threshold=0.35,
        temperature=0.0, description="Short fault — learn reroute/scale on failure",
    ),
    CurriculumStage(
        task="task-2", max_steps=60, pass_threshold=0.45,
        temperature=0.3, description="Standard fault — fast recovery",
    ),
    CurriculumStage(
        task="task-3", max_steps=40, pass_threshold=0.35,
        temperature=0.0, description="Short surge — protect VIP during spike",
    ),
    CurriculumStage(
        task="task-3", max_steps=60, pass_threshold=0.45,
        temperature=0.3, description="Standard surge — sustained VIP protection",
    ),
    # Final combined test
    CurriculumStage(
        task="task-1", max_steps=100, pass_threshold=0.55,
        temperature=0.0, description="Final: full ramp at low temp",
    ),
    CurriculumStage(
        task="task-2", max_steps=60, pass_threshold=0.50,
        temperature=0.0, description="Final: fault recovery at low temp",
    ),
    CurriculumStage(
        task="task-3", max_steps=60, pass_threshold=0.50,
        temperature=0.0, description="Final: surge protection at low temp",
    ),
]


class CurriculumTracker:
    """Tracks progress through the curriculum stages."""

    def __init__(self, stages: Optional[List[CurriculumStage]] = None):
        self._stages = stages or CURRICULUM
        self._current_idx: int = 0

    @property
    def current(self) -> CurriculumStage:
        return self._stages[self._current_idx]

    @property
    def current_index(self) -> int:
        return self._current_idx

    @property
    def total_stages(self) -> int:
        return len(self._stages)

    @property
    def is_complete(self) -> bool:
        return self._current_idx >= len(self._stages)

    def report_score(self, score: float) -> bool:
        """Report a score for the current stage. Returns True if passed."""
        if score >= self.current.pass_threshold:
            self._current_idx += 1
            return True
        else:
            self.current.retries += 1
            if self.current.should_skip:
                self._current_idx += 1
            return False

    def progress_summary(self) -> str:
        stage = self.current
        return (
            f"Stage {self._current_idx + 1}/{self.total_stages}: "
            f"{stage.description} "
            f"(task={stage.task}, max_steps={stage.max_steps}, "
            f"threshold={stage.pass_threshold}, retries={stage.retries})"
        )
