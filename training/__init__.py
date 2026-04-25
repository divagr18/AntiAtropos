"""AntiAtropos Training Module.

RL loss functions and training loop for LLM-based SRE agents.
Works with the simulator in pure-Python mode (no AWS/GPU needed for validation).
"""

from .losses import (
    compute_returns,
    compute_gae,
    reinforce_loss,
    reinforce_baseline_loss,
    grpo_loss,
    rloo_loss,
)
from .trainer import SRETrainer, TrainingConfig, EpisodeCollector

__all__ = [
    "compute_returns",
    "compute_gae",
    "reinforce_loss",
    "reinforce_baseline_loss",
    "grpo_loss",
    "rloo_loss",
    "SRETrainer",
    "TrainingConfig",
    "EpisodeCollector",
]
