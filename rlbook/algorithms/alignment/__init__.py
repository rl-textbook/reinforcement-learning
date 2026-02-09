"""rlbook.algorithms.alignment — RLHF alignment pipeline (Chapter 5)."""

from .config import (
    Config,
    sequence_log_probs,
    response_log_probs,
    get_lr_lambda,
    filter_quality_pairs,
)
from .sft import SFTTrainer
from .reward_model import RewardModel
from .dpo import DPOTrainer
from .ppo_alignment import PPOTrainer

__all__ = [
    "Config",
    "sequence_log_probs",
    "response_log_probs",
    "get_lr_lambda",
    "filter_quality_pairs",
    "SFTTrainer",
    "RewardModel",
    "DPOTrainer",
    "PPOTrainer",
]
