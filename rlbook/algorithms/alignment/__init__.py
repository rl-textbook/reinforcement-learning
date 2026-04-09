"""rlbook.algorithms.alignment — RLHF alignment pipeline (Chapter 5)."""

from .config import (
    Config,
    set_seed,
    flush_cuda,
    extract_prompt,
    extract_response,
    extract_prompt_and_response,
    get_shared_prompt_for_preference,
    build_preference_example,
    format_sft_example,
    format_pref_example,
    is_harmful_prompt,
    split_helpful_harmful,
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
    "set_seed",
    "flush_cuda",
    "extract_prompt",
    "extract_response",
    "extract_prompt_and_response",
    "get_shared_prompt_for_preference",
    "build_preference_example",
    "format_sft_example",
    "format_pref_example",
    "is_harmful_prompt",
    "split_helpful_harmful",
    "sequence_log_probs",
    "response_log_probs",
    "get_lr_lambda",
    "filter_quality_pairs",
    "SFTTrainer",
    "RewardModel",
    "DPOTrainer",
    "PPOTrainer",
]
