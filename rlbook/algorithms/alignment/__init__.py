# Alignment algorithms require torch + transformers — import only when needed
try:
    from rlbook.algorithms.alignment.config import (
        Config, sequence_log_probs, response_log_probs,
        is_harmful_prompt, split_helpful_harmful,
        extract_prompt_and_response, get_shared_prompt_for_preference,
        format_sft_example, format_pref_example,
        filter_quality_pairs, get_lr_lambda,
    )
    from rlbook.algorithms.alignment.sft import SFTTrainer
    from rlbook.algorithms.alignment.reward_model import RewardModel
    from rlbook.algorithms.alignment.dpo import DPOTrainer
    from rlbook.algorithms.alignment.ppo_alignment import PPOTrainer
    from rlbook.algorithms.alignment.gpt4_judge import gpt4_judge_pair, gpt4_judge_safety
except ImportError:
    pass
