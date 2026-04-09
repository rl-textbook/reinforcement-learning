"""Chapter 6 configuration and GSM8K parsing helpers."""

from dataclasses import dataclass
from typing import Tuple
import re

GT_RE = re.compile(r"####\s*([-+]?\d+(?:\.\d+)?)")


@dataclass
class CFG:
    # Model: small enough for full fine-tuning, strong enough for sparse RLVR rewards
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Data sizes (subsample for a fast, reproducible case study run)
    train_n: int = 4000
    test_n: int = 256

    # Prompt / generation
    max_prompt_tokens: int = 256
    max_new_tokens_train: int = 128
    max_new_tokens_eval: int = 128
    temperature_train: float = 1.0
    top_p_train: float = 0.95
    temperature_eval: float = 1.0
    top_p_eval: float = 0.95

    # SFT
    sft_steps: int = 200
    sft_lr: float = 2e-5
    sft_batch_size: int = 2
    sft_grad_accum: int = 8
    sft_max_seq_len: int = 512

    # GRPO (RLVR)
    grpo_steps: int = 400
    grpo_lr: float = 2e-6
    prompts_per_step: int = 4
    group_size: int = 8
    clip_eps: float = 0.2
    adv_eps: float = 1e-6
    grpo_epochs: int = 2  # PPO-style: multiple epochs over the same rollouts

    # Evaluation cadence
    eval_every: int = 100
    eval_curve_n: int = 128
    eval_ks_curve: Tuple[int, ...] = (1, 4, 8)
    eval_ks_scaling: Tuple[int, ...] = (1, 2, 4, 8, 16)
    eval_seed: int = 12345

    # Verifier tolerance
    tol: float = 1e-5


Config = CFG


def extract_gt_number_text(gsm8k_answer: str) -> str:
    m = GT_RE.search(gsm8k_answer)
    if m:
        return m.group(1).strip()
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", gsm8k_answer.replace(",", ""))
    return nums[-1] if nums else ""


def build_sft_completion(gsm8k_answer: str) -> str:
    gt = extract_gt_number_text(gsm8k_answer).replace(",", "")
    if "####" in gsm8k_answer:
        cot = gsm8k_answer.split("####")[0].rstrip()
    else:
        cot = gsm8k_answer.rstrip()
    if cot:
        return f"{cot}\nFINAL: {gt}\n"
    return f"FINAL: {gt}\n"


__all__ = [
    "CFG",
    "Config",
    "GT_RE",
    "extract_gt_number_text",
    "build_sft_completion",
]
