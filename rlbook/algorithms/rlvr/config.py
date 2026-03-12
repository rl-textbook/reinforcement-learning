"""Chapter 6 — RLVR/GRPO configuration and GSM8K parsing helpers."""

from dataclasses import dataclass
from typing import Tuple
import re

_GT_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

@dataclass
class Config:
    """Default configuration for the Chapter 6 Qwen/GSM8K case study.

    The field names match the newer Qwen v2 notebook. A few read-only
    properties keep the older Chapter 6 modular notebook compatible.
    """

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Data sizes
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

    # GRPO
    grpo_steps: int = 400
    grpo_lr: float = 2e-6
    prompts_per_step: int = 4
    group_size: int = 8
    clip_eps: float = 0.2
    adv_eps: float = 1e-6
    grpo_epochs: int = 2

    # Evaluation
    eval_every: int = 100
    eval_curve_n: int = 128
    eval_ks_curve: Tuple[int, ...] = (1, 4, 8)
    eval_ks_scaling: Tuple[int, ...] = (1, 2, 4, 8, 16)
    eval_seed: int = 12345

    # Verifier
    tol: float = 1e-5

    # Legacy aliases used by the older modular notebook.
    val_size: int = 64
    sync_old_every: int = 10

    @property
    def train_size(self) -> int:
        return self.train_n

    @property
    def test_size(self) -> int:
        return self.test_n

    @property
    def max_length(self) -> int:
        return self.sft_max_seq_len

    @property
    def temperature(self) -> float:
        return self.temperature_eval

    @property
    def top_p(self) -> float:
        return self.top_p_eval

    @property
    def eval_ks_learning_curves(self) -> Tuple[int, ...]:
        return self.eval_ks_curve

    @property
    def eval_ks_compute_scaling(self) -> Tuple[int, ...]:
        return self.eval_ks_scaling


def _clean_number_text(text: str) -> str:
    return text.replace(",", "").strip()


def extract_gt_number_text(gsm8k_answer: str) -> str:
    """Extract the final numeric GSM8K answer as normalized text."""
    if gsm8k_answer is None:
        return ""
    match = _GT_RE.search(gsm8k_answer)
    if match:
        return _clean_number_text(match.group(1))

    nums = _NUM_RE.findall(gsm8k_answer)
    if not nums:
        return ""
    return _clean_number_text(nums[-1])


def parse_gsm8k_answer(answer_text: str) -> str:
    """Backward-compatible alias for GSM8K final-answer extraction."""
    return extract_gt_number_text(answer_text)


def build_sft_completion(gsm8k_answer: str) -> str:
    """Rewrite a GSM8K worked solution to the strict 'FINAL:' contract."""
    gt = extract_gt_number_text(gsm8k_answer)
    cot = gsm8k_answer.split("####")[0].rstrip() if "####" in gsm8k_answer else gsm8k_answer.rstrip()

    if cot:
        return f"{cot}\nFINAL: {gt}\n"
    return f"FINAL: {gt}\n"


def gsm8k_solution_to_cot_and_final(answer_text: str) -> str:
    """Backward-compatible alias used by the older Chapter 6 notebook."""
    return build_sft_completion(answer_text).rstrip()


__all__ = [
    "Config",
    "extract_gt_number_text",
    "parse_gsm8k_answer",
    "build_sft_completion",
    "gsm8k_solution_to_cot_and_final",
]
