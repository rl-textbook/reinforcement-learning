"""rlbook.algorithms.rlvr — RLVR with GRPO for reasoning models (Chapter 6)."""

from .config import Config, parse_gsm8k_answer, gsm8k_solution_to_cot_and_final
from .verifier import MathVerifier
from .formatter import CoTFormatter
from .grpo import (
    response_token_logprobs,
    sample_response,
    grpo_step_one_prompt,
    sync_old_policy,
)
from .evaluation import pass_at_k, diagnostics_on_test, greedy_response

__all__ = [
    "Config",
    "parse_gsm8k_answer",
    "gsm8k_solution_to_cot_and_final",
    "MathVerifier",
    "CoTFormatter",
    "response_token_logprobs",
    "sample_response",
    "grpo_step_one_prompt",
    "sync_old_policy",
    "pass_at_k",
    "diagnostics_on_test",
    "greedy_response",
]
