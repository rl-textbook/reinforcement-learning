"""Chapter 5 — RLHF alignment configuration and shared utilities."""

import gc
import random
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed=7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    model_name: str = "gpt2"
    seed: int = 7
    max_length: int = 512
    max_new_tokens: int = 64

    sft_train_size: int = 1800
    sft_val_size: int = 200
    pref_train_size: int = 1800
    pref_val_size: int = 200

    batch_size: int = 8
    ppo_batch_size: int = 8

    min_response_tokens: int = 12

    sft_lr: float = 5e-5
    rm_lr: float = 2e-5
    dpo_lr: float = 5e-6
    ppo_lr: float = 5e-7

    sft_steps: int = 500
    rm_steps: int = 1000
    dpo_steps: int = 1000
    ppo_outer_steps: int = 400
    ppo_inner_epochs: int = 2

    beta_dpo: float = 0.1
    kl_coef_ppo: float = 0.02
    clip_eps: float = 0.2
    warmup_fraction: float = 0.1

    pref_probe_size: int = 128
    pref_probe_every: int = 25

    eval_benign_prompts: int = 50
    eval_harmful_prompts: int = 50
    eval_batch_size: int = 8
    showcase_num_candidates: int = 4

    pad_token_id: int = 0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def flush_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


ASSISTANT_MARKERS: Tuple[str, ...] = ("\n\nAssistant:", "\nAssistant:", "Assistant:")
HUMAN_MARKERS: Tuple[str, ...] = ("\n\nHuman:", "\nHuman:", "Human:")
HARMFUL_PATTERNS: Tuple[str, ...] = (
    r"\bkill\b", r"\bmurder\b", r"\bsuicide\b", r"\bbomb\b", r"\bpoison\b",
    r"\bsteal\b", r"\bweapon\b", r"\battack\b", r"\bharm\b", r"\bget away with it\b",
    r"\billegal\b", r"\bdrug\b", r"\bexplosive\b", r"\babuse\b", r"\bthreat\b",
)


def _last_marker_span(text: str, markers: Sequence[str]) -> Tuple[int, int]:
    best_idx = -1
    best_len = 0
    for marker in markers:
        idx = text.rfind(marker)
        if idx > best_idx or (idx == best_idx and len(marker) > best_len):
            best_idx = idx
            best_len = len(marker)
    return best_idx, best_len


def extract_prompt_and_response(text: str) -> Tuple[str, str]:
    """Split a HH-RLHF dialogue into the shared prompt and final assistant response.

    Functional correction versus the notebook wrapper:
    we split on the LAST assistant boundary so multi-turn dialogs keep the
    full conversation context for DPO/PPO/RM accounting.
    """
    text = (text or "").replace("\r", "")
    if not text.strip():
        return "", ""

    idx, marker_len = _last_marker_span(text, ASSISTANT_MARKERS)
    if idx == -1:
        fallback = text[:200].strip()
        return fallback, text.strip()

    prompt_end = idx + marker_len
    prompt = text[:prompt_end]
    response = text[prompt_end:].strip()
    return prompt, response


def extract_prompt(text: str) -> str:
    prompt, _ = extract_prompt_and_response(text)
    return prompt


def extract_response(text: str) -> str:
    _, response = extract_prompt_and_response(text)
    return response


def get_shared_prompt_for_preference(chosen_text: str, rejected_text: str) -> str:
    """Find the shared prompt prefix for a chosen/rejected HH-RLHF pair."""
    chosen_prompt, _ = extract_prompt_and_response(chosen_text)
    rejected_prompt, _ = extract_prompt_and_response(rejected_text)
    if chosen_prompt and chosen_prompt == rejected_prompt:
        return chosen_prompt

    chosen_text = chosen_text or ""
    rejected_text = rejected_text or ""
    min_len = min(len(chosen_text), len(rejected_text))
    common_prefix_end_idx = 0
    for i in range(min_len):
        if chosen_text[i] == rejected_text[i]:
            common_prefix_end_idx = i + 1
        else:
            break

    common_prefix = chosen_text[:common_prefix_end_idx]
    idx, marker_len = _last_marker_span(common_prefix, ASSISTANT_MARKERS)
    if idx != -1:
        return common_prefix[: idx + marker_len]

    human_idx, human_marker_len = _last_marker_span(common_prefix, HUMAN_MARKERS)
    if human_idx != -1:
        return common_prefix[: human_idx + human_marker_len]

    return common_prefix.strip()


def build_preference_example(chosen_text: str, rejected_text: str) -> Dict[str, str]:
    """Return a normalized preference example with shared prompt + response-only fields."""
    prompt = get_shared_prompt_for_preference(chosen_text, rejected_text)
    chosen_prompt, chosen_response = extract_prompt_and_response(chosen_text)
    rejected_prompt, rejected_response = extract_prompt_and_response(rejected_text)

    if prompt and (chosen_text or "").startswith(prompt):
        chosen_response = chosen_text[len(prompt):].strip()
    elif chosen_prompt:
        prompt = chosen_prompt

    if prompt and (rejected_text or "").startswith(prompt):
        rejected_response = rejected_text[len(prompt):].strip()
    elif not prompt and rejected_prompt:
        prompt = rejected_prompt

    return {
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "chosen_response": chosen_response,
        "rejected_response": rejected_response,
    }


def format_sft_example(example: Dict[str, str]) -> Dict[str, str]:
    """Normalize a raw HH-RLHF row into prompt/response form for SFT utilities."""
    chosen_text = example["chosen"]
    prompt, response = extract_prompt_and_response(chosen_text)
    return {
        "text": chosen_text,
        "chosen": chosen_text,
        "prompt": prompt,
        "response": response,
    }


def format_pref_example(example: Dict[str, str]) -> Dict[str, str]:
    """Normalize a raw HH-RLHF row into prompt/chosen/rejected response form."""
    built = build_preference_example(example["chosen"], example["rejected"])
    return {
        "prompt": built["prompt"],
        "chosen": built["chosen_response"],
        "rejected": built["rejected_response"],
        "chosen_full": built["chosen"],
        "rejected_full": built["rejected"],
    }


def is_harmful_prompt(prompt: str) -> bool:
    lowered = (prompt or "").lower()
    return any(re.search(pattern, lowered) for pattern in HARMFUL_PATTERNS)


def split_helpful_harmful(rows: Iterable[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    helpful: List[Dict[str, str]] = []
    harmful: List[Dict[str, str]] = []
    for ex in rows:
        prompt = ex.get("prompt") or extract_prompt(ex.get("chosen") or ex.get("chosen_full") or "")
        if is_harmful_prompt(prompt):
            harmful.append(ex)
        else:
            helpful.append(ex)
    return helpful, harmful


def filter_quality_pairs(dataset, tokenizer, min_tokens=12, max_length=None):
    """Remove degenerate preference pairs (too short, malformed, or identical).

    This keeps the notebook's quality-filtering idea but uses the actual
    final assistant responses and a robust shared prompt extraction.
    """
    if max_length is None:
        max_length = getattr(tokenizer, "model_max_length", None)
        if max_length is None or max_length > 100_000:
            max_length = 512

    filtered = []
    for ex in dataset:
        built = build_preference_example(ex.get("chosen", ""), ex.get("rejected", ""))
        if not built["prompt"] or not built["chosen_response"] or not built["rejected_response"]:
            continue
        if built["chosen_response"] == built["rejected_response"]:
            continue

        c_len = len(
            tokenizer(
                built["chosen_response"],
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]
        )
        r_len = len(
            tokenizer(
                built["rejected_response"],
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )["input_ids"]
        )
        if min(c_len, r_len) < min_tokens:
            continue

        filtered.append(built)

    return filtered


# ── Log-probability helpers ───────────────────────────────────────
def _sanitize_logits(logits):
    """Replace NaN/Inf in logits to prevent cascading NaN in training."""
    return torch.where(
        torch.isfinite(logits), logits,
        torch.full_like(logits, -1e4),
    )


def sequence_log_probs(model, input_ids, attention_mask):
    # Step 1 (Forward pass): run the language model to obtain next-token logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Step 2 (Sanitize logits): defend against NaN/Inf logits that would corrupt
    # log-probabilities
    logits = _sanitize_logits(outputs.logits)

    # Step 3 (Convert logits -> log-probabilities): log_softmax gives
    # log P(token | context) over the vocabulary at each time step.
    log_probs = F.log_softmax(logits, dim=-1)[:, :-1, :]

    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()

    # Step 4 (Gather token log-probs):
    token_logps = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    # Step 5 (Sum over non-pad positions): yields one scalar per sequence:
    # log pi(sequence) = sum_t log pi(token_t | token_<t), ignoring padding.
    return (token_logps * mask).sum(dim=-1)

def response_log_probs(model, input_ids, attention_mask, prompt_lengths):
    """Response-only log pi(y|x), handles left-padding correctly."""
    # Step 1: Run the policy model to obtain next-token logits
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # Step 2: Sanitize logits before converting them to log-probabilities
    logits = _sanitize_logits(outputs.logits)

    # Step 3: Convert logits -> log-probabilities over the vocabulary at each position.
    # For a causal LM, logits at position t predict token at position t+1, so we drop the last step.
    log_probs = F.log_softmax(logits, dim=-1)[:, :-1, :]
    targets = input_ids[:, 1:]

    mask = attention_mask[:, 1:].float()
    B, T_minus1 = mask.shape

    # Step 4: Compute how many pad tokens were added on the LEFT for each sequence.
    pad_count = input_ids.size(1) - attention_mask.sum(dim=1)

    # Step 5: Compute the first response token index in the PADDED sequence:
    response_start = pad_count + prompt_lengths
    positions = torch.arange(T_minus1, device=mask.device).unsqueeze(0)

    # Step 6: Create a response-only mask.
    response_mask = (positions >= (response_start.unsqueeze(1) - 1)).float()

    # Step 7: Combine masks so ONLY response tokens (and only non-pad tokens) contribute.
    mask = mask * response_mask

    # Step 8: Gather the model's log-prob assigned to the observed next token at each position and sum.
    token_logps = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


# ── LR warmup ─────────────────────────────────────────────────────
def get_lr_lambda(warmup_steps, total_steps):
    """Linear warmup schedule for LambdaLR."""
    def lr_lambda(step):
        # Step 1: During warmup, scale the learning-rate multiplier linearly from 0 to 1.
        if step < warmup_steps:
            return step / max(warmup_steps, 1)

        # Step 2: After warmup, keep the multiplier fixed at 1.
        return 1.0

    return lr_lambda


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
]
