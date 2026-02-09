"""
Configuration and shared utilities for Chapter 5: Preference-Based
Alignment (RLHF pipeline).

Includes Config, data processing, tokenization, and evaluation helpers
for the SFT → Reward Model → DPO/PPO pipeline.
"""

import torch
import torch.nn.functional as F


class Config:
    model_name: str = "gpt2"
    max_length: int = 512

    # Data sizes (demo-scale)
    sft_train_size: int = 2000
    sft_val_size: int = 200
    pref_train_size: int = 2000
    pref_val_size: int = 200

    # Training hyperparams
    batch_size: int = 8         # SFT / RM / DPO
    ppo_batch_size: int = 4     # PPO needs less VRAM (4 models on GPU)
    sft_steps: int = 500
    rm_steps: int = 1000
    dpo_steps: int = 1000
    ppo_outer_steps: int = 400
    ppo_inner_epochs: int = 2

    sft_lr: float = 5e-5
    rm_lr: float = 5e-5
    dpo_lr: float = 1e-5
    ppo_lr: float = 1e-5

    beta_dpo: float = 0.05     # DPO beta
    kl_coef_ppo: float = 0.01  # PPO KL coefficient
    clip_eps: float = 0.2      # PPO clipping epsilon

    max_new_tokens: int = 64   # generation length for RLHF
    pad_token_id: int = 50256  # for GPT-2; override with tokenizer.pad_token_id

    min_response_tokens: int = 8   # filter degenerate preference pairs
    warmup_fraction: float = 0.1   # LR warmup for RM and DPO


# --- LR warmup helper ---

def get_lr_lambda(warmup_steps, total_steps):
    """Linear warmup then constant LR."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0
    return lr_lambda


# --- Sequence-level log probability ---

def sequence_log_probs(model, input_ids, attention_mask):
    """
    Log-probability of each sequence under a causal LM, conditioned on its prefix.
    Returns a tensor of shape (B,) with summed log-probabilities.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, T, V)
    logits = torch.clamp(logits, min=-1e4, max=1e4)
    log_probs = F.log_softmax(logits, dim=-1)

    log_probs = log_probs[:, :-1, :]          # (B, T-1, V)
    targets = input_ids[:, 1:]                # (B, T-1)
    mask = attention_mask[:, 1:].float()      # (B, T-1)

    token_logps = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    return token_logps.sum(dim=-1)            # (B,)


def response_log_probs(model, input_ids, attention_mask, prompt_lengths):
    """
    Log-probability of only the *response* tokens, masking out the prompt.

    Handles left-padding correctly: with left-padded inputs the layout is
    [PAD ... PAD | prompt tokens | response tokens], so the response
    starts at position (pad_count + prompt_content_length).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    logits = torch.clamp(logits, min=-1e4, max=1e4)
    log_probs = F.log_softmax(logits, dim=-1)

    log_probs = log_probs[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()

    B, T_minus1 = mask.shape
    pad_count = input_ids.size(1) - attention_mask.sum(dim=1)
    response_start = pad_count + prompt_lengths

    positions = torch.arange(T_minus1, device=mask.device).unsqueeze(0)
    response_mask = (positions >= (response_start.unsqueeze(1) - 1)).float()
    mask = mask * response_mask

    token_logps = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    return token_logps.sum(dim=-1)


# --- Data processing helpers ---

HARMFUL_KEYWORDS = [
    "kill", "harm", "suicide", "self-harm", "bomb", "weapon", "terrorist",
    "hate", "racist", "assault", "abuse"
]

def is_harmful_prompt(text):
    low = text.lower()
    return any(kw in low for kw in HARMFUL_KEYWORDS)

def split_helpful_harmful(ds):
    helpful_indices = []
    harmful_indices = []
    for i, ex in enumerate(ds):
        prompt = ex["prompt"]
        if is_harmful_prompt(prompt):
            harmful_indices.append(i)
        else:
            helpful_indices.append(i)
    return ds.select(helpful_indices), ds.select(harmful_indices)


def extract_prompt_and_response(full_text):
    """Extract the prompt and final assistant response from a conversation string."""
    parts = full_text.rsplit("\n\nAssistant:", 1)
    if len(parts) == 2:
        prompt_text = parts[0] + "\n\nAssistant:"
        response_text = parts[1]
    else:
        prompt_text = full_text
        response_text = ""
    return prompt_text.strip(), response_text.strip()

def get_shared_prompt_for_preference(chosen_text, rejected_text):
    """Find the longest common prefix ending at a natural turn boundary."""
    min_len = min(len(chosen_text), len(rejected_text))
    common_prefix_end_idx = 0
    for i in range(min_len):
        if chosen_text[i] == rejected_text[i]:
            common_prefix_end_idx = i + 1
        else:
            break

    common_prefix = chosen_text[:common_prefix_end_idx]
    last_assistant_idx = common_prefix.rfind("\n\nAssistant:")
    last_human_idx = common_prefix.rfind("\n\nHuman:")

    if last_assistant_idx == -1 and last_human_idx == -1:
        return common_prefix.strip() if common_prefix else ""
    if last_assistant_idx > last_human_idx:
        return common_prefix[:last_assistant_idx + len("\n\nAssistant:")].strip()
    else:
        return common_prefix[:last_human_idx + len("\n\nHuman:")].strip()

def format_sft_example(ex):
    prompt, response = extract_prompt_and_response(ex["chosen"])
    return {"prompt": prompt, "response": response}

def format_pref_example(ex):
    prompt = get_shared_prompt_for_preference(ex["chosen"], ex["rejected"])
    chosen_suffix = ex["chosen"][len(prompt):]
    rejected_suffix = ex["rejected"][len(prompt):]
    return {"prompt": prompt, "chosen": chosen_suffix, "rejected": rejected_suffix}

def filter_quality_pairs(ds, tokenizer, min_tokens=8):
    """Remove preference pairs where chosen/rejected are too short or identical.

    Filtering degenerate pairs improves reward model training by removing
    examples with no meaningful preference signal.
    """
    good_indices = []
    for i, ex in enumerate(ds):
        c_len = len(tokenizer(ex["chosen"], add_special_tokens=False)["input_ids"])
        r_len = len(tokenizer(ex["rejected"], add_special_tokens=False)["input_ids"])
        if c_len < min_tokens or r_len < min_tokens:
            continue
        if ex["chosen"].strip() == ex["rejected"].strip():
            continue
        good_indices.append(i)
    return ds.select(good_indices)
