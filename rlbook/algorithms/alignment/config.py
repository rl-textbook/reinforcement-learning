"""Chapter 5 — RLHF alignment configuration and shared utilities."""

import torch
import torch.nn.functional as F


class Config:
    """Hyperparameters for the SFT -> RM -> DPO/PPO alignment pipeline."""

    model_name = "gpt2"
    max_length = 512
    sft_train_size = 2000
    sft_val_size = 200
    pref_train_size = 2000
    pref_val_size = 200
    batch_size = 8
    ppo_batch_size = 4           # separate — PPO holds 4 models in VRAM
    sft_steps = 500
    rm_steps = 1000
    dpo_steps = 1000
    ppo_outer_steps = 400
    ppo_inner_epochs = 2
    sft_lr = 5e-5
    rm_lr = 5e-5
    dpo_lr = 1e-5
    ppo_lr = 5e-6                # conservative — prevents NaN cascades
    beta_dpo = 0.05
    kl_coef_ppo = 0.01
    clip_eps = 0.2
    max_new_tokens = 64
    pad_token_id = 50256
    min_response_tokens = 8
    warmup_fraction = 0.1


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
    # log-probabilities.
    logits = _sanitize_logits(outputs.logits)

    # Step 3 (Convert logits -> log-probabilities): log_softmax gives
    # log P(token | context) over the vocabulary at each time step.
    log_probs = F.log_softmax(logits, dim=-1)[:, :-1, :]
    targets = input_ids[:, 1:]
    mask = attention_mask[:, 1:].float()

    # Step 4 (Gather token log-probs)
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

    # Step 5: Compute the first response token index in the PADDED sequence.
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


# ── Data quality filter ───────────────────────────────────────────


def filter_quality_pairs(dataset, tokenizer, min_tokens=8):
    """Remove degenerate preference pairs (too short or identical)."""
    filtered = []
    for ex in dataset:
        c_len = len(tokenizer.encode(ex.get("chosen", ""), add_special_tokens=False))
        r_len = len(tokenizer.encode(ex.get("rejected", ""), add_special_tokens=False))
        if c_len >= min_tokens and r_len >= min_tokens:
            if ex.get("chosen", "") != ex.get("rejected", ""):
                filtered.append(ex)
    return filtered
