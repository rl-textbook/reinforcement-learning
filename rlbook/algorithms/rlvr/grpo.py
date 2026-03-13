"""Chapter 6 — GRPO helpers for RLVR on GSM8K-style tasks."""

from contextlib import nullcontext
import random
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import extract_gt_number_text


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(model, device=None):
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _ctx(autocast_ctx):
    if autocast_ctx is None:
        return nullcontext()
    return autocast_ctx()


def response_token_logprobs(model, prompt_ids, response_ids, autocast_ctx=None):
    """Compute log-probabilities for response tokens in a single example."""
    assert prompt_ids.dim() == 2 and response_ids.dim() == 2
    assert prompt_ids.size(0) == 1 and response_ids.size(0) == 1

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)

    with _ctx(autocast_ctx):
        logits = model(input_ids=full_ids, attention_mask=attention_mask).logits

    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = full_ids[:, 1:]

    prompt_len = prompt_ids.size(1)
    response_len = response_ids.size(1)
    start = prompt_len - 1
    end = start + response_len

    token_logp = log_probs[:, start:end, :].gather(
        dim=-1,
        index=targets[:, start:end].unsqueeze(-1),
    ).squeeze(-1)

    return token_logp.squeeze(0)


@torch.no_grad()
def sample_response(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_prompt_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    device=None,
):
    """Generate one stochastic response and return prompt/response token splits."""
    device = _resolve_device(model, device)
    if seed is not None:
        _seed_everything(seed)

    tok_kwargs = {"return_tensors": "pt", "truncation": True}
    if max_prompt_tokens is not None:
        tok_kwargs["max_length"] = max_prompt_tokens
    inputs = tokenizer(prompt_text, **tok_kwargs)
    prompt_ids = inputs["input_ids"].to(device)

    gen_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response_ids = gen_ids[:, prompt_ids.size(1):]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text, prompt_ids, response_ids


@torch.no_grad()
def sample_groups(
    model,
    tokenizer,
    prompts: Sequence[str],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_prompt_tokens: Optional[int] = None,
    device=None,
):
    """Sample `group_size` completions for each prompt in a batch."""
    device = _resolve_device(model, device)
    _seed_everything(seed)

    # Step 1: Tokenize the prompt batch, padding and truncating as needed.
    tok_kwargs = {
        "return_tensors": "pt",
        "padding": True,
        "truncation": True,
    }
    if max_prompt_tokens is not None:
        tok_kwargs["max_length"] = max_prompt_tokens

    enc = tokenizer(list(prompts), **tok_kwargs).to(device)
    prompt_len = enc["input_ids"].shape[1]

    # Step 2: Sample G continuations per prompt in one batched generate() call.
    gen_ids = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Step 3: Repeat the prompt attention mask so it aligns with the flattened rollout batch.
    prompt_attn_rep = enc["attention_mask"].repeat_interleave(group_size, dim=0)

    # Step 4: Slice off the sampled suffix and decode it for verifier use.
    response_ids = gen_ids[:, prompt_len:]
    responses_text = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    return gen_ids, prompt_len, responses_text, prompt_attn_rep


def build_response_mask_from_gen(gen_ids: torch.Tensor, prompt_len: int, eos_id: int) -> torch.Tensor:
    """Build a full-sequence mask with 1s on response tokens through EOS."""
    # Step 1: Slice off the generated suffix for each sequence.
    n_rows, total_len = gen_ids.shape
    response_ids = gen_ids[:, prompt_len:]
    response_len = response_ids.shape[1]

    # Step 2: Find the first EOS in each response; if none appears, keep the full suffix.
    eos_pos = torch.full((n_rows,), response_len, device=gen_ids.device, dtype=torch.long)
    is_eos = response_ids == eos_id
    any_eos = is_eos.any(dim=1)
    if any_eos.any():
        first_eos = is_eos.float().argmax(dim=1)
        eos_pos = torch.where(any_eos, first_eos + 1, eos_pos)

    # Step 3: Mark response tokens up to and including EOS.
    arange = torch.arange(response_len, device=gen_ids.device).unsqueeze(0).expand(n_rows, response_len)
    response_mask = (arange < eos_pos.unsqueeze(1)).float()

    # Step 4: Embed that response-only mask back into the full prompt+response sequence.
    full_mask = torch.zeros((n_rows, total_len), device=gen_ids.device, dtype=torch.float32)
    full_mask[:, prompt_len:] = response_mask
    return full_mask


def token_logp_batch(model, ids: torch.Tensor, attn_mask: torch.Tensor, autocast_ctx=None) -> torch.Tensor:
    """Return token log-probs for `ids[:, 1:]` under the model."""
    # Step 1: Forward pass on the full prompt+response sequences.
    with _ctx(autocast_ctx):
        outputs = model(input_ids=ids, attention_mask=attn_mask)
    logits = outputs.logits
    
    # Step 2: Shift logits left so position t predicts token ids[:, t+1].
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = ids[:, 1:]

    # Step 3: Gather the log-prob assigned to each observed next token.
    return logp.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def grpo_step_one_prompt(
    policy,
    old_policy,
    tokenizer,
    verifier,
    formatter,
    question_text: str,
    answer_text: str,
    group_size: int = 8,
    clip_eps: float = 0.2,
    adv_eps: float = 1e-8,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    max_prompt_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    autocast_ctx=None,
):
    """Backward-compatible single-prompt GRPO update used by the older notebook."""
    device = _resolve_device(policy)
    prompt = formatter.format(question_text)
    samples = []
    rewards = []

    policy.eval()
    old_policy.eval()

    for j in range(group_size):
        with torch.no_grad():
            response_text, prompt_ids, response_ids = sample_response(
                old_policy,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                max_prompt_tokens=max_prompt_tokens,
                seed=None if seed is None else seed + j,
                device=device,
            )
        reward_value, _ = verifier.reward(response_text, answer_text)
        samples.append((prompt_ids, response_ids))
        rewards.append(reward_value)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    mu = rewards_t.mean()
    sigma = rewards_t.std(unbiased=False)
    advantages = (rewards_t - mu) / (sigma + adv_eps)

    policy.train()
    total_obj = torch.tensor(0.0, device=device)
    total_tokens = 0

    for idx, (prompt_ids, response_ids) in enumerate(samples):
        with torch.no_grad():
            old_logp = response_token_logprobs(old_policy, prompt_ids, response_ids, autocast_ctx=autocast_ctx)
        new_logp = response_token_logprobs(policy, prompt_ids, response_ids, autocast_ctx=autocast_ctx)

        ratio = torch.exp(new_logp - old_logp)
        ratio_clip = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        adv = advantages[idx].detach()
        obj_t = torch.minimum(ratio * adv, ratio_clip * adv)

        total_obj = total_obj + obj_t.sum()
        total_tokens += int(obj_t.numel())

    mean_obj = total_obj / max(1, total_tokens)
    loss = -mean_obj

    metrics = {
        "mean_reward": float(rewards_t.mean().item()),
        "reward_std": float(rewards_t.std(unbiased=False).item()),
        "mean_adv": float(advantages.mean().item()),
        "tokens": int(total_tokens),
    }
    return loss, metrics


def sync_old_policy(old_policy, policy) -> None:
    """PPO-style parameter sync used by the older modular notebook."""
    old_policy.load_state_dict(policy.state_dict())


def grpo_update_step(
    model,
    tokenizer,
    optimizer,
    verifier,
    formatter,
    batch_examples: Sequence[Mapping[str, str]],
    seed: int,
    group_size: int,
    max_prompt_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    clip_eps: float,
    adv_eps: float,
    grpo_epochs: int = 1,
    autocast_ctx=None,
    scaler=None,
    device=None,
    ground_truth_extractor: Optional[Callable[[str], str]] = None,
) -> Dict[str, float]:
    """Batched GRPO update matching the Qwen v2 case-study notebook."""
    device = _resolve_device(model, device)
    extractor = ground_truth_extractor or extract_gt_number_text

    # Step 1: Turn off dropout and build the prompt/answer batch used for verification.
    model.eval() 
    prompts = [formatter.format(ex["question"]) for ex in batch_examples]
    ground_truths = [extractor(ex["answer"]) for ex in batch_examples]
    batch_size = len(prompts)

    # Step 2: Sample G rollouts per prompt with the current model.
    gen_ids, prompt_len, response_texts, prompt_attn_rep = sample_groups(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_prompt_tokens=max_prompt_tokens,
        device=device,
    )

    total_samples = batch_size * group_size

    # Step 3: Verify each sampled completion to produce binary rewards in a flat vector.
    rewards = torch.zeros((total_samples,), device=device, dtype=torch.float32)
    parse_fail = 0
    exact = 0

    for i in range(batch_size):
        gt = ground_truths[i]
        group = response_texts[i * group_size:(i + 1) * group_size]
        for j, response_text in enumerate(group):
            idx = i * group_size + j
            reward_value, reason = verifier.reward(response_text, gt)
            rewards[idx] = reward_value
            exact += int(reward_value == 1.0)
            parse_fail += int(reason == "no-final-marker")

    # Step 4: Reshape rewards to (B, G) and compute within-group z-score advantages.
    rewards_grouped = rewards.view(batch_size, group_size)
    advantages = torch.zeros_like(rewards_grouped)

    for i in range(batch_size):
        row = rewards_grouped[i]
        mu = row.mean()
        sigma = row.std(unbiased=False)
        advantages[i] = (row - mu) / (sigma + adv_eps)

    # Step 5: Flatten advantages back to the rollout dimension and freeze them for PPO epochs.
    advantages = advantages.view(total_samples).detach()
    adv_nz = float((advantages.abs() > 1e-12).float().mean().item())

    # Step 6: Build masks used by the token-level GRPO objective.
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = int(eos_id[0])

    response_mask_full = build_response_mask_from_gen(gen_ids, prompt_len, int(eos_id))
    attn_full = torch.cat([prompt_attn_rep, response_mask_full[:, prompt_len:]], dim=1)
    response_mask_shift = response_mask_full[:, 1:]

    # Step 7: Snapshot old token log-probabilities once and keep them fixed across epochs.
    with torch.no_grad():
        old_logp = token_logp_batch(model, gen_ids, attn_full, autocast_ctx=autocast_ctx)

    # Step 8: Reuse the same sampled rollouts for multiple PPO-style epochs.
    last_loss = None
    for _ in range(grpo_epochs):
        optimizer.zero_grad(set_to_none=True)

        # Step 8a: Recompute current token log-probs under the updated policy.
        new_logp = token_logp_batch(model, gen_ids, attn_full, autocast_ctx=autocast_ctx)
        ratio = torch.exp(new_logp - old_logp)
        ratio_clip = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        # Step 8b: Apply the clipped GRPO objective on response tokens only.
        adv = advantages.unsqueeze(1)
        obj = torch.minimum(ratio * adv, ratio_clip * adv)
        obj = obj * response_mask_shift

        total_obj = obj.sum()
        total_tok = response_mask_shift.sum().clamp_min(1.0)
        mean_obj = total_obj / total_tok
        loss = -mean_obj

        if not torch.isfinite(loss).all():
            raise RuntimeError(f"Non-finite GRPO loss: {loss}")

        # Step 8c: Backpropagation, gradient clipping, and an optimizer step.
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        last_loss = loss

    # Step 9: Return the same training metrics tracked by the helper module.
    metrics = {
        "loss": float(last_loss.item()) if last_loss is not None else 0.0,
        "mean_reward": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "adv_nz": float(adv_nz),
        "exact": int(exact),
        "parse_fail": int(parse_fail),
        "tokens": int(response_mask_shift.sum().item()),
    }
    return metrics


__all__ = [
    "response_token_logprobs",
    "sample_response",
    "sample_groups",
    "build_response_mask_from_gen",
    "token_logp_batch",
    "grpo_step_one_prompt",
    "sync_old_policy",
    "grpo_update_step",
]
