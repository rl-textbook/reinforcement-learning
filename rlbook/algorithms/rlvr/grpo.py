"""Sampling and GRPO update helpers for the Chapter 6 RLVR case study."""

from typing import Dict, List

import torch
import torch.nn.functional as F
from transformers import set_seed

from .config import extract_gt_number_text


@torch.no_grad()
def sample_groups(
    model,
    tokenizer,
    prompts: List[str],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    max_prompt_tokens: int,
    device,
):
    """
    Returns:
      gen_ids: (B*G, T) token IDs (prompt+response)
      prompt_len: int (shared padded prompt length)
      responses_text: list[str] length B*G (decoded response-only)
      prompt_attn_rep: (B*G, prompt_len) attention mask for prompt part (after padding)
    """
    set_seed(seed)

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    ).to(device)

    prompt_len = enc["input_ids"].shape[1]

    gen_ids = model.generate(
        **enc,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        num_return_sequences=group_size,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    prompt_attn_rep = enc["attention_mask"].repeat_interleave(group_size, dim=0)
    resp_ids = gen_ids[:, prompt_len:]
    responses_text = tokenizer.batch_decode(resp_ids, skip_special_tokens=True)

    return gen_ids, prompt_len, responses_text, prompt_attn_rep


def build_response_mask_from_gen(gen_ids: torch.Tensor, prompt_len: int, eos_id: int) -> torch.Tensor:
    """
    Returns a full-sequence mask (N, T) with 1s on response tokens up to (and including) EOS.
    """
    n_rows, total_len = gen_ids.shape
    resp = gen_ids[:, prompt_len:]
    resp_len = resp.shape[1]

    eos_pos = torch.full((n_rows,), resp_len, device=gen_ids.device, dtype=torch.long)
    is_eos = resp == eos_id
    any_eos = is_eos.any(dim=1)
    if any_eos.any():
        first = is_eos.float().argmax(dim=1)
        eos_pos = torch.where(any_eos, first + 1, eos_pos)  # include EOS

    ar = torch.arange(resp_len, device=gen_ids.device).unsqueeze(0).expand(n_rows, resp_len)
    resp_mask = (ar < eos_pos.unsqueeze(1)).float()

    full = torch.zeros((n_rows, total_len), device=gen_ids.device, dtype=torch.float32)
    full[:, prompt_len:] = resp_mask
    return full


def token_logp_batch(model, ids: torch.Tensor, attn_mask: torch.Tensor, autocast_ctx) -> torch.Tensor:
    """
    Returns token logp for ids[:,1:] under the model: shape (N, T-1).
    Numerically stable: logits are cast to float32 for log-softmax.
    """
    with autocast_ctx():
        out = model(input_ids=ids, attention_mask=attn_mask)
    logits = out.logits  # (N, T, V)
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    targets = ids[:, 1:]
    return logp.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def grpo_update_step(
    model,
    tokenizer,
    verifier,
    formatter,
    batch_examples: List[Dict],
    optimizer,
    cfg,
    seed: int,
    autocast_ctx,
    use_scaler: bool,
    scaler,
    device,
) -> Dict:
    model.eval()  # dropout off

    prompts = [formatter.format(ex["question"]) for ex in batch_examples]
    gts = [extract_gt_number_text(ex["answer"]) for ex in batch_examples]
    batch_size = len(prompts)
    group_size = cfg.group_size

    # 1) sample group
    gen_ids, prompt_len, resp_texts, prompt_attn_rep = sample_groups(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=group_size,
        max_new_tokens=cfg.max_new_tokens_train,
        temperature=cfg.temperature_train,
        top_p=cfg.top_p_train,
        seed=seed,
        max_prompt_tokens=cfg.max_prompt_tokens,
        device=device,
    )

    total_samples = batch_size * group_size

    # 2) rewards
    rewards = torch.zeros((total_samples,), device=device, dtype=torch.float32)
    parse_fail = 0
    exact = 0
    for i in range(batch_size):
        gt = gts[i]
        group = resp_texts[i * group_size:(i + 1) * group_size]
        for j in range(group_size):
            idx = i * group_size + j
            reward_value, reason = verifier.reward(group[j], gt)
            rewards[idx] = reward_value
            exact += int(reward_value == 1.0)
            parse_fail += int(reason == "no-final-marker")

    # 3) group-relative advantages (z-score within each prompt group)
    rewards_g = rewards.view(batch_size, group_size)
    adv = torch.zeros_like(rewards_g)
    for i in range(batch_size):
        row = rewards_g[i]
        mu = row.mean()
        sigma = row.std(unbiased=False)

        # If sigma==0, advantages are all zero (no within-group signal). This is expected.
        adv[i] = (row - mu) / (sigma + cfg.adv_eps)

    adv = adv.view(total_samples).detach()
    adv_nz = float((adv.abs() > 1e-12).float().mean().item())

    # masks / attention
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = int(eos_id[0])

    resp_mask_full = build_response_mask_from_gen(gen_ids, prompt_len, eos_id)  # (N, T)
    attn_full = torch.cat([prompt_attn_rep, resp_mask_full[:, prompt_len:]], dim=1)  # (N, T)
    resp_mask_shift = resp_mask_full[:, 1:]  # (N, T-1)

    # old logp snapshot
    with torch.no_grad():
        old_logp = token_logp_batch(model, gen_ids, attn_full, autocast_ctx)  # (N, T-1)

    # PPO-style multiple epochs
    last_loss = None
    for _ in range(cfg.grpo_epochs):
        optimizer.zero_grad(set_to_none=True)

        new_logp = token_logp_batch(model, gen_ids, attn_full, autocast_ctx)  # (N, T-1)
        ratio = torch.exp(new_logp - old_logp)
        ratio_clip = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps)

        adv_expanded = adv.unsqueeze(1)
        obj = torch.minimum(ratio * adv_expanded, ratio_clip * adv_expanded)
        obj = obj * resp_mask_shift

        total_obj = obj.sum()
        total_tok = resp_mask_shift.sum().clamp_min(1.0)
        mean_obj = total_obj / total_tok

        loss = -mean_obj

        if not torch.isfinite(loss).all():
            raise RuntimeError(f"Non-finite GRPO loss: {loss}")

        if use_scaler:
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

    metrics = {
        "loss": float(last_loss.item()) if last_loss is not None else 0.0,
        "mean_reward": float(rewards.mean().item()),
        "reward_std": float(rewards.std(unbiased=False).item()),
        "adv_nz": float(adv_nz),
        "exact": int(exact),
        "parse_fail": int(parse_fail),
        "tokens": int(resp_mask_shift.sum().item()),
    }
    return metrics


__all__ = [
    "sample_groups",
    "build_response_mask_from_gen",
    "token_logp_batch",
    "grpo_update_step",
]
