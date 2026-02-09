"""Chapter 6 — GRPO (Group Relative Policy Optimization) for RLVR.

Implements:
  - response_token_logprobs: per-token log-probs on response tokens
  - sample_response: stochastic decoding with prompt/response split
  - grpo_step_one_prompt: one GRPO update (group sampling → advantage → clipped objective)
  - sync_old_policy: periodic parameter copy (PPO-style)
"""

import torch
import torch.nn.functional as F


def response_token_logprobs(model, prompt_ids, response_ids):
    """Compute log π(token_t | prefix) for each response token.

    Args:
        model: Language model.
        prompt_ids: LongTensor (1, P).
        response_ids: LongTensor (1, R).

    Returns:
        logp: FloatTensor (R,) — per-token log-probs of response tokens.
    """
    assert prompt_ids.dim() == 2 and response_ids.dim() == 2
    assert prompt_ids.size(0) == 1 and response_ids.size(0) == 1

    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    logits = model(full_ids).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = full_ids[:, 1:]

    P = prompt_ids.size(1)
    R = response_ids.size(1)
    start = P - 1
    end = start + R

    token_logp = log_probs[:, start:end, :].gather(
        dim=-1, index=targets[:, start:end].unsqueeze(-1)
    ).squeeze(-1)

    return token_logp.squeeze(0)


def sample_response(model, tokenizer, prompt_text,
                    max_new_tokens=256, temperature=1.0, top_p=0.95):
    """Generate a single response via stochastic decoding.

    Returns:
        response_text: Decoded response string.
        prompt_ids: LongTensor (1, P) on model device.
        response_ids: LongTensor (1, R) on model device.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = inputs["input_ids"].to(device)

    gen_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )

    response_ids = gen_ids[:, prompt_ids.size(1):]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return response_text, prompt_ids, response_ids


def grpo_step_one_prompt(policy, old_policy, tokenizer, verifier, formatter,
                         question_text, answer_text,
                         group_size=8, clip_eps=0.2, adv_eps=1e-8):
    """One GRPO update for a single prompt.

    1. Sample G completions from old_policy.
    2. Score each with the verifier (binary reward).
    3. Compute group-relative advantages (z-score).
    4. Clipped PPO-style objective over per-token ratios.

    Returns:
        loss: Scalar tensor (gradient-ready, call loss.backward()).
        metrics: Dict with mean_reward, reward_std, mean_adv, tokens.
    """
    device = next(policy.parameters()).device

    prompt = formatter.format(question_text)
    samples = []
    rewards = []

    policy.eval()
    old_policy.eval()

    for _ in range(group_size):
        with torch.no_grad():
            resp_text, prompt_ids, resp_ids = sample_response(
                old_policy, tokenizer, prompt
            )
        r, _ = verifier.reward(resp_text, answer_text)
        samples.append((prompt_ids, resp_ids))
        rewards.append(r)

    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

    # Group-relative advantage (z-score)
    mu = rewards_t.mean()
    sigma = rewards_t.std(unbiased=False)
    adv = (rewards_t - mu) / (sigma + adv_eps)

    # Clipped GRPO objective
    policy.train()
    total_obj = 0.0
    total_tokens = 0

    for j, (prompt_ids, resp_ids) in enumerate(samples):
        with torch.no_grad():
            old_logp = response_token_logprobs(old_policy, prompt_ids, resp_ids)
        new_logp = response_token_logprobs(policy, prompt_ids, resp_ids)

        ratio = torch.exp(new_logp - old_logp)
        ratio_clip = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        A = adv[j].detach()
        obj_t = torch.minimum(ratio * A, ratio_clip * A)

        total_obj = total_obj + obj_t.sum()
        total_tokens += obj_t.numel()

    mean_obj = total_obj / max(1, total_tokens)
    loss = -mean_obj

    metrics = {
        "mean_reward": float(rewards_t.mean().item()),
        "reward_std": float(rewards_t.std(unbiased=False).item()),
        "mean_adv": float(adv.mean().item()),
        "tokens": int(total_tokens),
    }
    return loss, metrics


def sync_old_policy(old_policy, policy):
    """PPO-style: periodically copy policy parameters into old_policy."""
    old_policy.load_state_dict(policy.state_dict())
