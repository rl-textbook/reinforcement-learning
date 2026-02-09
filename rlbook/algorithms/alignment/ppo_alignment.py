"""Chapter 5 — PPO for LLM alignment with reward normalization.

Supports both vanilla and mitigated configurations via constructor
parameters (kl_coef, repetition_penalty, reward_clip).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import LogitsProcessor

from .config import response_log_probs


class _SanitizeLogitsProcessor(LogitsProcessor):
    """Replace non-finite logits during generation."""
    def __call__(self, input_ids, scores):
        return torch.where(
            torch.isfinite(scores), scores,
            torch.full_like(scores, -1e9),
        )


class PPOTrainer:
    """PPO-RLHF trainer with per-token KL, reward normalization, and
    optional mitigations (higher KL penalty, repetition penalty, reward clipping).

    Args:
        policy: Trainable language model.
        ref_policy: Frozen reference model (SFT checkpoint).
        reward_model: Learned scalar reward model.
        tokenizer: HuggingFace tokenizer.
        cfg: Config object with hyperparameters.
        kl_coef: KL coefficient (default: cfg.kl_coef_ppo). Use 0.1 for mitigated PPO.
        repetition_penalty: Generation repetition penalty (default: 1.0). Use 1.2 for mitigated.
        reward_clip: Clip raw RM rewards to ±this value (default: None). Use 2.0 for mitigated.
    """

    def __init__(self, policy, ref_policy, reward_model, tokenizer, cfg,
                 kl_coef=None, repetition_penalty=1.0, reward_clip=None):
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = next(policy.parameters()).device
        self.kl_coef = kl_coef if kl_coef is not None else cfg.kl_coef_ppo
        self.repetition_penalty = repetition_penalty
        self.reward_clip = reward_clip

        hidden_size = policy.transformer.wte.weight.size(1)
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)
        self.optimizer = AdamW(
            list(policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.ppo_lr,
        )
        # Running reward statistics for normalization
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0

    def _update_reward_stats(self, rewards):
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item() if rewards.numel() > 1 else 1.0
        n = rewards.numel()
        total = self.reward_count + n
        delta = batch_mean - self.reward_mean
        self.reward_mean += delta * n / max(total, 1)
        self.reward_var = (self.reward_var * self.reward_count + batch_var * n) / max(total, 1)
        self.reward_count = total

    def _normalize_rewards(self, rewards):
        # Guard against NaN/Inf corrupting running stats
        if not torch.isfinite(rewards).all():
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=2.0, neginf=-2.0)
        self._update_reward_stats(rewards)
        std = max(self.reward_var ** 0.5, 1e-8)
        normed = (rewards - self.reward_mean) / std
        return torch.clamp(normed, -5.0, 5.0)

    @torch.no_grad()
    def generate_responses(self, batch_prompts):
        self.policy.eval()
        input_ids = batch_prompts["input_ids"].to(self.device)
        attention_mask = batch_prompts["attention_mask"].to(self.device)
        prompt_lengths = attention_mask.sum(dim=1)
        generated = self.policy.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=self.cfg.max_new_tokens, do_sample=True,
            temperature=1.0, top_k=50, pad_token_id=self.cfg.pad_token_id,
            repetition_penalty=self.repetition_penalty,
            logits_processor=[_SanitizeLogitsProcessor()],
        )
        attn_full = (generated != self.cfg.pad_token_id).long()
        return generated, attn_full, prompt_lengths

    @torch.no_grad()
    def compute_rewards_and_old_logprobs(self, full_ids, full_mask, prompt_lengths):
        self.policy.eval()
        self.ref_policy.eval()
        full_ids = full_ids.to(self.device)
        full_mask = full_mask.to(self.device)
        prompt_lengths = prompt_lengths.to(self.device)

        rewards = self.reward_model(full_ids, full_mask)
        if self.reward_clip is not None:
            rewards = torch.clamp(rewards, -self.reward_clip, self.reward_clip)

        logp_pi = response_log_probs(self.policy, full_ids, full_mask, prompt_lengths)
        logp_ref = response_log_probs(self.ref_policy, full_ids, full_mask, prompt_lengths)

        # Per-token average KL (not summed) — prevents scale explosion
        resp_token_counts = (full_mask.sum(dim=1) - prompt_lengths).clamp(min=1).float()
        kl_per_token = (logp_pi - logp_ref) / resp_token_counts
        total_rewards = rewards - self.kl_coef * kl_per_token
        total_rewards = self._normalize_rewards(total_rewards)
        return total_rewards.detach(), logp_pi.detach()

    def ppo_update(self, full_ids, full_mask, prompt_lengths, rewards, old_logprobs,
                   num_epochs=2):
        self.policy.train()
        self.value_head.train()
        full_ids = full_ids.to(self.device)
        full_mask = full_mask.to(self.device)
        prompt_lengths = prompt_lengths.to(self.device)
        rewards = rewards.to(self.device)
        old_logprobs = old_logprobs.to(self.device)

        with torch.no_grad():
            out = self.policy(input_ids=full_ids, attention_mask=full_mask,
                              output_hidden_states=True)
            hidden = out.hidden_states[-1]
            lengths = full_mask.sum(dim=1) - 1
            last_h = hidden[torch.arange(hidden.size(0), device=self.device), lengths]
        values = self.value_head(last_h).squeeze(-1)
        advantages = (rewards - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards.detach()
        eps = self.cfg.clip_eps

        for _ in range(num_epochs):
            logprobs = response_log_probs(self.policy, full_ids, full_mask, prompt_lengths)
            ratio = torch.clamp(torch.exp(logprobs - old_logprobs), 0.0, 10.0)
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - eps, 1 + eps) * advantages,
            ).mean()
            out = self.policy(input_ids=full_ids, attention_mask=full_mask,
                              output_hidden_states=True)
            hidden = out.hidden_states[-1]
            lengths = full_mask.sum(dim=1) - 1
            last_h = hidden[torch.arange(hidden.size(0), device=self.device), lengths]
            value_pred = self.value_head(last_h).squeeze(-1)
            value_loss = F.mse_loss(value_pred, returns)
            loss = policy_loss + 0.5 * value_loss
            if torch.isnan(loss):
                continue
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 0.5)
            self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()
