"""
Proximal Policy Optimization (PPO) for Language Model Alignment.

Optimizes a language model policy using PPO with a learned reward
model and KL penalty against a reference policy.
See Chapter 5, Section 5.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import LogitsProcessor

from rlbook.algorithms.alignment.config import response_log_probs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _SanitizeLogitsProcessor(LogitsProcessor):
    """Replace NaN/Inf logits with large-but-finite values to prevent
    multinomial sampling crashes during PPO training."""

    def __call__(self, input_ids, scores):
        return torch.where(
            torch.isfinite(scores), scores, torch.full_like(scores, -1e9)
        )


class PPOTrainer:
    def __init__(self, policy, ref_policy, reward_model, tokenizer, cfg):
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.cfg = cfg

        hidden_size = policy.transformer.wte.weight.size(1)
        self.value_head = nn.Linear(hidden_size, 1).to(device)

        self.optimizer = AdamW(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.ppo_lr,
        )

        # Running stats for reward normalization
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
        self._update_reward_stats(rewards)
        std = max(self.reward_var ** 0.5, 1e-8)
        return (rewards - self.reward_mean) / std

    @torch.no_grad()
    def generate_responses(self, batch_prompts):
        self.policy.eval()
        input_ids = batch_prompts["input_ids"].to(device)
        attention_mask = batch_prompts["attention_mask"].to(device)

        prompt_lengths = attention_mask.sum(dim=1)

        generated = self.policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            pad_token_id=self.cfg.pad_token_id,
            logits_processor=[_SanitizeLogitsProcessor()],
        )
        attn_full = (generated != self.cfg.pad_token_id).long()
        return generated, attn_full, prompt_lengths

    @torch.no_grad()
    def compute_rewards_and_old_logprobs(self, full_ids, full_mask, prompt_lengths):
        self.policy.eval()
        self.ref_policy.eval()

        full_ids = full_ids.to(device)
        full_mask = full_mask.to(device)
        prompt_lengths = prompt_lengths.to(device)

        rewards = self.reward_model(full_ids, full_mask)

        logp_policy = response_log_probs(self.policy, full_ids, full_mask, prompt_lengths)
        logp_ref = response_log_probs(self.ref_policy, full_ids, full_mask, prompt_lengths)
        kl_penalty = logp_policy - logp_ref

        total_rewards = rewards - self.cfg.kl_coef_ppo * kl_penalty
        # Normalize rewards for stable PPO updates
        total_rewards = self._normalize_rewards(total_rewards)

        return total_rewards.detach(), logp_policy.detach()

    def ppo_update(self, full_ids, full_mask, prompt_lengths, rewards, old_logprobs,
                   num_epochs=2):
        self.policy.train()
        self.value_head.train()

        full_ids = full_ids.to(device)
        full_mask = full_mask.to(device)
        prompt_lengths = prompt_lengths.to(device)
        rewards = rewards.to(device)
        old_logprobs = old_logprobs.to(device)

        with torch.no_grad():
            outputs = self.policy(
                input_ids=full_ids, attention_mask=full_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            lengths = full_mask.sum(dim=1) - 1
            last_hidden = hidden[
                torch.arange(hidden.size(0), device=device), lengths
            ]

        values = self.value_head(last_hidden).squeeze(-1)
        advantages = (rewards - values).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = rewards.detach()

        eps = self.cfg.clip_eps

        for _ in range(num_epochs):
            logprobs = response_log_probs(
                self.policy, full_ids, full_mask, prompt_lengths
            )
            ratio = torch.clamp(torch.exp(logprobs - old_logprobs), 0.0, 10.0)
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * advantages
            ).mean()

            outputs = self.policy(
                input_ids=full_ids, attention_mask=full_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            lengths = full_mask.sum(dim=1) - 1
            last_hidden = hidden[
                torch.arange(hidden.size(0), device=device), lengths
            ]
            value_pred = self.value_head(last_hidden).squeeze(-1)
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
