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
    optional mitigations (higher KL penalty, repetition penalty, reward clipping)."""

    def __init__(
        self,
        policy,
        ref_policy,
        reward_model,
        tokenizer,
        cfg,
        kl_coef=None,
        repetition_penalty=1.0,
        reward_clip=None,
    ):
        # Step 1: Store the core RLHF components (actor, reference anchor, and reward signal).
        self.policy = policy
        self.ref_policy = ref_policy
        self.reward_model = reward_model

        # Step 2: Store tokenization + hyperparameters used throughout rollouts and updates.
        self.tokenizer = tokenizer
        self.cfg = cfg

        # Step 3: Choose the compute device from the policy (everything else follows this device).
        self.device = next(policy.parameters()).device

        # Step 4: Set PPO/RLHF regularizers described in theory.
        self.kl_coef = kl_coef if kl_coef is not None else cfg.kl_coef_ppo
        self.repetition_penalty = repetition_penalty
        self.reward_clip = reward_clip

        # Step 5: Initialize the critic/value head V(s) used for advantage estimation.
        hidden_size = policy.transformer.wte.weight.size(1)
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)

        # Step 6: Create an optimizer over BOTH the actor (policy) and critic (value head).
        self.optimizer = AdamW(
            list(policy.parameters()) + list(self.value_head.parameters()),
            lr=cfg.ppo_lr,
        )

        # Step 7: Initialize running reward statistics for normalization.
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0

    def _update_reward_stats(self, rewards):
        # Step 1: Compute batch reward statistics from the current rollouts.
        batch_mean = rewards.mean().item()
        batch_var = rewards.var().item() if rewards.numel() > 1 else 1.0
        n = rewards.numel()

        # Step 2: Update running mean/variance using the new batch (online normalization).
        total = self.reward_count + n
        delta = batch_mean - self.reward_mean
        self.reward_mean += delta * n / max(total, 1)
        self.reward_var = (self.reward_var * self.reward_count + batch_var * n) / max(total, 1)
        self.reward_count = total

    def _normalize_rewards(self, rewards):
        # Step 3: Guard against NaN/Inf rewards before updating running statistics.
        if not torch.isfinite(rewards).all():
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=2.0, neginf=-2.0)

        # Step 4: Update running statistics using the current batch.
        self._update_reward_stats(rewards)

        # Step 5: Normalize rewards to approximately zero mean and unit variance.
        std = max(self.reward_var ** 0.5, 1e-8)
        normed = (rewards - self.reward_mean) / std

        # Step 6: Clip normalized rewards into a bounded range.
        # This limits the influence of rare extreme rewards on advantages and PPO gradients.
        return torch.clamp(normed, -5.0, 5.0)

    @torch.no_grad()
    def generate_responses(self, batch_prompts):
        self.policy.eval()

        input_ids = batch_prompts["input_ids"].to(self.device)
        attention_mask = batch_prompts["attention_mask"].to(self.device)

        # Step 1: Compute prompt lengths (# of non-pad prompt tokens).
        prompt_lengths = attention_mask.sum(dim=1)

        # Step 2: Sample a continuation for each prompt x.
        # In PPO terms, this produces the rollout/trajectory under the current policy.
        generated = self.policy.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            pad_token_id=self.cfg.pad_token_id,
            repetition_penalty=self.repetition_penalty,
            logits_processor=[_SanitizeLogitsProcessor()],
        )

        # Step 3: Build the attention mask for the full (prompt + sampled response) sequence.
        # This mask is used later when computing token log-probabilities, KL penalties, and value estimates.
        attn_full = (generated != self.cfg.pad_token_id).long()
        return generated, attn_full, prompt_lengths

    @torch.no_grad()
    def compute_rewards_and_old_logprobs(self, full_ids, full_mask, prompt_lengths):
        self.policy.eval()
        self.ref_policy.eval()

        full_ids = full_ids.to(self.device)
        full_mask = full_mask.to(self.device)
        prompt_lengths = prompt_lengths.to(self.device)

        # Step 1: Compute the raw reward-model score r_phi(x, y) for each sampled rollout.
        rewards = self.reward_model(full_ids, full_mask)

        # Step 1a: Optional reward clipping (practical stability tweak).
        if self.reward_clip is not None:
            rewards = torch.clamp(rewards, -self.reward_clip, self.reward_clip)

        # Step 2: Compute response-only log-probabilities under:
        # - the current policy pi_theta  (used for PPO's "old log-probs")
        # - the reference policy pi_ref  (used for the KL penalty)
        logp_pi = response_log_probs(self.policy, full_ids, full_mask, prompt_lengths)
        logp_ref = response_log_probs(self.ref_policy, full_ids, full_mask, prompt_lengths)

        # Step 3: Compute a per-token KL proxy (average over response length).
        resp_token_counts = (full_mask.sum(dim=1) - prompt_lengths).clamp(min=1).float()
        kl_per_token = (logp_pi - logp_ref) / resp_token_counts

        # Step 4: Shape the final scalar reward with KL regularization.
        # r_tilde(x,y) = r_phi(x,y) - kl_coef * KL_tok_proxy(x,y)
        total_rewards = rewards - self.kl_coef * kl_per_token

        # Step 5: Normalize/clamp rewards using running statistics.
        total_rewards = self._normalize_rewards(total_rewards)
        return total_rewards.detach(), logp_pi.detach()

    def ppo_update(self, full_ids, full_mask, prompt_lengths, rewards, old_logprobs, num_epochs=2):
        self.policy.train()
        self.value_head.train()
        full_ids = full_ids.to(self.device)
        full_mask = full_mask.to(self.device)
        prompt_lengths = prompt_lengths.to(self.device)
        rewards = rewards.to(self.device)
        old_logprobs = old_logprobs.to(self.device)

        # Step 1: Compute the value baseline V(s) for each rollout.
        # In theory, PPO uses a value function baseline to reduce variance in policy gradients.
        with torch.no_grad():
            out = self.policy(input_ids=full_ids, attention_mask=full_mask, output_hidden_states=True)
            hidden = out.hidden_states[-1]             # final-layer token embeddings (B, T, H)
            lengths = full_mask.sum(dim=1) - 1         # index of last non-pad token for each sample
            last_h = hidden[torch.arange(hidden.size(0), device=self.device), lengths]

        # Step 2: Map the sequence representation to a scalar value estimate V(s).
        values = self.value_head(last_h).squeeze(-1)

        # Step 3: Advantage estimation:
        # For sequence-level rewards, we use A = R - V as a simple baseline-corrected advantage.
        advantages = (rewards - values).detach()

        # Step 4: Normalize advantages (stabilizes PPO updates across batches).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Step 5: Define the return targets for the value head, returns = rewards.
        returns = rewards.detach()
        eps = self.cfg.clip_eps

        # Step 6: PPO update phase (repeat K epochs over the same rollout batch).
        for _ in range(num_epochs):
            # Step 7: Recompute current policy log-probabilities for the sampled responses.
            logprobs = response_log_probs(self.policy, full_ids, full_mask, prompt_lengths)

            # Step 8: Importance sampling ratio r(theta) = exp(log pi_theta - log pi_old).
            ratio = torch.clamp(torch.exp(logprobs - old_logprobs), 0.0, 10.0)

            # Step 9: PPO clipped surrogate objective.
            # Take min(r*A, clip(r, 1-eps, 1+eps)*A) to prevent overly large policy updates.
            policy_loss = -torch.min(
                ratio * advantages,
                torch.clamp(ratio, 1 - eps, 1 + eps) * advantages,
            ).mean()

            # Step 10: Value function update (critic regression).
            out = self.policy(input_ids=full_ids, attention_mask=full_mask, output_hidden_states=True)
            hidden = out.hidden_states[-1]
            lengths = full_mask.sum(dim=1) - 1
            last_h = hidden[torch.arange(hidden.size(0), device=self.device), lengths]
            value_pred = self.value_head(last_h).squeeze(-1)

            # Step 11: Value loss (mean-squared error between predicted values and returns).
            value_loss = F.mse_loss(value_pred, returns)

            # Step 12: Combined PPO loss (policy + weighted value loss).
            loss = policy_loss + 0.5 * value_loss
            if torch.isnan(loss):
                continue
            self.optimizer.zero_grad()
            loss.backward()

            # Step 13: Gradient clipping for stability (both actor and critic components).
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), 0.5)
            self.optimizer.step()

        return loss.item(), policy_loss.item(), value_loss.item()
