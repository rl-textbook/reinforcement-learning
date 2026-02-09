"""
Direct Preference Optimization (DPO) Trainer.

Aligns language models directly from preference data without
a separate reward model. See Chapter 5, Section 5.4.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from rlbook.algorithms.alignment.config import sequence_log_probs, response_log_probs


class DPOTrainer:
    def __init__(self, policy, ref_policy, beta=0.1, lr=5e-6):
        self.policy = policy
        self.ref_policy = ref_policy
        self.beta = beta
        self.optimizer = AdamW(self.policy.parameters(), lr=lr)

    def train_step(self, batch):
        self.policy.train()
        self.ref_policy.eval()

        chosen_ids = batch["chosen_input_ids"]
        chosen_mask = batch["chosen_attention_mask"]
        rejected_ids = batch["rejected_input_ids"]
        rejected_mask = batch["rejected_attention_mask"]
        prompt_lengths = batch.get("prompt_lengths", None)

        if prompt_lengths is not None:
            pi_chosen = response_log_probs(self.policy, chosen_ids, chosen_mask, prompt_lengths)
            pi_rejected = response_log_probs(self.policy, rejected_ids, rejected_mask, prompt_lengths)
            with torch.no_grad():
                ref_chosen = response_log_probs(self.ref_policy, chosen_ids, chosen_mask, prompt_lengths)
                ref_rejected = response_log_probs(self.ref_policy, rejected_ids, rejected_mask, prompt_lengths)
        else:
            pi_chosen = sequence_log_probs(self.policy, chosen_ids, chosen_mask)
            pi_rejected = sequence_log_probs(self.policy, rejected_ids, rejected_mask)
            with torch.no_grad():
                ref_chosen = sequence_log_probs(self.ref_policy, chosen_ids, chosen_mask)
                ref_rejected = sequence_log_probs(self.ref_policy, rejected_ids, rejected_mask)

        pi_logratios = pi_chosen - pi_rejected
        ref_logratios = ref_chosen - ref_rejected
        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()

        implicit_reward_chosen = self.beta * (pi_chosen - ref_chosen)
        implicit_reward_rejected = self.beta * (pi_rejected - ref_rejected)
        reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean().item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item(), reward_margin
