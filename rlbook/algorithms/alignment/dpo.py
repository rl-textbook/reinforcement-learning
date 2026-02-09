"""Chapter 5 — Direct Preference Optimization (Rafailov et al. 2023)."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .config import sequence_log_probs, response_log_probs


class DPOTrainer:
    """DPO trainer with gradient clipping for stable training."""

    def __init__(self, policy, ref_policy, beta=0.1, lr=5e-6):
        self.policy = policy
        self.ref_policy = ref_policy
        self.beta = beta
        self.optimizer = AdamW(policy.parameters(), lr=lr)

    def train_step(self, batch):
        self.policy.train()
        self.ref_policy.eval()
        c_ids, c_mask = batch["chosen_input_ids"], batch["chosen_attention_mask"]
        r_ids, r_mask = batch["rejected_input_ids"], batch["rejected_attention_mask"]
        pl = batch.get("prompt_lengths", None)

        if pl is not None:
            pi_c = response_log_probs(self.policy, c_ids, c_mask, pl)
            pi_r = response_log_probs(self.policy, r_ids, r_mask, pl)
            with torch.no_grad():
                ref_c = response_log_probs(self.ref_policy, c_ids, c_mask, pl)
                ref_r = response_log_probs(self.ref_policy, r_ids, r_mask, pl)
        else:
            pi_c = sequence_log_probs(self.policy, c_ids, c_mask)
            pi_r = sequence_log_probs(self.policy, r_ids, r_mask)
            with torch.no_grad():
                ref_c = sequence_log_probs(self.ref_policy, c_ids, c_mask)
                ref_r = sequence_log_probs(self.ref_policy, r_ids, r_mask)

        logits = self.beta * ((pi_c - pi_r) - (ref_c - ref_r))
        loss = -F.logsigmoid(logits).mean()
        margin = (self.beta * (pi_c - ref_c) - self.beta * (pi_r - ref_r)).mean().item()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        return loss.item(), margin
