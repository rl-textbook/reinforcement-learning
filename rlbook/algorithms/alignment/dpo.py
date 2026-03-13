"""Chapter 5 — Direct Preference Optimization (Rafailov et al. 2023)."""

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from .config import response_log_probs, sequence_log_probs


class DPOTrainer:
    """DPO trainer with gradient clipping for stable training."""

    def __init__(self, policy, ref_policy, beta=0.1, lr=5e-6):
        self.policy = policy
        self.ref_policy = ref_policy

        # beta controls how strongly the optimized policy is allowed to deviate
        # from the reference policy (smaller beta -> more conservative updates).
        self.beta = beta

        self.optimizer = AdamW(policy.parameters(), lr=lr)

    def train_step(self, batch):
        # Step 1: Put models in the correct modes for training.
        self.policy.train()
        self.ref_policy.eval()

        # Step 2: Unpack a batch of preference pairs (x, y_w, y_l).
        # "chosen"   corresponds to y_w (the preferred/winner response)
        # "rejected" corresponds to y_l (the less-preferred/loser response)
        c_ids, c_mask = batch["chosen_input_ids"], batch["chosen_attention_mask"]
        r_ids, r_mask = batch["rejected_input_ids"], batch["rejected_attention_mask"]
        pl = batch.get("prompt_lengths", None)

        # Step 3: Compute log-probabilities under the current policy pi_theta.
        # We need log pi_theta(y_w | x) and log pi_theta(y_l | x).
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

        # Step 4: Form the DPO preference logit from the theory.
        # The DPO Bradley-Terry model uses:
        #   beta * ( log pi_theta(y_w|x) - log pi_ref(y_w|x)
        #          - log pi_theta(y_l|x) + log pi_ref(y_l|x) )
        # which is equivalent to:
        #   beta * ( (pi_c - pi_r) - (ref_c - ref_r) )
        logits = self.beta * ((pi_c - pi_r) - (ref_c - ref_r))

        # Step 5: DPO loss.
        loss = -F.logsigmoid(logits).mean()

        # Step 6: Compute a useful diagnostic: the average implicit reward margin.
        margin = (self.beta * (pi_c - ref_c) - self.beta * (pi_r - ref_r)).mean().item()

        # Step 7: Backpropagate and update the policy parameters.
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)

        # Step 8: Apply one optimizer step (theta <- theta - alpha * grad).
        self.optimizer.step()

        # Step 9: Return the scalar loss and diagnostic margin for logging/monitoring.
        return loss.item(), margin
