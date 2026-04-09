"""Chapter 5 — Reward model."""

import torch.nn as nn


"""Reward model r_phi(x, y). Implements a scalar reward function r_phi(x, y) for a (prompt, response) pair.
"""
class RewardModel(nn.Module):
    """Learned scalar reward from transformer hidden states."""
    def __init__(self, base_model, hidden_size):
        super().__init__()

        # Step 1: Use a pretrained transformer LM backbone as a feature extractor for [prompt; response].
        self.base_model = base_model

        # Step 2 (Add the scalar value head): Map a pooled hidden representation (size H) -> scalar reward.
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # Step 3 (Encode the concatenated sequence [x; y] with the backbone):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Step 4 (Extract token-level hidden states):
        # Because the backbone is an AutoModelForCausalLM, we request hidden states explicitly and then take the final layer.
        hidden = outputs.hidden_states[-1]              # (B, T, H)

        # Step 5 (Pool token states into a single sequence representation):
        mask_exp = attention_mask.unsqueeze(-1).float() # (B, T, 1)
        sum_h = (hidden * mask_exp).sum(dim=1)          # (B, H)
        count = mask_exp.sum(dim=1).clamp(min=1)        # (B, 1)
        mean_h = sum_h / count                          # (B, H)
        return self.value_head(mean_h).squeeze(-1)
