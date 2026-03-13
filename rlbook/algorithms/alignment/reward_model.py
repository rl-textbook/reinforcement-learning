"""Chapter 5 — Reward model with mean pooling over non-pad tokens."""

import torch.nn as nn


class RewardModel(nn.Module):
    """Learned scalar reward from transformer hidden states.

    Uses mean pooling over non-pad tokens (more stable than last-token
    pooling when left-padding is used).
    """

    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True,)
        hidden = outputs.hidden_states[-1]                  # (B, T, H)
        mask_exp = attention_mask.unsqueeze(-1).float()      # (B, T, 1)
        sum_h = (hidden * mask_exp).sum(dim=1)               # (B, H)
        count = mask_exp.sum(dim=1).clamp(min=1)             # (B, 1)
        mean_h = sum_h / count                               # (B, H)
        return self.value_head(mean_h).squeeze(-1)
