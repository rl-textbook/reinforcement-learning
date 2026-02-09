"""
Reward Model for preference-based learning.

Learns to predict scalar rewards from human preference data
using mean pooling over non-pad tokens (more stable than
last-token pooling with left-padding).
See Chapter 5, Section 5.3.
"""

import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # (B, T, H)
        # Mean pooling over non-pad tokens (more stable than last-token)
        mask_exp = attention_mask.unsqueeze(-1).float()  # (B, T, 1)
        sum_h = (hidden * mask_exp).sum(dim=1)           # (B, H)
        count = mask_exp.sum(dim=1).clamp(min=1)         # (B, 1)
        mean_h = sum_h / count                           # (B, H)
        return self.value_head(mean_h).squeeze(-1)
