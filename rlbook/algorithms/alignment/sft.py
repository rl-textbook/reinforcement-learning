"""Chapter 5 — Supervised fine-tuning trainer."""

import torch
from torch.optim import AdamW


class SFTTrainer:
    """Simple SFT trainer with causal-LM loss."""

    def __init__(self, model, lr=5e-5):
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=lr)

    def train_step(self, batch):
        self.model.train()
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def eval_step(self, batch):
        self.model.eval()
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss.item()
