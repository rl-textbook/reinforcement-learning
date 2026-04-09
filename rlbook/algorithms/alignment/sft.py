"""Chapter 5 — Supervised fine-tuning trainer."""

from torch.optim import AdamW


class SFTTrainer:
    """Minimal SFT trainer that implements the MLE / teacher-forcing objective. We maximize J_SFT(theta) = E_{(x,y)}[ log pi_theta(y|x) ]
    """

    def __init__(self, model, lr=5e-5):
        # Step 1: Store the policy model pi_theta (a causal LM) and set up AdamW
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=lr)

    def train_step(self, batch):
        # Step 2 (Get a demonstration mini-batch from D_SFT):
        self.model.train()
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        # Step 3 (Teacher-forced forward pass -> compute SFT loss):
        loss = outputs.loss  # corresponds to L_SFT(theta) on this mini-batch

        # Step 4: Compute gradients  delta_theta L_SFT(theta)  via backpropagation.
        self.optimizer.zero_grad()
        loss.backward()

        # Step 5: Apply one AdamW step: theta <- theta - alpha * AdamW(delta_theta L_SFT).
        self.optimizer.step()

        # Step 6 (Return scalar for logging/monitoring):
        return loss.item()

    def eval_step(self, batch):
        # Step 7 (Evaluate without learning):
        # Compute the same teacher-forced loss on held-out data.
        self.model.eval()
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        return outputs.loss.item()

