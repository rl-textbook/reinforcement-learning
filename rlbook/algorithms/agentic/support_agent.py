"""LLM-based support agent for email analysis (Chapter 7)."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class SupportAgent:
    """LLM agent for email analysis using GPT-2."""

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading GPT-2 on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained('gpt2').to(self.device)
        self.model.eval()

    def classify_urgency(self, email: str) -> str:
        """Simple urgency classification."""
        email_lower = email.lower()
        urgent_keywords = ['furious', 'urgent', 'immediately', 'asap',
                           'angry', 'terrible', '3rd time', 'again']
        if any(kw in email_lower for kw in urgent_keywords):
            return 'high'
        return 'normal'

    def classify_category(self, email: str) -> str:
        """Classify email category."""
        email_lower = email.lower()
        if any(w in email_lower for w in ['refund', 'money back', 'charged']):
            return 'refund'
        elif any(w in email_lower for w in ['damaged', 'defective', 'wrong', 'broken']):
            return 'replacement'
        elif any(w in email_lower for w in ['cancel', 'subscription']):
            return 'cancellation'
        elif any(w in email_lower for w in ['arrived', 'shipping', 'tracking', 'delivery']):
            return 'shipping'
        elif any(w in email_lower for w in ['fraud', 'hacked', 'unauthorized']):
            return 'fraud'
        else:
            return 'general'

    def suggest_initial_action(self, email: str) -> int:
        """Suggest initial action based on classification."""
        category = self.classify_category(email)
        category_actions = {
            'refund': 3, 'replacement': 4, 'cancellation': 5,
            'shipping': 4, 'fraud': 2, 'general': 0,
        }
        return category_actions.get(category, 0)

    def generate_plan(self, email: str, max_new_tokens: int = 50) -> str:
        """Generate investigation plan."""
        prompt = f"Email: {email}\n\nInvestigation plan:"
        inputs = self.tokenizer(prompt, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id)
        plan = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return plan.split('Investigation plan:')[-1].strip()
