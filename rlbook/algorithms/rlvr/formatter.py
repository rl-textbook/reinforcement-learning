"""Chapter 6 — Prompt formatting: enforce chain-of-thought + FINAL marker."""


class CoTFormatter:
    """Encourages step-by-step reasoning with a strict final answer marker.

    The verifier expects a parseable numeric answer after 'FINAL:'.
    """

    def __init__(self):
        self.system = (
            "Solve the following math problem step by step. "
            "End your response with 'FINAL: <final numeric answer>'."
        )

    def format(self, question_text):
        return (
            f"{self.system}\n\n"
            f"Problem: {question_text}\n"
            f"Solution:"
        )
