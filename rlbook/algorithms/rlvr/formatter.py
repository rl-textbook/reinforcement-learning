"""Chapter 6 — Prompt formatting for RLVR/RLHF-style reasoning tasks."""


class CoTFormatter:
    """Encourages step-by-step reasoning with a strict final answer marker."""

    def __init__(self) -> None:
        self.system = (
            "Solve the following math problem step by step. "
            "End your response with 'FINAL: <final numeric answer>'."
        )

    def format(self, question_text: str) -> str:
        return (
            f"{self.system}\n\n"
            f"Problem: {question_text}\n"
            f"Solution:\n"
        )


__all__ = ["CoTFormatter"]
