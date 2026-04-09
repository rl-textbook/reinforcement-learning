"""Prompt formatting helpers for the Chapter 6 RLVR case study."""


class CoTFormatter:
    def __init__(self):
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
