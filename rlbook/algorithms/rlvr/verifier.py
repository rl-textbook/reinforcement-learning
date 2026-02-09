"""Chapter 6 — Verifiable reward: binary correctness for numeric QA."""

import re


class MathVerifier:
    """Verifier for numeric QA tasks (e.g., GSM8K-style).

    Reward: 1.0 if extracted numeric answer matches ground truth, else 0.0.
    Assumes the model ends with 'FINAL: <number>'.
    Falls back to the last number found if marker is missing.
    """

    def __init__(self, tol=1e-5):
        self.tol = tol
        self.marker = re.compile(r"FINAL:\s*(-?\d+(?:\.\d+)?)")
        self.number = re.compile(r"-?\d+(?:\.\d+)?")

    def extract_answer(self, response_text):
        m = self.marker.search(response_text)
        if m:
            return float(m.group(1))
        nums = self.number.findall(response_text)
        if nums:
            return float(nums[-1])
        return None

    def reward(self, response_text, ground_truth_text):
        pred = self.extract_answer(response_text)
        if pred is None:
            return 0.0, "no-parse"
        try:
            gt = float(ground_truth_text)
        except Exception:
            return 0.0, "bad-ground-truth"
        ok = abs(pred - gt) <= self.tol
        return (1.0 if ok else 0.0), f"pred={pred}, gt={gt}"
