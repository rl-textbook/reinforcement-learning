"""Strict numeric verifier for the Chapter 6 RLVR case study."""

import re


class MathVerifier:
    def __init__(self, tol=1e-5):
        self.tol = tol
        self.marker = re.compile(r"FINAL:\s*([-+]?\d+(?:\.\d+)?)")

    def extract_answer(self, response_text: str):
        m = self.marker.search(response_text)
        if not m:
            return None
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            return None

    def reward(self, response_text: str, ground_truth_text: str):
        pred = self.extract_answer(response_text)
        if pred is None:
            return 0.0, "no-final-marker"
        try:
            gt = float(ground_truth_text.replace(",", ""))
        except Exception:
            return 0.0, "bad-ground-truth"
        ok = abs(pred - gt) <= self.tol
        return (1.0 if ok else 0.0), f"pred={pred}, gt={gt}"


__all__ = ["MathVerifier"]
