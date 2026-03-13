"""Chapter 6 — Verifiable reward for numeric QA tasks."""

import re
from typing import Optional, Tuple


class MathVerifier:
    """Verifier for numeric QA tasks such as GSM8K.

    By default the verifier is strict: the model must emit a `FINAL:` marker.
    Set `require_final_marker=False` to recover the older fallback behavior
    that uses the last number in the response when the marker is missing.
    """

    def __init__(self, tol: float = 1e-5, require_final_marker: bool = True) -> None:
        self.tol = tol
        self.require_final_marker = require_final_marker
        self.marker = re.compile(r"FINAL:\s*([-+]?\d[\d,]*(?:\.\d+)?)")
        self.number = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")

    def extract_answer(self, response_text: str) -> Optional[float]:
        # Step 1: Prefer the explicit FINAL: marker when it is present.  
        marker_match = self.marker.search(response_text)
        if marker_match:
            try:
                return float(marker_match.group(1).replace(",", ""))
            except Exception:
                return None
                
        # Step 2: In the practical notebook we keep strict parsing on.
        if self.require_final_marker:
            return None

        # Step 3: Optional backward-compatible fallback: use the last parseable number.
        nums = self.number.findall(response_text)
        if not nums:
            return None
        try:
            return float(nums[-1].replace(",", ""))
        except Exception:
            return None

    def reward(self, response_text: str, ground_truth_text: str) -> Tuple[float, str]:
        # Step 3: Parse the model prediction under the FINAL: contract.
        marker_match = self.marker.search(response_text)
        pred = self.extract_answer(response_text)

        if pred is None:
            if self.require_final_marker and marker_match is None:
                return 0.0, "no-final-marker"
            return 0.0, "no-parse"

        # Step 4: Normalize the ground-truth answer and compare numerically.
        try:
            gt = float(str(ground_truth_text).replace(",", ""))
        except Exception:
            return 0.0, "bad-ground-truth"

        ok = abs(pred - gt) <= self.tol
        return (1.0 if ok else 0.0), f"pred={pred}, gt={gt}"


__all__ = ["MathVerifier"]
