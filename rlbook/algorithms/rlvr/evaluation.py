"""Chapter 6 — Evaluation: Pass@k and reasoning quality diagnostics."""

import re
import numpy as np
import torch

from .grpo import sample_response

SELF_CORR_PAT = re.compile(
    r"\b(check|quick check|wait|actually|oops|recheck)\b", re.IGNORECASE
)


def pass_at_k(model, tokenizer, verifier, formatter, dataset, k,
              max_new_tokens=256, temperature=1.0, top_p=0.95):
    """Compute Pass@k on a dataset.

    Args:
        dataset: Iterable of dicts with keys 'question', 'answer'.
        k: Number of samples per problem.

    Returns:
        Pass@k in [0, 1].
    """
    model.eval()
    successes = 0

    for ex in dataset:
        prompt = formatter.format(ex["question"])
        ok = False
        for _ in range(k):
            with torch.no_grad():
                resp_text, _, _ = sample_response(
                    model, tokenizer, prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            r, _ = verifier.reward(resp_text, ex["answer"])
            if r > 0.0:
                ok = True
                break
        successes += (1 if ok else 0)

    return successes / max(1, len(dataset))


def estimate_reasoning_steps(text):
    """Heuristic: max(non-empty lines, sentence-ish segments)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sent = [s.strip() for s in re.split(r"[\.!\?]+", text) if s.strip()]
    return max(len(lines), len(sent))


def operator_counts(text):
    """Count math operator markers in text."""
    ops = {"+": 0, "-": 0, "*": 0, "/": 0, "=": 0, "×": 0, "÷": 0}
    for k in ops:
        ops[k] = text.count(k)
    return ops


@torch.no_grad()
def greedy_response(model, tokenizer, prompt_text, max_new_tokens=128):
    """Generate a single greedy (deterministic) response."""
    device = next(model.parameters()).device
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    gen = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    resp_ids = gen[:, inputs["input_ids"].size(1):]
    return tokenizer.decode(resp_ids[0], skip_special_tokens=True)


@torch.no_grad()
def diagnostics_on_test(model, tokenizer, formatter, test_data, cfg,
                        n_examples=None, use_greedy=True):
    """Compute reasoning quality metrics on test data.

    Returns dict with: avg_len_tokens, avg_steps, avg_op_markers, self_correction_rate.
    """
    data = test_data[:n_examples] if n_examples else test_data

    lengths, steps, op_plus, self_corr = [], [], [], []

    for ex in data:
        prompt = formatter.format(ex["question"])
        if use_greedy:
            resp = greedy_response(model, tokenizer, prompt,
                                   max_new_tokens=cfg.max_new_tokens_eval)
        else:
            resp, _, _ = sample_response(
                model, tokenizer, prompt,
                max_new_tokens=cfg.max_new_tokens_eval,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
            )

        lengths.append(len(tokenizer(resp).input_ids))
        steps.append(estimate_reasoning_steps(resp))
        ops = operator_counts(resp)
        op_plus.append(sum(ops.values()))
        self_corr.append(1.0 if SELF_CORR_PAT.search(resp) else 0.0)

    return {
        "avg_len_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "avg_op_markers": float(np.mean(op_plus)) if op_plus else 0.0,
        "self_correction_rate": float(np.mean(self_corr)) if self_corr else 0.0,
    }
