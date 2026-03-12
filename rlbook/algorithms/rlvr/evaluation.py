"""Chapter 6 — Evaluation helpers for the RLVR/GRPO case study."""

import re
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import extract_gt_number_text
from .grpo import build_response_mask_from_gen, sample_groups, sample_response

SELF_CORR_PAT = re.compile(
    r"\b(wait|actually|let me check|check again|recheck|double check|oops|mistake|correct myself)\b",
    re.IGNORECASE,
)


def heuristic_reasoning_steps(text: str) -> float:
    body = text.split("FINAL:")[0]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) >= 2:
        return float(len(lines))
    sents = [seg.strip() for seg in re.split(r"[\.!\?]\s+", body) if seg.strip()]
    return float(max(1, len(sents))) if body.strip() else 0.0


def heuristic_math_markers(text: str) -> float:
    ops = ["+", "-", "*", "/", "=", "×", "÷"]
    return float(sum(text.count(op) for op in ops))


def heuristic_self_correction(text: str) -> float:
    return 1.0 if SELF_CORR_PAT.search(text) else 0.0


# Backward-compatible names used by the older modular notebook.
estimate_reasoning_steps = heuristic_reasoning_steps


def operator_counts(text: str) -> Dict[str, int]:
    ops = {"+": 0, "-": 0, "*": 0, "/": 0, "=": 0, "×": 0, "÷": 0}
    for op in ops:
        ops[op] = text.count(op)
    return ops


@torch.no_grad()
def greedy_response(model, tokenizer, prompt_text: str, max_new_tokens: int = 128, device=None) -> str:
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.eval()
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    gen_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    response_ids = gen_ids[:, inputs["input_ids"].size(1):]
    return tokenizer.decode(response_ids[0], skip_special_tokens=True)


@torch.no_grad()
def evaluate_passk_and_diagnostics(
    model,
    tokenizer,
    verifier,
    formatter,
    dataset,
    n_eval: int,
    ks: Tuple[int, ...],
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_prompts: int = 8,
    max_prompt_tokens: Optional[int] = None,
    device=None,
    ground_truth_extractor: Optional[Callable[[str], str]] = None,
):
    """Evaluate Pass@k and reasoning diagnostics using grouped sampling."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    extractor = ground_truth_extractor or extract_gt_number_text

    model.eval()
    kmax = max(ks)
    n_eval = min(n_eval, len(dataset))

    successes = {k: 0 for k in ks}
    lengths = []
    steps = []
    math_markers = []
    self_corr = []
    parse_fail = 0
    exact = 0

    for start in range(0, n_eval, batch_prompts):
        end = min(n_eval, start + batch_prompts)
        batch = [dataset[i] for i in range(start, end)]

        prompts = [formatter.format(ex["question"]) for ex in batch]
        ground_truths = [extractor(ex["answer"]) for ex in batch]

        gen_ids, prompt_len, response_texts, _ = sample_groups(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            group_size=kmax,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed + start,
            max_prompt_tokens=max_prompt_tokens,
            device=device,
        )

        batch_size = len(batch)

        for i in range(batch_size):
            gt = ground_truths[i]
            group = response_texts[i * kmax:(i + 1) * kmax]
            rewards = []
            for response_text in group:
                reward_value, _ = verifier.reward(response_text, gt)
                rewards.append(reward_value)

            for k in ks:
                if any(rewards[:k]):
                    successes[k] += 1

            first_reward, first_reason = verifier.reward(group[0], gt)
            exact += int(first_reward == 1.0)
            parse_fail += int(first_reason == "no-final-marker")

            steps.append(heuristic_reasoning_steps(group[0]))
            math_markers.append(heuristic_math_markers(group[0]))
            self_corr.append(heuristic_self_correction(group[0]))

        response_mask_full = build_response_mask_from_gen(gen_ids, prompt_len, tokenizer.eos_token_id)
        first_indices = torch.arange(0, batch_size * kmax, kmax, device=gen_ids.device)
        response_len_tokens = response_mask_full[first_indices, prompt_len:].sum(dim=1).cpu().numpy()
        lengths.extend(response_len_tokens.tolist())

    passk = {k: successes[k] / max(1, n_eval) for k in ks}
    diag = {
        "len_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "steps": float(np.mean(steps)) if steps else 0.0,
        "math_markers": float(np.mean(math_markers)) if math_markers else 0.0,
        "selfcorr_rate": float(np.mean(self_corr)) if self_corr else 0.0,
    }
    extra = {
        "exact_k1": int(exact),
        "parse_fail_k1": int(parse_fail),
        "n_eval": int(n_eval),
    }
    return passk, diag, extra


def pass_at_k(
    model,
    tokenizer,
    verifier,
    formatter,
    dataset,
    k: int,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
    top_p: float = 0.95,
    batch_prompts: int = 8,
    max_prompt_tokens: Optional[int] = None,
    device=None,
    ground_truth_extractor: Optional[Callable[[str], str]] = None,
) -> float:
    """Backward-compatible Pass@k wrapper."""
    passk, _, _ = evaluate_passk_and_diagnostics(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        dataset=dataset,
        n_eval=len(dataset),
        ks=(k,),
        seed=0,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        batch_prompts=batch_prompts,
        max_prompt_tokens=max_prompt_tokens,
        device=device,
        ground_truth_extractor=ground_truth_extractor,
    )
    return passk[k]


@torch.no_grad()
def diagnostics_on_test(
    model,
    tokenizer,
    formatter,
    test_data,
    cfg,
    n_examples: Optional[int] = None,
    use_greedy: bool = True,
    device=None,
):
    """Backward-compatible diagnostics used by the older modular notebook."""
    data = test_data[:n_examples] if n_examples else test_data

    lengths, steps, op_markers, self_corr = [], [], [], []

    for ex in data:
        prompt = formatter.format(ex["question"])
        if use_greedy:
            response = greedy_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=cfg.max_new_tokens_eval,
                device=device,
            )
        else:
            response, _, _ = sample_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=cfg.max_new_tokens_eval,
                temperature=getattr(cfg, "temperature_eval", getattr(cfg, "temperature", 1.0)),
                top_p=getattr(cfg, "top_p_eval", getattr(cfg, "top_p", 0.95)),
                max_prompt_tokens=getattr(cfg, "max_prompt_tokens", None),
                device=device,
            )

        lengths.append(len(tokenizer(response).input_ids))
        steps.append(heuristic_reasoning_steps(response))
        op_markers.append(sum(operator_counts(response).values()))
        self_corr.append(heuristic_self_correction(response))

    return {
        "avg_len_tokens": float(np.mean(lengths)) if lengths else 0.0,
        "avg_steps": float(np.mean(steps)) if steps else 0.0,
        "avg_op_markers": float(np.mean(op_markers)) if op_markers else 0.0,
        "self_correction_rate": float(np.mean(self_corr)) if self_corr else 0.0,
    }


__all__ = [
    "SELF_CORR_PAT",
    "heuristic_reasoning_steps",
    "heuristic_math_markers",
    "heuristic_self_correction",
    "estimate_reasoning_steps",
    "operator_counts",
    "greedy_response",
    "evaluate_passk_and_diagnostics",
    "pass_at_k",
    "diagnostics_on_test",
]
