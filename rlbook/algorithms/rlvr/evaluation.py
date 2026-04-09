"""Evaluation helpers for the Chapter 6 RLVR case study."""

import re
from typing import Tuple

import numpy as np
import torch

from .config import extract_gt_number_text
from .grpo import build_response_mask_from_gen, sample_groups

SELF_CORR_RE = re.compile(
    r"\b(wait|actually|let me check|check again|recheck|double check|oops|mistake|correct myself)\b",
    re.I,
)


def heuristic_reasoning_steps(text: str) -> float:
    body = text.split("FINAL:")[0]
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if len(lines) >= 2:
        return float(len(lines))
    sents = [sent.strip() for sent in re.split(r"[\.!\?]\s+", body) if sent.strip()]
    return float(max(1, len(sents))) if body.strip() else 0.0


def heuristic_math_markers(text: str) -> float:
    ops = ["+", "-", "*", "/", "="]
    return float(sum(text.count(op) for op in ops))


def heuristic_self_correction(text: str) -> float:
    return 1.0 if SELF_CORR_RE.search(text) else 0.0


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
    max_prompt_tokens: int = 256,
    device="cpu",
):
    model.eval()
    kmax = max(ks)

    successes = {k: 0 for k in ks}
    lens = []
    steps = []
    mathm = []
    selfc = []
    parse_fail = 0
    exact = 0

    for start in range(0, n_eval, batch_prompts):
        end = min(n_eval, start + batch_prompts)
        batch = [dataset[i] for i in range(start, end)]
        prompts = [formatter.format(ex["question"]) for ex in batch]
        gts = [extract_gt_number_text(ex["answer"]) for ex in batch]

        gen_ids, prompt_len, resp_texts, _ = sample_groups(
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

        # Group by prompt (hf order: p0_k, p1_k, ...)
        for i in range(batch_size):
            gt = gts[i]
            group = resp_texts[i * kmax:(i + 1) * kmax]
            rewards = []
            for j in range(kmax):
                reward_value, _ = verifier.reward(group[j], gt)
                rewards.append(reward_value)
            for k in ks:
                if any(rewards[:k]):
                    successes[k] += 1

            # diagnostics from first sample
            first_reward, reason = verifier.reward(group[0], gt)
            exact += int(first_reward == 1.0)
            parse_fail += int(reason == "no-final-marker")

            steps.append(heuristic_reasoning_steps(group[0]))
            mathm.append(heuristic_math_markers(group[0]))
            selfc.append(heuristic_self_correction(group[0]))

        # length in tokens for first sample per prompt
        eos_id = tokenizer.eos_token_id
        if isinstance(eos_id, (list, tuple)):
            eos_id = int(eos_id[0])
        resp_mask_full = build_response_mask_from_gen(gen_ids, prompt_len, eos_id)
        idx_first = torch.arange(0, batch_size * kmax, kmax, device=device)
        resp_len_tok = resp_mask_full[idx_first, prompt_len:].sum(dim=1).cpu().numpy()
        lens.extend(resp_len_tok.tolist())

    passk = {k: successes[k] / n_eval for k in ks}
    diag = {
        "len_tokens": float(np.mean(lens)) if lens else 0.0,
        "steps": float(np.mean(steps)) if steps else 0.0,
        "math_markers": float(np.mean(mathm)) if mathm else 0.0,
        "selfcorr_rate": float(np.mean(selfc)) if selfc else 0.0,
    }
    extra = {"exact_k1": int(exact), "parse_fail_k1": int(parse_fail), "n_eval": int(n_eval)}
    return passk, diag, extra


@torch.no_grad()
def sample_one_per_prompt(
    model,
    tokenizer,
    verifier,
    formatter,
    dataset,
    n_eval: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_prompt_tokens: int = 256,
    device="cpu",
):
    model.eval()
    prompts = [formatter.format(dataset[i]["question"]) for i in range(n_eval)]
    gts = [extract_gt_number_text(dataset[i]["answer"]) for i in range(n_eval)]
    _, _, resp_texts, _ = sample_groups(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        group_size=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_prompt_tokens=max_prompt_tokens,
        device=device,
    )
    rewards = []
    for i in range(n_eval):
        reward_value, _ = verifier.reward(resp_texts[i], gts[i])
        rewards.append(reward_value)
    return prompts, gts, resp_texts, rewards


__all__ = [
    "SELF_CORR_RE",
    "heuristic_reasoning_steps",
    "heuristic_math_markers",
    "heuristic_self_correction",
    "evaluate_passk_and_diagnostics",
    "sample_one_per_prompt",
]
