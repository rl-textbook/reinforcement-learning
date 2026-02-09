"""Chapter 6 — RLVR/GRPO configuration."""

import re


class Config:
    """Hyperparameters for the RLVR pipeline: SFT → GRPO on GSM8K."""
    model_name = "gpt2"
    train_size = 512
    val_size = 64
    test_size = 128
    max_length = 512
    max_new_tokens_eval = 128

    # SFT baseline
    sft_batch_size = 4
    sft_steps = 200
    sft_lr = 5e-5

    # GRPO (RLVR)
    grpo_steps = 200
    grpo_lr = 5e-6
    group_size = 8
    clip_eps = 0.2
    adv_eps = 1e-8
    sync_old_every = 10

    # Sampling
    temperature = 1.0
    top_p = 0.95

    # Evaluation protocol (held-out test set only)
    eval_every = 25
    eval_ks_learning_curves = (1, 4, 8)
    eval_ks_compute_scaling = (1, 2, 4, 8, 16)


# ── GSM8K parsing helpers ─────────────────────────────────────────


def parse_gsm8k_answer(answer_text: str) -> str:
    """Extract the final numeric answer from GSM8K format ('#### 29')."""
    m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer_text)
    if not m:
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer_text)
        if not nums:
            return ""
        s = nums[-1]
    else:
        s = m.group(1)
    return s.replace(",", "")


def gsm8k_solution_to_cot_and_final(answer_text: str) -> str:
    """Convert GSM8K rationale + '####' into Chapter 6 'FINAL:' contract."""
    parts = answer_text.split("####")
    rationale = parts[0].strip()
    final = parse_gsm8k_answer(answer_text)
    if rationale:
        return rationale + "\n\nFINAL: " + final
    return "FINAL: " + final
