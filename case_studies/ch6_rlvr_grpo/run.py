#!/usr/bin/env python3
"""Chapter 6 Case Study — RLVR with GRPO on GSM8K.

Run from the repo root after installing the package:

    pip install -e .
    python case_studies/ch6_rlvr_grpo/run.py

Optional dependencies for this chapter:

    pip install -U transformers datasets accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import set_seed

from rlbook.algorithms.rlvr import (
    CFG,
    CoTFormatter,
    MathVerifier,
    evaluate_passk_and_diagnostics,
    sample_one_per_prompt,
)
from rlbook.core.rlvr import load_model_and_tokenizer, run_grpo_training_loop, sft_train

DEFAULT_SEED = 42


def seed_everything(seed: int) -> None:
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Chapter 6 RLVR case study from the command line.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global random seed.")
    parser.add_argument("--outdir", type=Path, default=Path("images"), help="Directory for saved figures/results.")
    parser.add_argument("--model-name", type=str, default=None, help="Override cfg.model_name.")
    parser.add_argument("--train-n", type=int, default=None, help="Override cfg.train_n.")
    parser.add_argument("--test-n", type=int, default=None, help="Override cfg.test_n.")
    parser.add_argument("--sft-steps", type=int, default=None, help="Override cfg.sft_steps.")
    parser.add_argument("--grpo-steps", type=int, default=None, help="Override cfg.grpo_steps.")
    parser.add_argument("--eval-every", type=int, default=None, help="Override cfg.eval_every.")
    parser.add_argument("--eval-curve-n", type=int, default=None, help="Override cfg.eval_curve_n.")
    parser.add_argument("--prompts-per-step", type=int, default=None, help="Override cfg.prompts_per_step.")
    parser.add_argument("--group-size", type=int, default=None, help="Override cfg.group_size.")
    parser.add_argument(
        "--batch-prompts-eval",
        type=int,
        default=8,
        help="How many prompts to evaluate per batch when computing Pass@k/diagnostics.",
    )
    parser.add_argument(
        "--traj-subset-n",
        type=int,
        default=64,
        help="Held-out subset size for early-vs-late qualitative trajectory comparison.",
    )
    parser.add_argument(
        "--qualitative-k",
        type=int,
        default=5,
        help="How many early-fail/late-success examples to print.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing before training.",
    )
    return parser.parse_args()


def apply_overrides(cfg: CFG, args: argparse.Namespace) -> CFG:
    overrides = {
        "model_name": args.model_name,
        "train_n": args.train_n,
        "test_n": args.test_n,
        "sft_steps": args.sft_steps,
        "grpo_steps": args.grpo_steps,
        "eval_every": args.eval_every,
        "eval_curve_n": args.eval_curve_n,
        "prompts_per_step": args.prompts_per_step,
        "group_size": args.group_size,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def evaluate_stage(
    label: str,
    *,
    model,
    tokenizer,
    verifier,
    formatter,
    test_ds,
    cfg: CFG,
    batch_prompts_eval: int,
    device,
) -> Dict[str, Any]:
    curve_passk, curve_diag, curve_extra = evaluate_passk_and_diagnostics(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        dataset=test_ds,
        n_eval=min(cfg.eval_curve_n, len(test_ds)),
        ks=cfg.eval_ks_curve,
        seed=cfg.eval_seed,
        max_new_tokens=cfg.max_new_tokens_eval,
        temperature=cfg.temperature_eval,
        top_p=cfg.top_p_eval,
        batch_prompts=batch_prompts_eval,
        max_prompt_tokens=cfg.max_prompt_tokens,
        device=device,
    )
    print(f"[{label} | curve subset] Pass@k: {curve_passk} | diag: {curve_diag} | extra: {curve_extra}")

    full_passk, full_diag, full_extra = evaluate_passk_and_diagnostics(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        dataset=test_ds,
        n_eval=len(test_ds),
        ks=cfg.eval_ks_curve,
        seed=cfg.eval_seed,
        max_new_tokens=cfg.max_new_tokens_eval,
        temperature=cfg.temperature_eval,
        top_p=cfg.top_p_eval,
        batch_prompts=batch_prompts_eval,
        max_prompt_tokens=cfg.max_prompt_tokens,
        device=device,
    )
    print(f"[{label} | full test] Pass@k: {full_passk} | diag: {full_diag} | extra: {full_extra}")

    scaling_passk, _, _ = evaluate_passk_and_diagnostics(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        dataset=test_ds,
        n_eval=len(test_ds),
        ks=cfg.eval_ks_scaling,
        seed=cfg.eval_seed,
        max_new_tokens=cfg.max_new_tokens_eval,
        temperature=cfg.temperature_eval,
        top_p=cfg.top_p_eval,
        batch_prompts=batch_prompts_eval,
        max_prompt_tokens=cfg.max_prompt_tokens,
        device=device,
    )
    print(f"[{label} | compute scaling] Pass@k: {scaling_passk}")

    return {
        "curve_passk": curve_passk,
        "curve_diag": curve_diag,
        "curve_extra": curve_extra,
        "full_passk": full_passk,
        "full_diag": full_diag,
        "full_extra": full_extra,
        "scaling_passk": scaling_passk,
    }


def save_learning_curves(
    outdir: Path,
    *,
    eval_steps,
    grpo_pass1,
    grpo_pass4,
    grpo_pass8,
    base_passk_curve,
    sft_passk_curve,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)

    curves = [
        ("Pass@1", grpo_pass1, base_passk_curve[1], sft_passk_curve[1]),
        ("Pass@4", grpo_pass4, base_passk_curve[4], sft_passk_curve[4]),
        ("Pass@8", grpo_pass8, base_passk_curve[8], sft_passk_curve[8]),
    ]

    for ax, (title, y, base_y, sft_y) in zip(axes, curves):
        ax.plot(eval_steps, y, marker="o", label="GRPO")
        ax.axhline(base_y, linestyle="--", label="Base")
        ax.axhline(sft_y, linestyle="--", label="SFT")
        ax.set_title(title)
        ax.set_xlabel("GRPO step")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Pass@k")
    axes[0].legend(loc="lower right")

    plt.suptitle("Figure 1: Learning Curves (Pass@k vs GRPO steps)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    outpath = outdir / "CaseStudy_Fig1_LearningCurves_v3.png"
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved:", outpath)
    return outpath


def save_test_time_scaling(outdir: Path, *, base_scaling, sft_scaling, grpo_scaling) -> Path:
    samples = list(base_scaling.keys())

    base_y = [base_scaling[n] for n in samples]
    sft_y = [sft_scaling[n] for n in samples]
    grpo_y = [grpo_scaling[n] for n in samples]

    p = grpo_scaling[1]
    ideal = [1.0 - (1.0 - p) ** n for n in samples]

    fig = plt.figure(figsize=(8, 5))
    plt.plot(samples, base_y, marker="o", label="Base (measured)")
    plt.plot(samples, sft_y, marker="o", label="SFT (measured)")
    plt.plot(samples, grpo_y, marker="o", label="GRPO (measured)")
    plt.plot(samples, ideal, linestyle="--", label="GRPO ideal (independent samples)")

    plt.xlabel("N (samples at test time)")
    plt.ylabel("Success rate (Pass@N)")
    plt.title("Figure 2: Test-Time Compute Scaling (best-of-N)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    outpath = outdir / "CaseStudy_Fig2_TestTimeScaling.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved:", outpath)
    return outpath


def save_reasoning_quality_plot(
    outdir: Path,
    *,
    eval_steps,
    diag_len,
    diag_steps,
    diag_math,
    base_diag_curve,
    sft_diag_curve,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

    axes[0].plot(eval_steps, diag_len, marker="o", label="GRPO")
    axes[0].axhline(base_diag_curve["len_tokens"], linestyle="--", label="Base")
    axes[0].axhline(sft_diag_curve["len_tokens"], linestyle="--", label="SFT")
    axes[0].set_title("Length (tokens)")
    axes[0].set_xlabel("GRPO step")
    axes[0].set_ylabel("Mean")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(eval_steps, diag_steps, marker="o", label="GRPO")
    axes[1].axhline(base_diag_curve["steps"], linestyle="--", label="Base")
    axes[1].axhline(sft_diag_curve["steps"], linestyle="--", label="SFT")
    axes[1].set_title("Reasoning steps (heuristic)")
    axes[1].set_xlabel("GRPO step")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(eval_steps, diag_math, marker="o", label="GRPO")
    axes[2].axhline(base_diag_curve["math_markers"], linestyle="--", label="Base")
    axes[2].axhline(sft_diag_curve["math_markers"], linestyle="--", label="SFT")
    axes[2].set_title("Math markers (heuristic)")
    axes[2].set_xlabel("GRPO step")
    axes[2].grid(True, alpha=0.3)

    axes[0].legend(loc="upper left")
    plt.suptitle("Figure 4: Reasoning Quality Evolution (analysis-only)", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.92])

    outpath = outdir / "CaseStudy_Fig3_ReasoningQualityEvolution.png"
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print("Saved:", outpath)
    return outpath


def save_summary_and_results(
    outdir: Path,
    *,
    cfg: CFG,
    summary: pd.DataFrame,
    base_results: Dict[str, Any],
    sft_results: Dict[str, Any],
    grpo_results: Dict[str, Any],
    training_curves: Dict[str, Any],
) -> None:
    summary_path = outdir / "chapter6_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    json_ready = {
        "config": asdict(cfg),
        "base": base_results,
        "sft": sft_results,
        "grpo": grpo_results,
        "training_curves": training_curves,
    }
    json_path = outdir / "chapter6_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2)
    print("Saved:", json_path)


def print_trajectory_examples(
    *,
    early_prompts,
    early_gts,
    early_texts,
    early_rewards,
    late_prompts,
    late_gts,
    late_texts,
    late_rewards,
    k: int,
) -> None:
    candidates = [
        i
        for i in range(len(early_rewards))
        if (early_rewards[i] == 0.0) and (late_rewards[i] == 1.0)
    ]

    if len(candidates) == 0:
        print(
            "Did not find any examples where SFT fails and GRPO succeeds in the sampled held-out subset."
        )
        print("Tip: increase --traj-subset-n to search a larger subset.")
        candidates = [0]

    picked = candidates[:k]
    print(f"Found {len(candidates)} candidate(s). Showing {len(picked)} example(s).")

    for j, idx in enumerate(picked, start=1):
        print("\n" + "=" * 100)
        print(
            f"Example {j}/{len(picked)} | subset_idx={idx} | "
            f"SFT reward={early_rewards[idx]} | GRPO reward={late_rewards[idx]}"
        )
        print("-" * 100)
        print("Problem prompt (subset):")
        print(early_prompts[idx] if early_prompts else late_prompts[idx])

        print("\nGround truth (subset):")
        print(early_gts[idx] if early_gts else late_gts[idx])

        print("\n--- Early policy (SFT) ---\n")
        print(early_texts[idx].strip())

        print("\n--- Late policy (GRPO) ---\n")
        print(late_texts[idx].strip())


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    cfg = apply_overrides(CFG(), args)
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {cfg.model_name}")
    print(f"Figures/results will be written to: {outdir.resolve()}")

    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"].select(range(min(cfg.train_n, len(ds["train"]))))
    test_ds = ds["test"].select(range(min(cfg.test_n, len(ds["test"]))))
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    formatter = CoTFormatter()
    verifier = MathVerifier(tol=cfg.tol)

    tokenizer, model, amp_dtype, use_scaler, scaler, autocast_ctx = load_model_and_tokenizer(cfg, device)
    print(f"Loaded: {cfg.model_name}")
    print(f"dtype: {next(model.parameters()).dtype} | device: {next(model.parameters()).device}")
    print(f"AMP_DTYPE: {amp_dtype} | USE_SCALER: {use_scaler}")

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled.")

    print("\n=== Baseline evaluation (Base model) ===")
    base_results = evaluate_stage(
        "Base",
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        test_ds=test_ds,
        cfg=cfg,
        batch_prompts_eval=args.batch_prompts_eval,
        device=device,
    )

    print("\n=== Stage 1 — Supervised Fine-Tuning (SFT) ===")
    sft_train(
        model=model,
        tokenizer=tokenizer,
        formatter=formatter,
        train_data=train_ds,
        cfg=cfg,
        autocast_ctx=autocast_ctx,
        use_scaler=use_scaler,
        scaler=scaler,
        device=device,
    )

    print("\n=== Evaluate SFT ===")
    sft_results = evaluate_stage(
        "SFT",
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        test_ds=test_ds,
        cfg=cfg,
        batch_prompts_eval=args.batch_prompts_eval,
        device=device,
    )

    traj_subset_n = min(args.traj_subset_n, len(test_ds))
    early_prompts, early_gts, early_texts, early_rewards = sample_one_per_prompt(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        dataset=test_ds,
        n_eval=traj_subset_n,
        seed=cfg.eval_seed + 999,
        max_new_tokens=cfg.max_new_tokens_eval,
        temperature=cfg.temperature_eval,
        top_p=cfg.top_p_eval,
        max_prompt_tokens=cfg.max_prompt_tokens,
        device=device,
    )
    print(f"Cached early trajectories on {traj_subset_n} prompts.")
    print(f"Early success rate: {sum(early_rewards) / len(early_rewards):.3f}")

    print("\n=== Stage 2 — RLVR with GRPO ===")
    training_curves = run_grpo_training_loop(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        train_ds=train_ds,
        test_ds=test_ds,
        cfg=cfg,
        seed=args.seed,
        autocast_ctx=autocast_ctx,
        use_scaler=use_scaler,
        scaler=scaler,
        device=device,
        batch_prompts_eval=args.batch_prompts_eval,
    )

    print("\n=== Final evaluation (GRPO) ===")
    grpo_results = evaluate_stage(
        "GRPO",
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        test_ds=test_ds,
        cfg=cfg,
        batch_prompts_eval=args.batch_prompts_eval,
        device=device,
    )

    late_prompts, late_gts, late_texts, late_rewards = sample_one_per_prompt(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        dataset=test_ds,
        n_eval=traj_subset_n,
        seed=cfg.eval_seed + 999,
        max_new_tokens=cfg.max_new_tokens_eval,
        temperature=cfg.temperature_eval,
        top_p=cfg.top_p_eval,
        max_prompt_tokens=cfg.max_prompt_tokens,
        device=device,
    )

    summary = pd.DataFrame(
        [
            {
                "Model": "Base",
                "Pass@1": base_results["full_passk"][1],
                "Pass@4": base_results["full_passk"][4],
                "Pass@8": base_results["full_passk"][8],
                "LenTok": base_results["full_diag"]["len_tokens"],
                "Steps": base_results["full_diag"]["steps"],
                "SelfCorr": base_results["full_diag"]["selfcorr_rate"],
            },
            {
                "Model": "SFT (CoT)",
                "Pass@1": sft_results["full_passk"][1],
                "Pass@4": sft_results["full_passk"][4],
                "Pass@8": sft_results["full_passk"][8],
                "LenTok": sft_results["full_diag"]["len_tokens"],
                "Steps": sft_results["full_diag"]["steps"],
                "SelfCorr": sft_results["full_diag"]["selfcorr_rate"],
            },
            {
                "Model": "GRPO (RLVR)",
                "Pass@1": grpo_results["full_passk"][1],
                "Pass@4": grpo_results["full_passk"][4],
                "Pass@8": grpo_results["full_passk"][8],
                "LenTok": grpo_results["full_diag"]["len_tokens"],
                "Steps": grpo_results["full_diag"]["steps"],
                "SelfCorr": grpo_results["full_diag"]["selfcorr_rate"],
            },
        ]
    )

    print("\n=== Summary table (Base vs SFT vs GRPO) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    save_learning_curves(
        outdir,
        eval_steps=training_curves["eval_steps"],
        grpo_pass1=training_curves["grpo_pass1"],
        grpo_pass4=training_curves["grpo_pass4"],
        grpo_pass8=training_curves["grpo_pass8"],
        base_passk_curve=base_results["curve_passk"],
        sft_passk_curve=sft_results["curve_passk"],
    )
    save_test_time_scaling(
        outdir,
        base_scaling=base_results["scaling_passk"],
        sft_scaling=sft_results["scaling_passk"],
        grpo_scaling=grpo_results["scaling_passk"],
    )
    save_reasoning_quality_plot(
        outdir,
        eval_steps=training_curves["eval_steps"],
        diag_len=training_curves["diag_len"],
        diag_steps=training_curves["diag_steps"],
        diag_math=training_curves["diag_math"],
        base_diag_curve=base_results["curve_diag"],
        sft_diag_curve=sft_results["curve_diag"],
    )
    save_summary_and_results(
        outdir,
        cfg=cfg,
        summary=summary,
        base_results=base_results,
        sft_results=sft_results,
        grpo_results=grpo_results,
        training_curves=training_curves,
    )

    print("\n=== Example trajectories (early vs late policy) ===")
    print_trajectory_examples(
        early_prompts=early_prompts,
        early_gts=early_gts,
        early_texts=early_texts,
        early_rewards=early_rewards,
        late_prompts=late_prompts,
        late_gts=late_gts,
        late_texts=late_texts,
        late_rewards=late_rewards,
        k=args.qualitative_k,
    )


if __name__ == "__main__":
    main()
