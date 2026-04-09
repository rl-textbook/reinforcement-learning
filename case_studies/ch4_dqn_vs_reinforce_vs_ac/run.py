#!/usr/bin/env python3
"""CLI case study for Chapter 4 deep RL on CartPole.

This version is aligned with the Chapter 4 case-study plotting conventions while
keeping the cleaner, more statistically defensible parts of the CLI workflow.

What it produces
----------------
1) Evaluation-only learning curves.
   - Combined plot: mean return across seeds with IQR band.
   - Per-algorithm plots: thin seed traces + thick mean curve + optional IQR band.
2) Sample efficiency via capped AUC over a common early-training budget.
   - Mean capped AUC across seeds with std error bars.
3) Stability over time.
   - Across-seed IQR of evaluation return as a function of step bins.
4) CSV artifacts for raw logs, aggregated curves, capped-AUC tables, stability,
   and a compact summary table.

Run from the project root so that ``rlbook`` is importable, e.g.

    python run.py
    python run.py --smoke-test
    python run.py --include-double-dqn
    python run.py --from-jsonl results/ch4_cartpole/eval_logs.jsonl --no-run
"""

from __future__ import annotations

import argparse
import json
import math
import random as pyrandom
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# Use a non-interactive backend and keep CPU thread count modest for small MLPs.
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


ALGO_ORDER = ["ActorCritic", "DQN", "REINFORCE", "DoubleDQN"]
COLORS = {
    "ActorCritic": "tab:blue",
    "DQN": "tab:orange",
    "REINFORCE": "tab:green",
    "DoubleDQN": "tab:red",
}
DEFAULT_SEEDS = [101, 202, 303]
DEFAULT_SOLVED_THRESHOLD = 450.0
DEFAULT_TOTAL_STEP_BUDGET = 50_000
DEFAULT_EVAL_EVERY_STEPS = 1_000
DEFAULT_AUC_CAP_STEPS = 20_000
DEFAULT_STABILITY_BIN_SIZE = 1_000
SMOKE_EPISODES = 2_000
SMOKE_STEPS_PER_EPISODE = 500
SMOKE_EVAL_EVERY_STEPS = 1_000
SMOKE_EVAL_EPISODES = 5
SMOKE_TOTAL_STEP_BUDGET = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chapter 4 CLI case study: REINFORCE vs Actor-Critic vs DQN on CartPole."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results"),
        help="Base output directory for logs, tables, and figures.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="chapter4_cartpole_cli",
        help="Subdirectory name inside --outdir for this run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Training seeds to run, e.g. --seeds 101 202 303",
    )
    parser.add_argument(
        "--include-double-dqn",
        action="store_true",
        help="Also run the optional Double DQN ablation.",
    )
    parser.add_argument(
        "--smoke-test", "--quick",
        dest="smoke_test",
        action="store_true",
        help="Run a shorter smoke test. --quick is kept as a backwards-compatible alias.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Skip training and only regenerate tables/plots from --from-jsonl.",
    )
    parser.add_argument(
        "--from-jsonl",
        type=Path,
        default=None,
        help="Existing JSONL evaluation log file to analyze. Useful with --no-run.",
    )
    parser.add_argument(
        "--save-formats",
        type=str,
        default="png,pdf",
        help="Comma-separated figure formats to save, e.g. png or png,pdf.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Centered rolling window for mean/IQR curves. Default 1 = no smoothing.",
    )
    parser.add_argument(
        "--solved-threshold",
        type=float,
        default=DEFAULT_SOLVED_THRESHOLD,
        help="Solved threshold used for summary statistics and reference lines.",
    )
    parser.add_argument(
        "--auc-cap-steps",
        type=int,
        default=DEFAULT_AUC_CAP_STEPS,
        help="Cap used for the sample-efficiency AUC comparison.",
    )
    parser.add_argument(
        "--stability-bin-size",
        type=int,
        default=DEFAULT_STABILITY_BIN_SIZE,
        help="Bin size in steps for the stability-over-time plot.",
    )
    parser.add_argument(
        "--stability-max-steps",
        type=int,
        default=None,
        help="Maximum step shown in the stability-over-time plot. Default: full observed run.",
    )

    # RunConfig-style arguments
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--steps-per-episode", type=int, default=500)
    parser.add_argument("--eval-every-steps", type=int, default=DEFAULT_EVAL_EVERY_STEPS)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--total-step-budget", type=int, default=DEFAULT_TOTAL_STEP_BUDGET)

    # Learning rates
    parser.add_argument("--reinforce-lr", type=float, default=1e-3)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)

    # DQNConfig-style arguments
    parser.add_argument("--dqn-lr", type=float, default=2e-4)
    parser.add_argument("--dqn-batch-size", type=int, default=64)
    parser.add_argument("--dqn-replay-capacity", type=int, default=100_000)
    parser.add_argument("--dqn-target-sync-every", type=int, default=1_000)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay-steps", type=int, default=100_000)

    return parser.parse_args()


def canonical_formats(spec: str) -> list[str]:
    formats = [fmt.strip().lower() for fmt in spec.split(",") if fmt.strip()]
    if not formats:
        raise ValueError("At least one output format is required.")
    return formats


def ordered_algorithms(items: Iterable[str]) -> list[str]:
    present = set(items)
    ordered = [name for name in ALGO_ORDER if name in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def compact_steps_label(steps: int) -> str:
    if steps % 1_000 == 0:
        return f"{steps // 1_000}k"
    if steps >= 1_000:
        return f"{steps / 1_000:.1f}k"
    return f"{steps}"


def area_under_curve(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) == 0:
        return float("nan")
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_arr, x_arr))
    return float(np.trapz(y_arr, x_arr))


def rolling_mean(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(arr) == 0:
        return arr
    return (
        pd.Series(arr)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )


def ensure_run_dir(base: Path, run_name: str) -> Path:
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_jsonl(rows: Sequence[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def records_to_frame(records: Sequence[dict]) -> pd.DataFrame:
    if not records:
        raise ValueError("No log records were provided.")
    df = pd.DataFrame(records).copy()
    required = {"algo", "seed", "steps", "return_mean"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required log columns: {missing}")
    df["algo"] = df["algo"].astype(str)
    df["seed"] = df["seed"].astype(int)
    df["steps"] = df["steps"].astype(int)
    df["return_mean"] = df["return_mean"].astype(float)
    if "return_iqr" in df.columns:
        df["return_iqr"] = df["return_iqr"].astype(float)
    else:
        df["return_iqr"] = np.nan
    return df.sort_values(["algo", "seed", "steps"]).reset_index(drop=True)


def aggregate_evals(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["algo", "steps"])["return_mean"]
        .agg(
            mean="mean",
            median="median",
            seed_std=lambda s: float(np.std(s.to_numpy(dtype=np.float64), ddof=0)),
            q25=lambda s: float(np.percentile(s, 25)),
            q75=lambda s: float(np.percentile(s, 75)),
            min="min",
            max="max",
            n="count",
        )
        .reset_index()
    )
    return grouped.sort_values(["algo", "steps"]).reset_index(drop=True)


def smoothed_aggregate(agg: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return agg.copy()
    out = agg.copy()
    for col in ["mean", "median", "seed_std", "q25", "q75", "min", "max"]:
        out[col] = (
            out.groupby("algo", sort=False)[col]
            .transform(lambda s: rolling_mean(s.to_numpy(), window))
        )
    return out


def resolve_plot_cap(df: pd.DataFrame, requested_cap: int) -> int:
    observed_max = int(df["steps"].max()) if len(df) else requested_cap
    return max(1, min(int(requested_cap), observed_max))


def first_solved_step(seed_frame: pd.DataFrame, threshold: float) -> float:
    hit = seed_frame.loc[seed_frame["return_mean"] >= threshold, "steps"]
    if hit.empty:
        return float("nan")
    return float(hit.iloc[0])


def _deduplicate_curve(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    curve = pd.DataFrame({"steps": np.asarray(x, dtype=np.float64), "value": np.asarray(y, dtype=np.float64)})
    curve = curve.groupby("steps", as_index=False)["value"].mean().sort_values("steps")
    return curve["steps"].to_numpy(dtype=np.float64), curve["value"].to_numpy(dtype=np.float64)


def capped_auc(x: Sequence[float], y: Sequence[float], cap_steps: int) -> float:
    x_arr, y_arr = _deduplicate_curve(x, y)
    if len(x_arr) == 0:
        return float("nan")

    cap = float(cap_steps)

    # Make the integration start explicit.
    if x_arr[0] > 0.0:
        x_arr = np.concatenate([[0.0], x_arr])
        y_arr = np.concatenate([[y_arr[0]], y_arr])

    # Truncate or extend to the common cap.
    if cap < x_arr[0]:
        x_arr = np.array([0.0, cap], dtype=np.float64)
        y_arr = np.array([y_arr[0], y_arr[0]], dtype=np.float64)
    elif x_arr[-1] < cap:
        x_arr = np.concatenate([x_arr, [cap]])
        y_arr = np.concatenate([y_arr, [y_arr[-1]]])
    elif x_arr[-1] > cap:
        y_cap = float(np.interp(cap, x_arr, y_arr))
        keep = x_arr < cap
        x_arr = np.concatenate([x_arr[keep], [cap]])
        y_arr = np.concatenate([y_arr[keep], [y_cap]])

    return area_under_curve(x_arr, y_arr)


def build_sample_efficiency(df: pd.DataFrame, cap_steps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for algo in ordered_algorithms(df["algo"].unique()):
        algo_df = df.loc[df["algo"] == algo].sort_values(["seed", "steps"])
        for seed, seed_frame in algo_df.groupby("seed", sort=True):
            auc = capped_auc(seed_frame["steps"], seed_frame["return_mean"], cap_steps)
            rows.append(
                {
                    "algo": algo,
                    "seed": int(seed),
                    "auc_cap_steps": int(cap_steps),
                    "auc_cap": float(auc),
                }
            )

    per_seed = pd.DataFrame(rows)
    if per_seed.empty:
        return per_seed, pd.DataFrame(columns=["algo", "auc_mean", "auc_std", "n"])

    summary = (
        per_seed.groupby("algo")["auc_cap"]
        .agg(
            auc_mean="mean",
            auc_std=lambda s: float(np.std(s.to_numpy(dtype=np.float64), ddof=0)),
            n="count",
        )
        .reset_index()
    )
    summary = summary.sort_values("algo", key=lambda s: s.map({a: i for i, a in enumerate(ALGO_ORDER)}).fillna(999))
    return per_seed, summary.reset_index(drop=True)


def build_stability_over_time(df: pd.DataFrame, bin_size: int, max_steps: int) -> pd.DataFrame:
    if bin_size <= 0:
        raise ValueError("stability bin size must be positive")

    work = df.loc[df["steps"] <= max_steps, ["algo", "seed", "steps", "return_mean"]].copy()
    if work.empty:
        return pd.DataFrame(columns=["algo", "step_bin", "across_seed_iqr", "across_seed_std", "n_seeds"])

    work["step_bin"] = (work["steps"] // bin_size) * bin_size

    # If multiple evals from the same seed fall in one bin, collapse within-seed first.
    per_seed_bin = (
        work.groupby(["algo", "seed", "step_bin"], as_index=False)["return_mean"]
        .mean()
        .sort_values(["algo", "seed", "step_bin"])
    )

    rows: list[dict] = []
    for (algo, step_bin), group in per_seed_bin.groupby(["algo", "step_bin"], sort=True):
        values = group["return_mean"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "algo": str(algo),
                "step_bin": int(step_bin),
                "across_seed_iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
                "across_seed_std": float(np.std(values, ddof=0)),
                "n_seeds": int(len(values)),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["algo", "step_bin"]).reset_index(drop=True)


def build_summary(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    sample_eff_summary: pd.DataFrame,
    stability_df: pd.DataFrame,
    solved_threshold: float,
    auc_cap_steps: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    algo_names = ordered_algorithms(df["algo"].unique())

    for algo in algo_names:
        algo_seed_df = df.loc[df["algo"] == algo].sort_values(["seed", "steps"])
        algo_agg = agg.loc[agg["algo"] == algo].sort_values("steps")
        final_per_seed = algo_seed_df.groupby("seed", as_index=False).tail(1).sort_values("seed")
        final_returns = final_per_seed["return_mean"].to_numpy(dtype=np.float64)
        best_per_seed = (
            algo_seed_df.groupby("seed", as_index=False)["return_mean"]
            .max()
            .rename(columns={"return_mean": "best_return"})
            .sort_values("seed")
        )
        best_returns = best_per_seed["best_return"].to_numpy(dtype=np.float64)
        best_to_final_drop = best_returns - final_returns
        solved_steps = np.array(
            [first_solved_step(seed_frame, solved_threshold) for _, seed_frame in algo_seed_df.groupby("seed")],
            dtype=np.float64,
        )

        auc_row = sample_eff_summary.loc[sample_eff_summary["algo"] == algo]
        auc_mean = float(auc_row["auc_mean"].iloc[0]) if not auc_row.empty else float("nan")
        auc_std = float(auc_row["auc_std"].iloc[0]) if not auc_row.empty else float("nan")

        stability_algo = stability_df.loc[stability_df["algo"] == algo]
        mean_iqr = (
            float(stability_algo["across_seed_iqr"].mean())
            if not stability_algo.empty
            else float("nan")
        )

        mean_values = algo_agg["mean"].to_numpy(dtype=np.float64)
        max_step = int(algo_agg["steps"].max()) if len(algo_agg) else -1

        rows.append(
            {
                "Algorithm": algo,
                "Num Seeds": int(final_per_seed["seed"].nunique()),
                "Final Return Mean": float(np.mean(final_returns)),
                "Final Return Std": float(np.std(final_returns, ddof=0)),
                "Final Return IQR": float(np.percentile(final_returns, 75) - np.percentile(final_returns, 25)),
                "Best Return Mean": float(np.mean(best_returns)),
                "Best Return Std": float(np.std(best_returns, ddof=0)),
                "Best Return IQR": float(np.percentile(best_returns, 75) - np.percentile(best_returns, 25)),
                "Mean Best-to-Final Drop": float(np.mean(best_to_final_drop)),
                "Best Eval Mean": float(np.max(mean_values)) if len(mean_values) else float("nan"),
                f"AUC <= {auc_cap_steps}": auc_mean,
                f"AUC <= {auc_cap_steps} Std": auc_std,
                "Mean Across-Seed IQR": mean_iqr,
                "Solved Seed Fraction": float(np.mean(~np.isnan(solved_steps))),
                "Median Steps to Solve": float(np.nanmedian(solved_steps)) if np.any(~np.isnan(solved_steps)) else float("nan"),
                "Mean Steps to Solve": float(np.nanmean(solved_steps)) if np.any(~np.isnan(solved_steps)) else float("nan"),
                "Last Eval Step": max_step,
            }
        )

    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str, outdir: Path, formats: Sequence[str]) -> None:
    for fmt in formats:
        fig.savefig(outdir / f"{stem}.{fmt}", dpi=200, bbox_inches="tight")


def plot_combined_learning_curves(
    agg: pd.DataFrame,
    outdir: Path,
    formats: Sequence[str],
    solved_threshold: float,
    smooth_window: int,
) -> None:
    algos = ordered_algorithms(agg["algo"].unique())
    smoothed = smoothed_aggregate(agg, smooth_window)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in algos:
        algo_df = smoothed.loc[smoothed["algo"] == algo].sort_values("steps")
        x = algo_df["steps"].to_numpy(dtype=np.float64)
        y = algo_df["mean"].to_numpy(dtype=np.float64)
        q25 = algo_df["q25"].to_numpy(dtype=np.float64)
        q75 = algo_df["q75"].to_numpy(dtype=np.float64)
        color = COLORS.get(algo, None)

        ax.plot(x, y, lw=2.5, marker="o", ms=4, color=color, label=algo)
        ax.fill_between(x, q25, q75, alpha=0.16, color=color)

    ax.axhline(solved_threshold, color="gray", linestyle="--", alpha=0.8, label=f"Solved ({solved_threshold:.0f})")
    ax.set_title("CartPole-v1 — learning curves (evals only)")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Evaluation return")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "learning_curves_combined", outdir, formats)
    plt.close(fig)


def plot_algorithm_learning_curve(
    df: pd.DataFrame,
    agg: pd.DataFrame,
    algo: str,
    outdir: Path,
    formats: Sequence[str],
    solved_threshold: float,
    smooth_window: int,
) -> None:
    algo_df = df.loc[df["algo"] == algo].sort_values(["seed", "steps"])
    algo_agg = smoothed_aggregate(agg.loc[agg["algo"] == algo].sort_values("steps"), smooth_window)
    color = COLORS.get(algo, None)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (seed, seed_frame) in enumerate(algo_df.groupby("seed", sort=True)):
        label = "Seed traces" if idx == 0 else None
        ax.plot(
            seed_frame["steps"].to_numpy(dtype=np.float64),
            seed_frame["return_mean"].to_numpy(dtype=np.float64),
            lw=1.2,
            marker="o",
            ms=3,
            alpha=0.25,
            color=color,
            label=label,
        )

    x = algo_agg["steps"].to_numpy(dtype=np.float64)
    y = algo_agg["mean"].to_numpy(dtype=np.float64)
    q25 = algo_agg["q25"].to_numpy(dtype=np.float64)
    q75 = algo_agg["q75"].to_numpy(dtype=np.float64)
    max_step = int(np.max(x)) if len(x) else 0

    ax.fill_between(x, q25, q75, alpha=0.16, color=color, label="IQR across seeds")
    ax.plot(x, y, lw=3.0, marker="o", ms=4, color=color, label=algo)
    ax.axhline(solved_threshold, color="gray", linestyle="--", alpha=0.8)

    title = f"{algo} — learning curve (evals only)"
    if max_step > 0:
        title += f" (≤ {max_step:,} steps)"
    ax.set_title(title)
    ax.set_xlabel("Steps")
    ax.set_ylabel("Evaluation return")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    stem = f"learning_curve_{algo.replace(' ', '_')}"
    save_figure(fig, stem, outdir, formats)
    plt.close(fig)


def plot_final_performance(summary: pd.DataFrame, outdir: Path, formats: Sequence[str], solved_threshold: float) -> None:
    labels = summary["Algorithm"].tolist()
    terminal_means = summary["Final Return Mean"].to_numpy(dtype=np.float64)
    terminal_errs = summary["Final Return Std"].to_numpy(dtype=np.float64)
    best_means = summary["Best Return Mean"].to_numpy(dtype=np.float64)
    best_errs = summary["Best Return Std"].to_numpy(dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2,
        terminal_means,
        width=width,
        yerr=terminal_errs,
        capsize=6,
        color=[COLORS.get(a, None) for a in labels],
        alpha=0.9,
        label="Terminal",
    )
    ax.bar(
        x + width / 2,
        best_means,
        width=width,
        yerr=best_errs,
        capsize=6,
        color=[COLORS.get(a, None) for a in labels],
        alpha=0.35,
        edgecolor="black",
        linewidth=1.0,
        hatch="//",
        label="Best seen",
    )
    ax.axhline(solved_threshold, color="gray", linestyle="--", alpha=0.8, label=f"Solved ({solved_threshold:.0f})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Evaluation return")
    ax.set_title("Terminal vs best-seen performance across seeds")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "terminal_vs_best_performance", outdir, formats)
    plt.close(fig)


def plot_sample_efficiency(
    sample_eff_summary: pd.DataFrame,
    cap_steps: int,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    labels = sample_eff_summary["algo"].tolist()
    aucs = sample_eff_summary["auc_mean"].to_numpy(dtype=np.float64)
    errs = sample_eff_summary["auc_std"].to_numpy(dtype=np.float64)
    step_label = compact_steps_label(cap_steps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, aucs, yerr=errs, capsize=6, color=[COLORS.get(a, None) for a in labels], alpha=0.9)
    ax.set_ylabel(f"AUC up to {step_label} (higher = better sample efficiency)")
    ax.set_title(f"Sample efficiency (AUC ≤ {step_label})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "sample_efficiency", outdir, formats)
    plt.close(fig)


def plot_stability_over_time(
    stability_df: pd.DataFrame,
    bin_size: int,
    outdir: Path,
    formats: Sequence[str],
) -> None:
    algos = ordered_algorithms(stability_df["algo"].unique())
    bin_label = compact_steps_label(bin_size)

    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in algos:
        algo_df = stability_df.loc[stability_df["algo"] == algo].sort_values("step_bin")
        if algo_df.empty:
            continue
        ax.plot(
            algo_df["step_bin"].to_numpy(dtype=np.float64),
            algo_df["across_seed_iqr"].to_numpy(dtype=np.float64),
            lw=2.2,
            marker="o",
            ms=4,
            color=COLORS.get(algo, None),
            label=algo,
        )

    ax.set_title("Stability over time (lower is more stable)")
    ax.set_xlabel(f"Steps ({bin_label} bins)")
    ax.set_ylabel("Across-seed IQR of eval return")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, "stability_over_time", outdir, formats)
    plt.close(fig)


def print_summary(summary: pd.DataFrame, auc_cap_steps: int) -> None:
    auc_col = f"AUC <= {auc_cap_steps}"
    display_cols = [
        "Algorithm",
        "Final Return Mean",
        "Best Return Mean",
        "Mean Best-to-Final Drop",
        auc_col,
        "Mean Across-Seed IQR",
        "Solved Seed Fraction",
        "Median Steps to Solve",
    ]
    print("\n=== Summary ===")
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


def build_run_config(args: argparse.Namespace):
    from rlbook.algorithms.deep import RunConfig
    
    if args.smoke_test:
        return RunConfig(
            gamma=args.gamma,
            episodes=SMOKE_EPISODES,
            steps_per_episode=SMOKE_STEPS_PER_EPISODE,
            eval_every_steps=SMOKE_EVAL_EVERY_STEPS,
            eval_episodes=SMOKE_EVAL_EPISODES,
            total_step_budget=SMOKE_TOTAL_STEP_BUDGET,
        )
    
    return RunConfig(
        gamma=args.gamma,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        eval_every_steps=args.eval_every_steps,
        eval_episodes=args.eval_episodes,
        total_step_budget=args.total_step_budget,
    )

def build_dqn_config(args: argparse.Namespace, run_cfg, *, double_dqn: bool):
    from rlbook.algorithms.deep import DQNConfig

    eps_decay_steps = int(args.eps_decay_steps)

    return DQNConfig(
        gamma=args.gamma,
        lr=args.dqn_lr,
        replay_capacity=args.dqn_replay_capacity,
        batch_size=args.dqn_batch_size,
        target_sync_every=args.dqn_target_sync_every,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=eps_decay_steps,
        double_dqn=bool(double_dqn),
    )


def run_experiments(args: argparse.Namespace) -> list[dict]:
    from rlbook.environments import CartPoleWrapper
    from rlbook.algorithms.deep import run_actor_critic, run_dqn, run_reinforce

    run_cfg = build_run_config(args)
    dqn_cfg = build_dqn_config(args, run_cfg, double_dqn=False)
    ddqn_cfg = build_dqn_config(args, run_cfg, double_dqn=True) if args.include_double_dqn else None

    all_logs: list[dict] = []
    total_jobs = len(args.seeds) * (4 if args.include_double_dqn else 3)
    job_idx = 0

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")

        job_idx += 1
        print(f"[{job_idx}/{total_jobs}] REINFORCE")
        env = CartPoleWrapper(seed=seed)
        all_logs.extend(run_reinforce(seed, env, run_cfg, lr=args.reinforce_lr))
        if hasattr(env, "close"):
            env.close()

        job_idx += 1
        print(f"[{job_idx}/{total_jobs}] Actor-Critic")
        env = CartPoleWrapper(seed=seed)
        all_logs.extend(
            run_actor_critic(
                seed,
                env,
                run_cfg,
                actor_lr=args.actor_lr,
                critic_lr=args.critic_lr,
            )
        )
        if hasattr(env, "close"):
            env.close()

        job_idx += 1
        print(f"[{job_idx}/{total_jobs}] DQN")
        env = CartPoleWrapper(seed=seed)
        all_logs.extend(run_dqn(seed, env, run_cfg, dqn_cfg))
        if hasattr(env, "close"):
            env.close()

        if args.include_double_dqn:
            job_idx += 1
            print(f"[{job_idx}/{total_jobs}] Double DQN")
            env = CartPoleWrapper(seed=seed)
            all_logs.extend(run_dqn(seed, env, run_cfg, ddqn_cfg))
            if hasattr(env, "close"):
                env.close()

    return all_logs


def save_metadata(args: argparse.Namespace, outdir: Path) -> None:
    metadata = {
        "seeds": list(map(int, args.seeds)),
        "include_double_dqn": bool(args.include_double_dqn),
        "smoke_test": bool(args.smoke_test),
        "smooth_window": int(args.smooth_window),
        "solved_threshold": float(args.solved_threshold),
        "auc_cap_steps": int(args.auc_cap_steps),
        "stability_bin_size": int(args.stability_bin_size),
        "stability_max_steps": None if args.stability_max_steps is None else int(args.stability_max_steps),
        "run_config": None,
        "dqn_config": None,
        "double_dqn_config": None,
        "learning_rates": {
            "reinforce_lr": float(args.reinforce_lr),
            "actor_lr": float(args.actor_lr),
            "critic_lr": float(args.critic_lr),
        },
    }

    if not args.no_run:
        try:
            run_cfg = build_run_config(args)
            metadata["run_config"] = asdict(run_cfg)
            metadata["dqn_config"] = asdict(build_dqn_config(args, run_cfg, double_dqn=False))
            if args.include_double_dqn:
                metadata["double_dqn_config"] = asdict(build_dqn_config(args, run_cfg, double_dqn=True))
        except Exception as exc:  # pragma: no cover - defensive only
            metadata["config_error"] = f"{type(exc).__name__}: {exc}"

    with (outdir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()
    formats = canonical_formats(args.save_formats)
    run_dir = ensure_run_dir(args.outdir, args.run_name)
    save_metadata(args, run_dir)

    if args.no_run:
        if args.from_jsonl is None:
            raise SystemExit("--no-run requires --from-jsonl so there is data to analyze.")
        log_path = args.from_jsonl
        print(f"Loading existing logs from {log_path}")
        records = load_jsonl(log_path)
    else:
        records = run_experiments(args)
        log_path = run_dir / "eval_logs.jsonl"
        save_jsonl(records, log_path)
        print(f"\nSaved {len(records)} evaluation rows to {log_path}")

    df = records_to_frame(records)
    agg = aggregate_evals(df)

    auc_cap_steps = resolve_plot_cap(df, args.auc_cap_steps)
    stability_max_steps = (
        int(df["steps"].max())
        if args.stability_max_steps is None
        else resolve_plot_cap(df, args.stability_max_steps)
    )
    sample_eff_per_seed, sample_eff_summary = build_sample_efficiency(df, auc_cap_steps)
    stability_df = build_stability_over_time(df, args.stability_bin_size, stability_max_steps)
    summary = build_summary(
        df,
        agg,
        sample_eff_summary=sample_eff_summary,
        stability_df=stability_df,
        solved_threshold=args.solved_threshold,
        auc_cap_steps=auc_cap_steps,
    )

    df.to_csv(run_dir / "eval_logs.csv", index=False)
    agg.to_csv(run_dir / "eval_curves_aggregated.csv", index=False)
    sample_eff_per_seed.to_csv(run_dir / "sample_efficiency_per_seed.csv", index=False)
    sample_eff_summary.to_csv(run_dir / "sample_efficiency_summary.csv", index=False)
    stability_df.to_csv(run_dir / "stability_over_time.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)

    plot_combined_learning_curves(
        agg,
        outdir=run_dir,
        formats=formats,
        solved_threshold=args.solved_threshold,
        smooth_window=args.smooth_window,
    )
    for algo in ordered_algorithms(df["algo"].unique()):
        plot_algorithm_learning_curve(
            df,
            agg,
            algo=algo,
            outdir=run_dir,
            formats=formats,
            solved_threshold=args.solved_threshold,
            smooth_window=args.smooth_window,
        )
    plot_final_performance(
        summary,
        outdir=run_dir,
        formats=formats,
        solved_threshold=args.solved_threshold,
    )
    plot_sample_efficiency(
        sample_eff_summary,
        cap_steps=auc_cap_steps,
        outdir=run_dir,
        formats=formats,
    )
    plot_stability_over_time(
        stability_df,
        bin_size=args.stability_bin_size,
        outdir=run_dir,
        formats=formats,
    )

    print_summary(summary, auc_cap_steps=auc_cap_steps)
    print(f"\nArtifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
