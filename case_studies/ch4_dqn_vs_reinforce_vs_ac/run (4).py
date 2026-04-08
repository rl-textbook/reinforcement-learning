#!/usr/bin/env python3
"""CLI case study for Chapter 4 deep RL on CartPole.

This script expands the minimal Chapter 4 runner into a publication-ready command line
experiment driver. It can:

1) run REINFORCE, Actor-Critic, and DQN across multiple seeds,
2) save raw evaluation logs and a summary table,
3) generate cleaner evaluation-only learning curves,
4) report sample efficiency via learning-curve AUC, and
5) quantify stability across seeds.

The script is intentionally aligned with the polished Chapter 4 notebook:
- same core algorithms,
- same evaluation-centric logging,
- same default CartPole setup,
- optional Double DQN ablation.

Run from the project root so that ``rlbook`` is importable, e.g.

    python run.py
    python run.py --quick
    python run.py --include-double-dqn
    python run.py --from-jsonl results/ch4_cartpole/eval_logs.jsonl --no-run

Outputs are written to ``--outdir/--run-name`` by default.
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


COLORS = {
    "REINFORCE": "tab:orange",
    "ActorCritic": "tab:green",
    "DQN": "tab:blue",
    "DoubleDQN": "tab:purple",
}
DEFAULT_SEEDS = [101, 202, 303]
DEFAULT_SOLVED_THRESHOLD = 475.0


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
        "--quick",
        action="store_true",
        help="Use a much smaller step budget for a smoke test.",
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
        default=3,
        help="Centered rolling window used to smooth evaluation curves.",
    )
    parser.add_argument(
        "--solved-threshold",
        type=float,
        default=DEFAULT_SOLVED_THRESHOLD,
        help="Solved threshold used for summary statistics.",
    )

    # RunConfig-style arguments
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--steps-per-episode", type=int, default=500)
    parser.add_argument("--eval-every-steps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--total-step-budget", type=int, default=100_000)

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


def first_solved_step(seed_frame: pd.DataFrame, threshold: float) -> float:
    hit = seed_frame.loc[seed_frame["return_mean"] >= threshold, "steps"]
    if hit.empty:
        return float("nan")
    return float(hit.iloc[0])


def build_summary(df: pd.DataFrame, agg: pd.DataFrame, solved_threshold: float) -> pd.DataFrame:
    rows: list[dict] = []
    algo_names = sorted(
        df["algo"].unique(),
        key=lambda a: (0, list(COLORS).index(a)) if a in COLORS else (1, str(a)),
    )
    for algo in algo_names:
        algo_seed_df = df.loc[df["algo"] == algo].sort_values(["seed", "steps"])
        algo_agg = agg.loc[agg["algo"] == algo].sort_values("steps")
        final_per_seed = algo_seed_df.groupby("seed", as_index=False).tail(1).sort_values("seed")
        final_returns = final_per_seed["return_mean"].to_numpy(dtype=np.float64)
        solved_steps = np.array(
            [first_solved_step(seed_frame, solved_threshold) for _, seed_frame in algo_seed_df.groupby("seed")],
            dtype=np.float64,
        )
        step_values = algo_agg["steps"].to_numpy(dtype=np.float64)
        mean_values = algo_agg["mean"].to_numpy(dtype=np.float64)

        auc = area_under_curve(step_values, mean_values)
        max_step = float(step_values[-1]) if len(step_values) else float("nan")
        normalized_auc = auc / max_step if (not math.isnan(max_step) and max_step > 0) else float("nan")

        rows.append(
            {
                "Algorithm": algo,
                "Num Seeds": int(final_per_seed["seed"].nunique()),
                "Final Return Mean": float(np.mean(final_returns)),
                "Final Return Std": float(np.std(final_returns, ddof=0)),
                "Final Return IQR": float(np.percentile(final_returns, 75) - np.percentile(final_returns, 25)),
                "Best Eval Mean": float(np.max(mean_values)) if len(mean_values) else float("nan"),
                "AUC": auc,
                "Normalized AUC": normalized_auc,
                "Mean Seed Std": float(np.mean(algo_agg["seed_std"].to_numpy(dtype=np.float64))),
                "Solved Seed Fraction": float(np.mean(~np.isnan(solved_steps))),
                "Median Steps to Solve": float(np.nanmedian(solved_steps)) if np.any(~np.isnan(solved_steps)) else float("nan"),
                "Mean Steps to Solve": float(np.nanmean(solved_steps)) if np.any(~np.isnan(solved_steps)) else float("nan"),
                "Last Eval Step": int(max_step) if not math.isnan(max_step) else -1,
            }
        )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, stem: str, outdir: Path, formats: Sequence[str]) -> None:
    for fmt in formats:
        fig.savefig(outdir / f"{stem}.{fmt}", dpi=200, bbox_inches="tight")


def ordered_algorithms(items: Iterable[str]) -> list[str]:
    ordered = [name for name in COLORS if name in set(items)]
    extras = sorted(set(items) - set(ordered))
    return ordered + extras


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
        x = algo_df["steps"].to_numpy()
        y = algo_df["mean"].to_numpy()
        q25 = algo_df["q25"].to_numpy()
        q75 = algo_df["q75"].to_numpy()

        ax.plot(x, y, lw=2.5, marker="o", ms=4, color=COLORS.get(algo, None), label=algo)
        ax.fill_between(x, q25, q75, alpha=0.15, color=COLORS.get(algo, None))

    ax.axhline(solved_threshold, color="gray", linestyle="--", alpha=0.7, label=f"Solved ({solved_threshold:.0f})")
    ax.set_title("CartPole-v1 Learning Curves (evaluation-only, mean ± IQR across seeds)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Evaluation Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "learning_curves_combined", outdir, formats)
    plt.close(fig)


def plot_individual_learning_curves(
    agg: pd.DataFrame,
    outdir: Path,
    formats: Sequence[str],
    solved_threshold: float,
    smooth_window: int,
) -> None:
    algos = ordered_algorithms(agg["algo"].unique())
    smoothed = smoothed_aggregate(agg, smooth_window)
    fig, axes = plt.subplots(1, len(algos), figsize=(6 * len(algos), 5), squeeze=False)

    for ax, algo in zip(axes.flat, algos):
        algo_df = smoothed.loc[smoothed["algo"] == algo].sort_values("steps")
        x = algo_df["steps"].to_numpy()
        y = algo_df["mean"].to_numpy()
        q25 = algo_df["q25"].to_numpy()
        q75 = algo_df["q75"].to_numpy()

        ax.plot(x, y, lw=2.5, marker="o", ms=4, color=COLORS.get(algo, None))
        ax.fill_between(x, q25, q75, alpha=0.2, color=COLORS.get(algo, None))
        ax.axhline(solved_threshold, color="gray", linestyle="--", alpha=0.7)
        ax.set_title(algo)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Evaluation Return")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Algorithm Learning Curves (evaluation-only, mean ± IQR)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "learning_curves_individual", outdir, formats)
    plt.close(fig)


def plot_final_performance(summary: pd.DataFrame, outdir: Path, formats: Sequence[str], solved_threshold: float) -> None:
    labels = summary["Algorithm"].tolist()
    means = summary["Final Return Mean"].to_numpy(dtype=np.float64)
    errs = summary["Final Return Std"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, means, yerr=errs, capsize=6, color=[COLORS.get(a, None) for a in labels], alpha=0.85)
    ax.axhline(solved_threshold, color="gray", linestyle="--", alpha=0.7, label=f"Solved ({solved_threshold:.0f})")
    ax.set_ylabel("Final Evaluation Return")
    ax.set_title("Final Performance Across Seeds (mean ± std)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "final_performance_bar", outdir, formats)
    plt.close(fig)


def plot_sample_efficiency(summary: pd.DataFrame, outdir: Path, formats: Sequence[str]) -> None:
    labels = summary["Algorithm"].tolist()
    aucs = summary["AUC"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, aucs, color=[COLORS.get(a, None) for a in labels], alpha=0.85)
    ax.set_ylabel("Learning-Curve AUC")
    ax.set_title("Sample Efficiency (higher is better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "sample_efficiency_bar", outdir, formats)
    plt.close(fig)


def plot_seed_stability(summary: pd.DataFrame, outdir: Path, formats: Sequence[str]) -> None:
    labels = summary["Algorithm"].tolist()
    stability = summary["Mean Seed Std"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, stability, color=[COLORS.get(a, None) for a in labels], alpha=0.85)
    ax.set_ylabel("Mean Std of Eval Return Across Seeds")
    ax.set_title("Stability Across Seeds (lower is better)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_figure(fig, "stability_across_seeds_bar", outdir, formats)
    plt.close(fig)


def plot_dashboard(
    agg: pd.DataFrame,
    summary: pd.DataFrame,
    outdir: Path,
    formats: Sequence[str],
    solved_threshold: float,
    smooth_window: int,
) -> None:
    algos = summary["Algorithm"].tolist()
    smoothed = smoothed_aggregate(agg, smooth_window)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    # Panel 1: combined learning curves
    ax = axes[0]
    for algo in algos:
        algo_df = smoothed.loc[smoothed["algo"] == algo].sort_values("steps")
        x = algo_df["steps"].to_numpy()
        y = algo_df["mean"].to_numpy()
        q25 = algo_df["q25"].to_numpy()
        q75 = algo_df["q75"].to_numpy()
        ax.plot(x, y, lw=2.2, color=COLORS.get(algo, None), label=algo)
        ax.fill_between(x, q25, q75, alpha=0.12, color=COLORS.get(algo, None))
    ax.axhline(solved_threshold, color="gray", linestyle="--", alpha=0.7)
    ax.set_title("Learning Curves")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Evaluation Return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    # Panel 2: sample efficiency
    ax = axes[1]
    ax.bar(algos, summary["AUC"].to_numpy(dtype=np.float64), color=[COLORS.get(a, None) for a in algos], alpha=0.85)
    ax.set_title("Sample Efficiency")
    ax.set_ylabel("Learning-Curve AUC")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: stability
    ax = axes[2]
    ax.bar(
        algos,
        summary["Mean Seed Std"].to_numpy(dtype=np.float64),
        color=[COLORS.get(a, None) for a in algos],
        alpha=0.85,
    )
    ax.set_title("Seed Stability")
    ax.set_ylabel("Mean Std Across Seeds")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Chapter 4 CLI Case Study Summary", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, "summary_dashboard", outdir, formats)
    plt.close(fig)


def print_summary(summary: pd.DataFrame) -> None:
    display_cols = [
        "Algorithm",
        "Final Return Mean",
        "Final Return Std",
        "AUC",
        "Mean Seed Std",
        "Solved Seed Fraction",
        "Median Steps to Solve",
    ]
    print("\n=== Summary ===")
    print(summary[display_cols].to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


def patch_helper_compatibility() -> None:
    """Compatibility patch for the current helper package.

    The edited Chapter 4 helper zip uses ``random.choices`` in reinforce.py. If that
    module forgot to import ``random``, we inject Python's standard ``random`` module
    so the CLI still runs cleanly.
    """
    import rlbook.algorithms.deep.reinforce as reinforce_module

    if not hasattr(reinforce_module, "random"):
        reinforce_module.random = pyrandom


def build_run_config(args: argparse.Namespace):
    from rlbook.algorithms.deep import RunConfig

    episodes = 500 if args.quick else args.episodes
    steps_per_episode = 200 if args.quick else args.steps_per_episode
    eval_every_steps = 2_000 if args.quick else args.eval_every_steps
    eval_episodes = 3 if args.quick else args.eval_episodes
    total_step_budget = 10_000 if args.quick else args.total_step_budget

    return RunConfig(
        gamma=args.gamma,
        episodes=episodes,
        steps_per_episode=steps_per_episode,
        eval_every_steps=eval_every_steps,
        eval_episodes=eval_episodes,
        total_step_budget=total_step_budget,
    )


def build_dqn_config(args: argparse.Namespace):
    from rlbook.algorithms.deep import DQNConfig

    return DQNConfig(
        gamma=args.gamma,
        lr=args.dqn_lr,
        replay_capacity=args.dqn_replay_capacity,
        batch_size=args.dqn_batch_size,
        target_sync_every=args.dqn_target_sync_every,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        eps_decay_steps=args.eps_decay_steps,
        double_dqn=False,
    )


def run_experiments(args: argparse.Namespace) -> list[dict]:
    from rlbook.environments import CartPoleWrapper
    from rlbook.algorithms.deep import run_actor_critic, run_dqn, run_reinforce

    patch_helper_compatibility()

    run_cfg = build_run_config(args)
    dqn_cfg = build_dqn_config(args)

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
            from rlbook.algorithms.deep import DQNConfig, run_dqn

            job_idx += 1
            print(f"[{job_idx}/{total_jobs}] Double DQN")
            env = CartPoleWrapper(seed=seed)
            ddqn_cfg = DQNConfig(
                gamma=args.gamma,
                lr=args.dqn_lr,
                replay_capacity=args.dqn_replay_capacity,
                batch_size=args.dqn_batch_size,
                target_sync_every=args.dqn_target_sync_every,
                eps_start=args.eps_start,
                eps_end=args.eps_end,
                eps_decay_steps=args.eps_decay_steps,
                double_dqn=True,
            )
            all_logs.extend(run_dqn(seed, env, run_cfg, ddqn_cfg))
            if hasattr(env, "close"):
                env.close()

    return all_logs


def save_metadata(args: argparse.Namespace, outdir: Path) -> None:
    metadata = {
        "seeds": list(map(int, args.seeds)),
        "include_double_dqn": bool(args.include_double_dqn),
        "quick": bool(args.quick),
        "smooth_window": int(args.smooth_window),
        "solved_threshold": float(args.solved_threshold),
        "run_config": None,
        "dqn_config": None,
        "learning_rates": {
            "reinforce_lr": float(args.reinforce_lr),
            "actor_lr": float(args.actor_lr),
            "critic_lr": float(args.critic_lr),
        },
    }

    if not args.no_run:
        # Only materialize these configs when the helper package is expected to be importable.
        try:
            metadata["run_config"] = asdict(build_run_config(args))
            metadata["dqn_config"] = asdict(build_dqn_config(args))
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
    summary = build_summary(df, agg, solved_threshold=args.solved_threshold)

    df.to_csv(run_dir / "eval_logs.csv", index=False)
    agg.to_csv(run_dir / "eval_curves_aggregated.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)

    plot_combined_learning_curves(
        agg,
        outdir=run_dir,
        formats=formats,
        solved_threshold=args.solved_threshold,
        smooth_window=args.smooth_window,
    )
    plot_individual_learning_curves(
        agg,
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
    plot_sample_efficiency(summary, outdir=run_dir, formats=formats)
    plot_seed_stability(summary, outdir=run_dir, formats=formats)
    plot_dashboard(
        agg,
        summary,
        outdir=run_dir,
        formats=formats,
        solved_threshold=args.solved_threshold,
        smooth_window=args.smooth_window,
    )

    print_summary(summary)
    print(f"\nArtifacts written to: {run_dir}")


if __name__ == "__main__":
    main()
