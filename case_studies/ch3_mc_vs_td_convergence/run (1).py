"""Chapter 3 case study: Classical RL algorithms on the Windy Desert.
- imports algorithms and plotting helpers from the installed ``rlbook`` package
- reproduces the notebook's figure titles, layouts, and core hyperparameters
- saves all figures plus a summary.json to a local ``figures`` directory
- prints readable diagnostics in the terminal for command-line use
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# -----------------------------------------------------------------------------
# Import helpers
# -----------------------------------------------------------------------------

def _import_rlbook() -> Dict[str, Any]:
    """Import required rlbook APIs with a notebook-specific error message."""
    try:
        import importlib

        environments = importlib.import_module("rlbook.environments")
        dynamic_programming = importlib.import_module("rlbook.algorithms.dynamic_programming")
        tabular = importlib.import_module("rlbook.algorithms.tabular")
        core = importlib.import_module("rlbook.core")
        viz = importlib.import_module("rlbook.viz")
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import rlbook. From the repository root, run:\n"
            "  pip install -e .\n"
            "Then rerun this script, for example:\n"
            "  python case_studies/ch3_classical_algorithms/run.py\n"
            "or, if you are in the script directory:\n"
            "  python run.py"
        ) from exc

    required = {
        environments: ["create_windy_desert_env", "apply_reward_shaping"],
        dynamic_programming: ["value_iteration", "extract_policy"],
        tabular: [
            "make_epsilon_greedy_policy",
            "first_visit_mc_prediction",
            "td0_prediction",
            "q_learning",
            "sarsa",
            "extract_greedy_policy",
            "evaluate_policy",
        ],
        core: ["rmse", "VisitationCounter", "mc_td_traces", "run_value_evolution"],
        viz: [
            "visualize_grid_world_desert",
            "plot_policy_evolution",
            "visualize_policy_path",
            "plot_visitation_mc_vs_td",
            "plot_value_snapshots",
            "animate_value_evolution",
        ],
    }

    missing: list[str] = []
    for module, names in required.items():
        for name in names:
            if not hasattr(module, name):
                missing.append(f"{module.__name__}.{name}")

    if missing:
        joined = "\n  - ".join(missing)
        raise SystemExit(
            "Your installed rlbook package is missing APIs required by the "
            "package-based Chapter 3 notebook. Missing symbols:\n"
            f"  - {joined}"
        )

    exported: Dict[str, Any] = {}
    for module, names in required.items():
        for name in names:
            exported[name] = getattr(module, name)
    return exported


# -----------------------------------------------------------------------------
# CLI / formatting helpers
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Chapter 3 Classical RL case study: MC vs TD(0) prediction "
            "and Q-Learning vs SARSA on the Windy Desert."
        )
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory to write figures, animation HTML, and summary.json.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip saving the value-evolution animation HTML file.",
    )

    parser.add_argument(
        "--prediction-episodes",
        type=int,
        default=5000,
        help="Total episodes for MC and TD(0) prediction traces.",
    )
    parser.add_argument(
        "--prediction-snapshot-every",
        type=int,
        default=100,
        help="Snapshot interval for MC and TD(0) prediction traces.",
    )
    parser.add_argument(
        "--visitation-episodes",
        type=int,
        default=2000,
        help="Episodes for MC/TD visitation-frequency comparison.",
    )
    parser.add_argument(
        "--evolution-episodes",
        type=int,
        default=3000,
        help="Episodes for static/animated value-evolution views.",
    )
    parser.add_argument(
        "--evolution-snapshot-every",
        type=int,
        default=50,
        help="Snapshot interval for value-evolution recording.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Learning rate for TD(0), Q-learning, and SARSA.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor.",
    )
    parser.add_argument(
        "--epsilon-prediction",
        type=float,
        default=0.0,
        help="Epsilon for the fixed policy used by MC/TD prediction.",
    )
    parser.add_argument(
        "--epsilon-control",
        type=float,
        default=0.1,
        help="Exploration epsilon for Q-learning and SARSA.",
    )
    parser.add_argument(
        "--control-episodes",
        type=int,
        default=10000,
        help="Training episodes for Q-learning and SARSA.",
    )
    parser.add_argument(
        "--control-snapshots",
        type=str,
        default="50,200,500,10000",
        help="Comma-separated snapshot episodes for policy-evolution plots.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=400,
        help="Max steps per episode for control algorithms.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Evaluation episodes for final greedy policies.",
    )
    return parser


def banner(title: str) -> None:
    line = "=" * len(title)
    print(f"\n{line}\n{title}\n{line}")


def section(title: str) -> None:
    print(f"\n--- {title} ---")


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def parse_snapshot_list(text: str) -> tuple[int, ...]:
    values = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not values:
        raise SystemExit("--control-snapshots must contain at least one integer episode index.")
    try:
        parsed = tuple(int(v) for v in values)
    except ValueError as exc:
        raise SystemExit("--control-snapshots must be a comma-separated list of integers.") from exc
    if any(v <= 0 for v in parsed):
        raise SystemExit("--control-snapshots values must be positive integers.")
    return parsed


def to_builtin(obj: Any) -> Any:
    """Recursively convert numpy/scalar-heavy objects into JSON-safe builtins."""
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj



def _signature(fn: Any):
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None



def call_supported(fn: Any, /, *args, **kwargs):
    """Call a function after filtering unsupported keyword arguments.

    This keeps the runner resilient to small API differences across rlbook
    versions while still preferring the notebook-style call signatures.
    """
    sig = _signature(fn)
    if sig is None:
        return fn(*args, **kwargs)

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)



def call_with_fallbacks(fn: Any, calls: Iterable[tuple[tuple[Any, ...], dict[str, Any]]]):
    last_exc: Exception | None = None
    for args, kwargs in calls:
        try:
            return call_supported(fn, *args, **kwargs)
        except TypeError as exc:
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No fallback calls were provided.")



def coerce_figure(obj: Any):
    if obj is None:
        return plt.gcf()
    if hasattr(obj, "savefig"):
        return obj
    if hasattr(obj, "figure"):
        return obj.figure
    if isinstance(obj, (tuple, list)):
        for item in obj:
            fig = coerce_figure(item)
            if fig is not None:
                return fig
    return plt.gcf()



def save_figure(fig_like: Any, path: Path, show: bool = False) -> None:
    fig = coerce_figure(fig_like)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path.name}")
    if show:
        plt.show()
    plt.close(fig)



def save_animation_html(animation: Any, path: Path) -> None:
    html = animation.to_jshtml()
    page = (
        "<!doctype html>\n"
        "<html lang='en'>\n"
        "<head><meta charset='utf-8'><title>MC vs TD(0) Value Evolution</title></head>\n"
        f"<body>{html}</body>\n"
        "</html>\n"
    )
    path.write_text(page, encoding="utf-8")
    print(f"Saved: {path.name}")



def maybe_len(obj: Any) -> int | None:
    try:
        return len(obj)
    except TypeError:
        return None



def summarize_snapshots(snaps: Any) -> dict[str, Any]:
    info: dict[str, Any] = {"type": type(snaps).__name__}
    n = maybe_len(snaps)
    if n is not None:
        info["len"] = n
    if isinstance(snaps, Mapping):
        try:
            info["keys"] = [int(k) if str(k).isdigit() else str(k) for k in snaps.keys()]
        except Exception:
            info["keys"] = [str(k) for k in snaps.keys()]
    return info



def env_metadata(env: Any) -> dict[str, Any]:
    return {
        "grid_height": getattr(env, "grid_height", None),
        "grid_width": getattr(env, "grid_width", None),
        "num_states": len(getattr(env, "states", [])) if hasattr(env, "states") else None,
        "num_actions": len(getattr(env, "actions", [])) if hasattr(env, "actions") else None,
        "start_state": getattr(env, "start_state", None),
        "goal_state": getattr(env, "goal_state", None),
    }



def visualize_desert(modules: Dict[str, Any], *, env: Any, ax: Any | None = None,
                     values: Any | None = None, title: str | None = None,
                     show_legend: bool | None = None):
    visualize_grid_world_desert = modules["visualize_grid_world_desert"]
    kwargs = {"ax": ax, "title": title, "show_legend": show_legend, "values": values}
    return call_with_fallbacks(
        visualize_grid_world_desert,
        [
            ((), {"env": env, **kwargs}),
            ((env,), kwargs),
        ],
    )



def visualize_path(modules: Dict[str, Any], *, env: Any, policy: Any, ax: Any | None = None,
                   title: str | None = None, color: str | None = None):
    visualize_policy_path = modules["visualize_policy_path"]
    kwargs = {"ax": ax, "title": title, "color": color}
    return call_with_fallbacks(
        visualize_policy_path,
        [
            ((env, policy), kwargs),
            ((), {"env": env, "policy": policy, **kwargs}),
        ],
    )



def evaluate_greedy_policy(modules: Dict[str, Any], env: Any, policy: Any, episodes: int):
    evaluate_policy = modules["evaluate_policy"]
    sig = _signature(evaluate_policy)
    kwargs = {}
    if sig is None or "episodes" in sig.parameters:
        kwargs["episodes"] = episodes
    elif "num_episodes" in sig.parameters:
        kwargs["num_episodes"] = episodes
    return evaluate_policy(env, policy, **kwargs)



def train_control(modules: Dict[str, Any], fn_name: str, *, env: Any, episodes: int, alpha: float,
                  gamma: float, epsilon: float, snapshots: tuple[int, ...], max_steps: int):
    fn = modules[fn_name]
    sig = _signature(fn)
    kwargs: dict[str, Any] = {
        "episodes": episodes,
        "alpha": alpha,
        "gamma": gamma,
        "epsilon": epsilon,
        "max_steps": max_steps,
    }

    if sig is None or "snapshots" in sig.parameters:
        kwargs["snapshots"] = snapshots
    elif "snapshot_episodes" in sig.parameters:
        kwargs["snapshot_episodes"] = snapshots
    elif "snapshot_steps" in sig.parameters:
        kwargs["snapshot_steps"] = snapshots

    return fn(env, **kwargs)


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------

def main() -> int:
    args = build_parser().parse_args()
    control_snapshots = parse_snapshot_list(args.control_snapshots)
    modules = _import_rlbook()

    np.random.seed(args.seed)
    random.seed(args.seed)

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    banner("Chapter 3 Case Study: Classical RL Algorithms")
    print("MC vs TD(0) Prediction and Q-Learning vs SARSA on the Windy Desert")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {save_dir}")

    create_windy_desert_env = modules["create_windy_desert_env"]
    apply_reward_shaping = modules["apply_reward_shaping"]
    value_iteration = modules["value_iteration"]
    extract_policy = modules["extract_policy"]
    make_epsilon_greedy_policy = modules["make_epsilon_greedy_policy"]
    first_visit_mc_prediction = modules["first_visit_mc_prediction"]
    td0_prediction = modules["td0_prediction"]
    q_learning = modules["q_learning"]
    sarsa = modules["sarsa"]
    extract_greedy_policy = modules["extract_greedy_policy"]
    rmse = modules["rmse"]
    VisitationCounter = modules["VisitationCounter"]
    mc_td_traces = modules["mc_td_traces"]
    run_value_evolution = modules["run_value_evolution"]
    plot_policy_evolution = modules["plot_policy_evolution"]
    plot_visitation_mc_vs_td = modules["plot_visitation_mc_vs_td"]
    plot_value_snapshots = modules["plot_value_snapshots"]
    animate_value_evolution = modules["animate_value_evolution"]

    summary: Dict[str, Any] = {
        "seed": args.seed,
        "save_dir": str(save_dir),
        "figures": [],
        "parts": {},
    }

    # ------------------------------------------------------------------
    # Part 1/2: MC vs TD(0) Prediction on the default Windy Desert
    # ------------------------------------------------------------------
    section("Part 1/2 - MC vs TD(0) Prediction")
    env_pred = create_windy_desert_env()
    print(
        f"Prediction env: {getattr(env_pred, 'grid_height', '?')}x{getattr(env_pred, 'grid_width', '?')}"
        f", states={len(getattr(env_pred, 'states', []))}"
    )

    fig_layout = plt.figure(figsize=(7, 5))
    ax_layout = fig_layout.add_subplot(1, 1, 1)
    visualize_desert(
        modules,
        env=env_pred,
        ax=ax_layout,
        title="Windy Desert Layout",
        show_legend=True,
    )
    path = save_dir / "01_windy_desert_layout.png"
    save_figure(fig_layout, path, args.show)
    summary["figures"].append(path.name)

    V_star = value_iteration(env_pred)
    greedy_pi = extract_policy(env_pred, V_star)
    eps_policy = make_epsilon_greedy_policy(
        greedy_pi,
        getattr(env_pred, "actions", None),
        epsilon=args.epsilon_prediction,
    )
    print(
        f"Prediction config: epsilon={args.epsilon_prediction}, episodes={args.prediction_episodes}, "
        f"snapshot_every={args.prediction_snapshot_every}, alpha_td={args.alpha}, gamma={args.gamma}"
    )

    mc_snapshots, td_snapshots = mc_td_traces(
        eps_policy,
        env_pred,
        total_episodes=args.prediction_episodes,
        snapshot_every=args.prediction_snapshot_every,
        alpha_td=args.alpha,
        gamma=args.gamma,
    )
    V_mc_final = mc_snapshots[-1]
    V_td_final = td_snapshots[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    visualize_desert(
        modules,
        env=env_pred,
        ax=axes[0],
        title="Windy Desert Layout",
        show_legend=True,
    )
    visualize_desert(
        modules,
        env=env_pred,
        ax=axes[1],
        values=V_mc_final,
        title=f"MC Value (ε={args.epsilon_prediction})",
        show_legend=False,
    )
    visualize_desert(
        modules,
        env=env_pred,
        ax=axes[2],
        values=V_td_final,
        title=f"TD(0) Value (ε={args.epsilon_prediction})",
        show_legend=False,
    )
    fig.tight_layout()
    path = save_dir / "02_mc_vs_td_values.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    iters = list(range(args.prediction_snapshot_every, args.prediction_episodes + 1, args.prediction_snapshot_every))
    mc_err = [rmse(V, V_star) for V in mc_snapshots]
    td_err = [rmse(V, V_star) for V in td_snapshots]

    if len(iters) != len(mc_err):
        # Keep the notebook look, but adapt if the installed package records a different count.
        iters = list(range(1, len(mc_err) + 1))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(iters, mc_err, "o-", label="MC", markersize=3)
    ax.plot(iters, td_err, "s-", label="TD(0)", markersize=3)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMSE to V*")
    ax.set_title(f"MC vs TD(0) Convergence (ε={args.epsilon_prediction})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = save_dir / "03_mc_vs_td_rmse.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    mc_visit = VisitationCounter(len(env_pred.states))
    td_visit = VisitationCounter(len(env_pred.states))
    first_visit_mc_prediction(
        eps_policy,
        env_pred,
        num_episodes=args.visitation_episodes,
        gamma=args.gamma,
        visit=mc_visit,
    )
    td0_prediction(
        eps_policy,
        env_pred,
        num_episodes=args.visitation_episodes,
        gamma=args.gamma,
        visit=td_visit,
    )
    fig_like = plot_visitation_mc_vs_td(mc_visit, td_visit, env_pred, mode="freq")
    path = save_dir / "04_mc_td_visitation.png"
    save_figure(fig_like, path, args.show)
    summary["figures"].append(path.name)

    rec_mc, rec_td = run_value_evolution(
        env_pred,
        eps_policy,
        total_episodes=args.evolution_episodes,
        snapshot_every=args.evolution_snapshot_every,
        gamma=args.gamma,
        alpha_td=args.alpha,
    )
    fig_like = plot_value_snapshots(
        rec_mc,
        rec_td,
        checkpoints=[50, 500, 1500, 3000],
    )
    path = save_dir / "05_mc_td_value_snapshots.png"
    save_figure(fig_like, path, args.show)
    summary["figures"].append(path.name)

    animation_saved = None
    if not args.skip_animation:
        try:
            anim = animate_value_evolution(rec_mc, rec_td, interval_ms=40)
            animation_path = save_dir / "06_mc_td_value_evolution_animation.html"
            save_animation_html(anim, animation_path)
            animation_saved = animation_path.name
            plt.close("all")
            summary["figures"].append(animation_path.name)
        except Exception as exc:
            print(f"WARNING: Could not save animation HTML: {exc}", file=sys.stderr)

    mc_final_rmse = float(rmse(V_mc_final, V_star))
    td_final_rmse = float(rmse(V_td_final, V_star))
    print(f"Final MC RMSE:  {mc_final_rmse:.4f}")
    print(f"Final TD RMSE:  {td_final_rmse:.4f}")

    summary["parts"]["prediction"] = {
        "environment": env_metadata(env_pred),
        "epsilon": args.epsilon_prediction,
        "gamma": args.gamma,
        "alpha_td": args.alpha,
        "prediction_episodes": args.prediction_episodes,
        "prediction_snapshot_every": args.prediction_snapshot_every,
        "visitation_episodes": args.visitation_episodes,
        "evolution_episodes": args.evolution_episodes,
        "evolution_snapshot_every": args.evolution_snapshot_every,
        "mc_snapshot_count": len(mc_snapshots),
        "td_snapshot_count": len(td_snapshots),
        "mc_final_rmse": mc_final_rmse,
        "td_final_rmse": td_final_rmse,
        "animation_file": animation_saved,
    }

    # ------------------------------------------------------------------
    # Part 3: Q-Learning vs SARSA control with reward shaping
    # ------------------------------------------------------------------
    section("Part 3 - Q-Learning vs SARSA Control")
    env_ctrl = create_windy_desert_env(p_intended=0.90, p_perp=0.05)
    apply_reward_shaping(env_ctrl, step_cost=-1.0, hole_penalty=-100.0, goal_reward=+10.0)
    print(
        f"Control env: {getattr(env_ctrl, 'grid_height', '?')}x{getattr(env_ctrl, 'grid_width', '?')}"
        f", states={len(getattr(env_ctrl, 'states', []))}, p_intended=0.90, p_perp=0.05"
    )
    print(
        f"Control config: episodes={args.control_episodes}, alpha={args.alpha}, gamma={args.gamma}, "
        f"epsilon={args.epsilon_control}, snapshots={control_snapshots}, max_steps={args.max_steps}"
    )

    Q_q, snaps_q = train_control(
        modules,
        "q_learning",
        env=env_ctrl,
        episodes=args.control_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon_control,
        snapshots=control_snapshots,
        max_steps=args.max_steps,
    )
    Q_s, snaps_s = train_control(
        modules,
        "sarsa",
        env=env_ctrl,
        episodes=args.control_episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon_control,
        snapshots=control_snapshots,
        max_steps=args.max_steps,
    )

    fig_like = plot_policy_evolution(
        env_ctrl,
        snaps_q,
        title=f"Q-learning policy evolution (ε={args.epsilon_control})",
    )
    path = save_dir / "07_q_learning_policy_evolution.png"
    save_figure(fig_like, path, args.show)
    summary["figures"].append(path.name)

    fig_like = plot_policy_evolution(
        env_ctrl,
        snaps_s,
        title=f"SARSA policy evolution (ε={args.epsilon_control})",
    )
    path = save_dir / "08_sarsa_policy_evolution.png"
    save_figure(fig_like, path, args.show)
    summary["figures"].append(path.name)

    pi_q = extract_greedy_policy(Q_q, env_ctrl)
    pi_s = extract_greedy_policy(Q_s, env_ctrl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    visualize_path(
        modules,
        env=env_ctrl,
        policy=pi_q,
        color="C0",
        ax=axes[0],
        title="Q-learning: final representative path",
    )
    visualize_path(
        modules,
        env=env_ctrl,
        policy=pi_s,
        color="C1",
        ax=axes[1],
        title="SARSA: final representative path",
    )
    fig.tight_layout()
    path = save_dir / "09_q_vs_sarsa_final_paths.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    visualize_path(
        modules,
        env=env_ctrl,
        policy=pi_q,
        color="C0",
        ax=ax,
        title="Final representative paths: Q (blue) vs SARSA (orange)",
    )
    visualize_path(
        modules,
        env=env_ctrl,
        policy=pi_s,
        color="C1",
        ax=ax,
    )
    legend_lines = [
        Line2D([0], [0], color="C0", lw=3, label="Q-learning"),
        Line2D([0], [0], color="C1", lw=3, label="SARSA"),
    ]
    ax.legend(handles=legend_lines, loc="lower right")
    fig.tight_layout()
    path = save_dir / "10_q_vs_sarsa_path_overlay.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    ret_ql, steps_ql, success_ql = evaluate_greedy_policy(
        modules,
        env_ctrl,
        pi_q,
        args.eval_episodes,
    )
    ret_sa, steps_sa, success_sa = evaluate_greedy_policy(
        modules,
        env_ctrl,
        pi_s,
        args.eval_episodes,
    )

    print(
        f"Q-Learning ({args.control_episodes:,} eps): "
        f"return={float(ret_ql):.2f}, steps={float(steps_ql):.1f}, success={float(success_ql):.0f}%"
    )
    print(
        f"SARSA      ({args.control_episodes:,} eps): "
        f"return={float(ret_sa):.2f}, steps={float(steps_sa):.1f}, success={float(success_sa):.0f}%"
    )

    summary["parts"]["control"] = {
        "environment": env_metadata(env_ctrl),
        "reward_shaping": {
            "step_cost": -1.0,
            "hole_penalty": -100.0,
            "goal_reward": 10.0,
            "p_intended": 0.90,
            "p_perp": 0.05,
        },
        "episodes": args.control_episodes,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon_control,
        "max_steps": args.max_steps,
        "snapshot_episodes": list(control_snapshots),
        "q_learning_snapshots": summarize_snapshots(snaps_q),
        "sarsa_snapshots": summarize_snapshots(snaps_s),
        "evaluation_episodes": args.eval_episodes,
        "evaluation": {
            "q_learning": {
                "mean_return": float(ret_ql),
                "mean_steps": float(steps_ql),
                "success_rate": float(success_ql),
            },
            "sarsa": {
                "mean_return": float(ret_sa),
                "mean_steps": float(steps_sa),
                "success_rate": float(success_sa),
            },
        },
    }

    summary_path = save_dir / "summary.json"
    summary_path.write_text(json.dumps(to_builtin(summary), indent=2), encoding="utf-8")
    print(f"Saved: {summary_path.name}")

    print("\nSummary:")
    print(json.dumps(to_builtin(summary), indent=2))
    print(f"\nDone. All Chapter 3 outputs are in: {save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
