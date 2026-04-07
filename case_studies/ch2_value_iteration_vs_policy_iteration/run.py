"""Chapter 2 case study: Value Iteration vs Policy Iteration on Grid World.

This CLI script mirrors the Chapter 2 notebook at a high level while staying
repo-friendly:
- imports algorithms from the installed ``rlbook`` package
- saves figures into ``case_studies/ch2_value_iteration_vs_policy_iteration/figures``
- exits non-zero if VI and PI disagree on the tested environments
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


def _import_rlbook() -> Dict[str, Any]:
    """Import rlbook modules with a helpful error if the package is missing."""
    try:
        from rlbook.environments import create_grid_world
        from rlbook.algorithms.dynamic_programming import (
            value_iteration,
            extract_policy,
            policy_iteration,
            value_iteration_trace,
            policy_iteration_trace,
            run_value_iter_steps,
            run_policy_iter_steps,
        )
        from rlbook.core import PolicyAgent, RandomWalkAgent, compare_agents, run_episode
        from rlbook.viz import (
            START_COLOR,
            HOLE_COLOR,
            GOAL_COLOR,
            OPEN_CELL_COLOR,
            HEATMAP_CMAP,
            BAR_COLORS,
            add_state_legend,
            visualize_grid_world,
            visualize_agent_path,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import rlbook. From the repository root, run:\n"
            "  pip install -e .\n"
            "Then rerun:\n"
            "  python case_studies/ch2_value_iteration_vs_policy_iteration/run.py"
        ) from exc

    return {
        "create_grid_world": create_grid_world,
        "value_iteration": value_iteration,
        "extract_policy": extract_policy,
        "policy_iteration": policy_iteration,
        "value_iteration_trace": value_iteration_trace,
        "policy_iteration_trace": policy_iteration_trace,
        "run_value_iter_steps": run_value_iter_steps,
        "run_policy_iter_steps": run_policy_iter_steps,
        "PolicyAgent": PolicyAgent,
        "RandomWalkAgent": RandomWalkAgent,
        "compare_agents": compare_agents,
        "run_episode": run_episode,
        "START_COLOR": START_COLOR,
        "HOLE_COLOR": HOLE_COLOR,
        "GOAL_COLOR": GOAL_COLOR,
        "OPEN_CELL_COLOR": OPEN_CELL_COLOR,
        "HEATMAP_CMAP": HEATMAP_CMAP,
        "BAR_COLORS": BAR_COLORS,
        "add_state_legend": add_state_legend,
        "visualize_grid_world": visualize_grid_world,
        "visualize_agent_path": visualize_agent_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Chapter 2 Value Iteration vs Policy Iteration case study."
    )
    parser.add_argument("--theta", type=float, default=1e-3, help="Convergence threshold.")
    parser.add_argument("--vi-max-iterations", type=int, default=100, help="Max VI sweeps.")
    parser.add_argument("--pi-max-iterations", type=int, default=1000, help="Max PI iterations.")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes for agent evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
        help="Directory to write figures and summary.json.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    return parser


def save_figure(fig, path, show=False):
    import matplotlib.pyplot as plt

    # If fig is actually an Axes, get its figure
    if hasattr(fig, "figure"):
        fig = fig.figure

    fig.tight_layout()
    fig.savefig(path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def nonterminal_policy(policy: Dict[int, Any]) -> Dict[int, Any]:
    return {s: a for s, a in policy.items() if a is not None}


def compute_metrics(
    env: Any,
    policy: Dict[int, Any] | None,
    episodes: int,
    run_episode_fn: Any,
    policy_agent_cls: Any,
    random_agent_cls: Any,
) -> Dict[str, float]:
    """Evaluate one policy (or a random agent if policy is None) over several episodes."""
    agent = policy_agent_cls(env, policy) if policy is not None else random_agent_cls(env)
    rewards, steps, successes = [], [], []
    goal_state = env.states[-1].id
    for _ in range(episodes):
        total_reward, num_steps, path = run_episode_fn(env, agent, max_steps=100)
        rewards.append(float(total_reward))
        steps.append(int(num_steps))
        successes.append(1 if path[-1] == goal_state else 0)
    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "success_rate": float(100.0 * np.mean(successes)),
    }


def policy_match(env, pi_a: Dict[int, Any], pi_b: Dict[int, Any]) -> bool:
    non_terminal_states = [state.id for state in env.states if not state.terminal]
    return all(pi_a.get(s) == pi_b.get(s) for s in non_terminal_states)


def max_nonterminal_value_diff(env, v_a: Dict[int, float], v_b: Dict[int, float]) -> float:
    non_terminal_states = [state.id for state in env.states if not state.terminal]
    return max(abs(v_a[s] - v_b[s]) for s in non_terminal_states)


def plot_value_heatmaps(
    save_path: Path,
    title: str,
    matrices: list[np.ndarray],
    subplot_titles: list[str],
    *,
    grid_size: int,
    goal_state: int,
    hole_ids: set[int],
    heatmap_cmap,
    start_color,
    goal_color,
    hole_color,
    open_cell_color,
    add_state_legend,
    show: bool,
) -> None:
    fig, axes = plt.subplots(1, len(matrices), figsize=(6 * len(matrices), 5))
    if len(matrices) == 1:
        axes = [axes]

    for ax, matrix, subtitle in zip(axes, matrices, subplot_titles):
        im = ax.imshow(matrix, cmap=heatmap_cmap)

        for row in range(grid_size):
            for col in range(grid_size):
                ax.text(
                    col,
                    row,
                    f"{matrix[row, col]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    zorder=3,
                )

        semantic_states = [(0, start_color), (goal_state, goal_color)]
        semantic_states.extend((hid, hole_color) for hid in sorted(hole_ids))
        for state_id, facecolor in semantic_states:
            row, col = divmod(state_id, grid_size)
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=1.5,
                    zorder=2,
                )
            )

        if add_state_legend is not None:
            add_state_legend(
                ax,
                include_regular=True,
                regular_facecolor=open_cell_color,
                regular_label="Empty Cell",
                fontsize=8,
            )

        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(subtitle)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Value")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save_path, show)


def main() -> int:
    args = build_parser().parse_args()
    modules = _import_rlbook()

    np.random.seed(args.seed)
    random.seed(args.seed)

    create_grid_world = modules["create_grid_world"]
    value_iteration = modules["value_iteration"]
    extract_policy = modules["extract_policy"]
    policy_iteration = modules["policy_iteration"]
    value_iteration_trace = modules["value_iteration_trace"]
    policy_iteration_trace = modules["policy_iteration_trace"]
    run_value_iter_steps = modules["run_value_iter_steps"]
    run_policy_iter_steps = modules["run_policy_iter_steps"]
    PolicyAgent = modules["PolicyAgent"]
    RandomWalkAgent = modules["RandomWalkAgent"]
    compare_agents = modules["compare_agents"]
    run_episode = modules["run_episode"]
    visualize_grid_world = modules["visualize_grid_world"]
    visualize_agent_path = modules["visualize_agent_path"]

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "seed": args.seed,
        "theta": args.theta,
        "save_dir": str(save_dir),
        "cases": {},
    }

    # ------------------------------------------------------------------
    # Case 1: 4x4 deterministic grid
    # ------------------------------------------------------------------
    env4 = create_grid_world(grid_size=4, transition_noise=0.0)
    goal4 = env4.states[-1].id

    v_vi4 = value_iteration(env4, theta=args.theta, max_iterations=args.vi_max_iterations)
    pi_vi4 = extract_policy(env4, v_vi4)
    pi_pi4, v_pi4 = policy_iteration(env4, theta=args.theta, max_iterations=args.pi_max_iterations)

    match4 = policy_match(env4, pi_vi4, pi_pi4)
    max_diff4 = max_nonterminal_value_diff(env4, v_vi4, v_pi4)

    fig, ax = plt.subplots(figsize=(6, 6))
    visualize_grid_world(
        values=v_vi4,
        policy=pi_vi4,
        grid_size=4,
        goal_state=goal4,
        ax=ax,
        title="Value Iteration (4x4)",
    )
    save_figure(fig, save_dir / "01_value_iteration_4x4.png", args.show)

    fig, ax = plt.subplots(figsize=(6, 6))
    visualize_grid_world(
        values=v_pi4,
        policy=pi_pi4,
        grid_size=4,
        goal_state=goal4,
        ax=ax,
        title="Policy Iteration (4x4)",
    )
    save_figure(fig, save_dir / "02_policy_iteration_4x4.png", args.show)

    deltas_vi4, policies_vi4 = value_iteration_trace(
        env4, theta=args.theta, max_iterations=args.vi_max_iterations
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_deltas = [max(d, args.theta * 0.1) for d in deltas_vi4]
    ax.semilogy(range(1, len(plot_deltas) + 1), plot_deltas, "o-")
    ax.axhline(args.theta, linestyle="--", label=f"θ = {args.theta:g}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max Bellman Error (Δ)")
    ax.set_title("Value Iteration Convergence (4x4)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_figure(fig, save_dir / "03_vi_convergence_4x4.png", args.show)

    changes_pi4, policies_pi4 = policy_iteration_trace(
        env4, theta=args.theta, max_iterations=args.pi_max_iterations
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(changes_pi4) + 1), changes_pi4, "s-")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Policy Changes")
    ax.set_title("Policy Iteration Convergence (4x4)")
    ax.grid(True, alpha=0.3)
    save_figure(fig, save_dir / "04_pi_convergence_4x4.png", args.show)

    summary["cases"]["grid_4x4"] = {
        "policies_match_nonterminal": match4,
        "max_value_difference_nonterminal": max_diff4,
        "vi_iterations": len(deltas_vi4),
        "pi_iterations": len(changes_pi4),
    }

    # ------------------------------------------------------------------
    # Case 2: 5x5 deterministic grid with holes
    # ------------------------------------------------------------------
    grid_size = 5
    holes = [(1, 1), (2, 3), (3, 1)]
    hole_ids = {r * grid_size + c for r, c in holes}

    env5 = create_grid_world(grid_size=grid_size, transition_noise=0.0, holes=holes)
    goal5 = env5.states[-1].id

    v_vi5 = value_iteration(env5, theta=args.theta, max_iterations=args.vi_max_iterations)
    pi_vi5 = extract_policy(env5, v_vi5)
    pi_pi5, v_pi5 = policy_iteration(env5, theta=args.theta, max_iterations=args.pi_max_iterations)

    match5 = policy_match(env5, pi_vi5, pi_pi5)
    max_diff5 = max_nonterminal_value_diff(env5, v_vi5, v_pi5)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    visualize_grid_world(
        values=v_vi5,
        policy=pi_vi5,
        grid_size=grid_size,
        holes=hole_ids,
        goal_state=goal5,
        ax=axes[0],
        title="VI: 5×5 with Holes",
    )
    visualize_grid_world(
        values=v_pi5,
        policy=pi_pi5,
        grid_size=grid_size,
        holes=hole_ids,
        goal_state=goal5,
        ax=axes[1],
        title="PI: 5×5 with Holes",
    )
    fig.tight_layout()
    save_figure(fig, save_dir / "05_vi_vs_pi_policy_5x5.png", args.show)

    plot_value_heatmaps(
        save_dir / "06_vi_vs_pi_values_5x5.png",
        "Optimal Value Functions (5×5 Deterministic Grid World)",
        [
            np.array([v_vi5[s.id] for s in env5.states]).reshape(grid_size, grid_size),
            np.array([v_pi5[s.id] for s in env5.states]).reshape(grid_size, grid_size),
        ],
        ["Value Iteration", "Policy Iteration"],
        grid_size=grid_size,
        goal_state=goal5,
        hole_ids=hole_ids,
        heatmap_cmap=modules["HEATMAP_CMAP"],
        start_color=modules["START_COLOR"],
        goal_color=modules["GOAL_COLOR"],
        hole_color=modules["HOLE_COLOR"],
        open_cell_color=modules["OPEN_CELL_COLOR"],
        add_state_legend=modules["add_state_legend"],
        show=args.show,
    )

    steps_to_show = [1, 2, 3, 5, 10]
    fig, axes = plt.subplots(2, len(steps_to_show), figsize=(4 * len(steps_to_show), 8))
    for j, k in enumerate(steps_to_show):
        v_k = run_value_iter_steps(env5, k)
        pi_k = extract_policy(env5, v_k)
        visualize_grid_world(
            policy=pi_k,
            grid_size=grid_size,
            holes=hole_ids,
            goal_state=goal5,
            ax=axes[0, j],
            title=f"VI k = {k}",
        )

        pi_k2 = run_policy_iter_steps(env5, k)
        visualize_grid_world(
            policy=pi_k2,
            grid_size=grid_size,
            holes=hole_ids,
            goal_state=goal5,
            ax=axes[1, j],
            title=f"PI k = {k}",
        )

    axes[0, 0].set_ylabel("Value Iteration", fontsize=12)
    axes[1, 0].set_ylabel("Policy Iteration", fontsize=12)
    fig.suptitle("Policy Evolution: VI vs PI (5×5)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, save_dir / "07_policy_evolution_5x5.png", args.show)

    results = {
        "VI": compute_metrics(env5, pi_vi5, args.episodes, run_episode, PolicyAgent, RandomWalkAgent),
        "PI": compute_metrics(env5, pi_pi5, args.episodes, run_episode, PolicyAgent, RandomWalkAgent),
        "Random": compute_metrics(env5, None, args.episodes, run_episode, PolicyAgent, RandomWalkAgent),
    }

    labels = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    bar_colors = modules["BAR_COLORS"]

    axes[0].bar(labels, [results[label]["mean_reward"] for label in labels], color=bar_colors)
    axes[0].set_ylabel("Mean Reward")
    axes[0].set_title("Mean Reward")

    axes[1].bar(labels, [results[label]["mean_steps"] for label in labels], color=bar_colors)
    axes[1].set_ylabel("Mean Steps")
    axes[1].set_title("Mean Steps")

    axes[2].bar(labels, [results[label]["success_rate"] for label in labels], color=bar_colors)
    axes[2].set_ylabel("Success Rate (%)")
    axes[2].set_title("Success Rate")

    fig.suptitle("Agent Performance: 5×5 Grid with Holes", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, save_dir / "08_agent_performance_5x5.png", args.show)

    vi_agent = PolicyAgent(env5, pi_vi5)
    total_reward, num_steps, path = run_episode(env5, vi_agent, max_steps=100)

    fig = visualize_agent_path(
        env5,
        path,
        policy=pi_vi5,
        values=v_vi5,
        holes=hole_ids,
        title="VI Agent Path (5×5)",
    )
    # visualize_agent_path may or may not return a Figure; use current active figure if needed.
    if fig is None:
        fig = plt.gcf()
    save_figure(fig, save_dir / "09_vi_agent_path_5x5.png", args.show)

    summary["cases"]["grid_5x5_holes"] = {
        "holes": holes,
        "policies_match_nonterminal": match5,
        "max_value_difference_nonterminal": max_diff5,
        "agent_evaluation": results,
        "sample_vi_episode": {
            "reward": total_reward,
            "steps": num_steps,
            "path": path,
        },
    }

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved figures and summary to: {save_dir}")

    if not (match4 and match5):
        print("\nERROR: Value Iteration and Policy Iteration disagreed on at least one test case.", file=sys.stderr)
        return 1

    print("\nSuccess: VI and PI matched on both Chapter 2 test cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
