
"""Chapter 3 case study: Classical RL algorithms on the Windy Desert.

This CLI runner follows the Chapter 2 run.py style, but mirrors the Chapter 3
colab's *specific* plots and pedagogy:
- Windy Desert layout
- MC vs TD(0) final value snapshots
- MC vs TD(0) RMSE convergence
- Q-learning policy evolution
- SARSA policy evolution
- Final representative paths for Q-learning and SARSA

Compared with earlier attempts, this script deliberately avoids relying on
notebook-oriented plotting helpers from rlbook.viz. The learning algorithms still
come from the installed ``rlbook`` package, but the figures are rendered here in
a deterministic, CLI-friendly way so saved PNGs are stable and non-blank.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, Rectangle


def _import_rlbook() -> Dict[str, Any]:
    """Import only the rlbook APIs this CLI runner actually needs."""
    try:
        from rlbook.environments import create_windy_desert_env
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Could not import rlbook. From the repository root, run:\n"
            "  pip install -e .\n"
            "Then rerun:\n"
            "  python case_studies/ch3_mc_td_qlearning_sarsa/run.py"
        ) from exc

    # Optional import in case the installed package exports the helper.
    try:
        from rlbook.environments import apply_reward_shaping as pkg_apply_reward_shaping
    except Exception:
        pkg_apply_reward_shaping = None

    try:
        from rlbook.algorithms.dynamic_programming import value_iteration, extract_policy
        from rlbook.algorithms.tabular import (
            make_epsilon_greedy_policy,
            q_learning,
            sarsa,
            extract_greedy_policy,
            evaluate_policy,
        )
        from rlbook.core import rmse, mc_td_traces
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Your installed rlbook package is missing the Chapter 3 algorithm APIs.\n"
            "From the repository root, try:\n"
            "  pip install -e ."
        ) from exc

    return {
        "create_windy_desert_env": create_windy_desert_env,
        "apply_reward_shaping": pkg_apply_reward_shaping,
        "value_iteration": value_iteration,
        "extract_policy": extract_policy,
        "make_epsilon_greedy_policy": make_epsilon_greedy_policy,
        "q_learning": q_learning,
        "sarsa": sarsa,
        "extract_greedy_policy": extract_greedy_policy,
        "evaluate_policy": evaluate_policy,
        "rmse": rmse,
        "mc_td_traces": mc_td_traces,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Chapter 3 MC/TD and Q-Learning/SARSA Windy Desert case study."
    )
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
        "--alpha-td",
        type=float,
        default=0.1,
        help="TD(0) learning rate for the MC vs TD prediction comparison.",
    )
    parser.add_argument(
        "--epsilon-prediction",
        type=float,
        default=0.0,
        help="Epsilon for the fixed policy used in MC/TD prediction.",
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
        "--alpha-control",
        type=float,
        default=0.1,
        help="Learning rate for Q-learning and SARSA.",
    )
    parser.add_argument(
        "--epsilon-control",
        type=float,
        default=0.1,
        help="Exploration epsilon for Q-learning and SARSA.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor.",
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
        help="Evaluation episodes for the final greedy policies.",
    )
    return parser


def save_figure(fig: plt.Figure, path: Path, show: bool = False) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path.name}")
    if show:
        plt.show()
    plt.close(fig)


def parse_snapshot_list(text: str) -> tuple[int, ...]:
    chunks = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
    if not chunks:
        raise SystemExit("--control-snapshots must contain at least one integer.")
    try:
        values = tuple(int(chunk) for chunk in chunks)
    except ValueError as exc:
        raise SystemExit("--control-snapshots must be a comma-separated list of integers.") from exc
    if any(v <= 0 for v in values):
        raise SystemExit("--control-snapshots values must all be positive.")
    return values


def _signature(fn: Any):
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def call_supported(fn: Any, /, *args, **kwargs):
    """Call a function while filtering unsupported keyword arguments."""
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


def state_id(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (int, str, float, bool)):
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if hasattr(obj, "id"):
        try:
            return int(getattr(obj, "id"))
        except Exception:
            return getattr(obj, "id")
    return str(obj)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return state_id(obj)


def env_metadata(env: Any) -> dict[str, Any]:
    return {
        "grid_height": getattr(env, "grid_height", None),
        "grid_width": getattr(env, "grid_width", None),
        "num_states": len(getattr(env, "states", [])) if hasattr(env, "states") else None,
        "num_actions": len(getattr(env, "actions", [])) if hasattr(env, "actions") else None,
        "start_state": state_id(getattr(env, "start_state", None)),
        "goal_state": state_id(getattr(env, "goal_state", None)),
        "quicksand_ids": sorted(int(x) for x in getattr(env, "quicksand_ids", set())),
        "p_intended": getattr(env, "p_intended", None),
        "p_perp": getattr(env, "p_perp", None),
    }


def apply_reward_shaping_local(env: Any, step_cost: float, hole_penalty: float, goal_reward: float) -> None:
    """Notebook-style reward shaping, used as a fallback if rlbook does not export it."""
    if hasattr(env, "rewards") and hasattr(env, "transitions"):
        try:
            env.rewards.clear()
        except Exception:
            pass
        holes = getattr(env, "quicksand_ids", set())
        for (sid, action), nexts in env.transitions.items():
            for nxt, _p in nexts.items():
                nid = state_id(nxt)
                if nid == state_id(getattr(env, "goal_state", None)):
                    reward = goal_reward
                elif nid in holes:
                    reward = hole_penalty
                else:
                    reward = step_cost
                env.rewards[(sid, action, nid)] = reward
    else:
        raise RuntimeError("Environment does not expose transitions/rewards needed for reward shaping.")


DESERT_VALUE_CMAP = LinearSegmentedColormap.from_list(
    "desert_value_cmap",
    [
        "#E69F00",
        "#E6C27A",
        "#F3F1EA",
        "#A7CBE2",
        "#0072B2",
    ],
    N=256,
)

START_COLOR = "#cfe8ff"
GOAL_COLOR = "#ffd700"
QUICKSAND_COLOR = "#e8c07d"
OPEN_CELL_COLOR = "white"


def grid_center(env: Any, sid: int) -> tuple[float, float]:
    row, col = divmod(int(sid), env.grid_width)
    return col + 0.5, env.grid_height - 1 - row + 0.5


def most_likely_next_state_id(env: Any, sid: int, action: int) -> Optional[int]:
    nexts = getattr(env, "transitions", {}).get((sid, action), {})
    if not nexts:
        return None
    best_prob = max(nexts.values())
    candidates = [int(state_id(ns)) for ns, prob in nexts.items() if prob == best_prob]
    if not candidates:
        return None
    if sid in candidates:
        return sid
    return min(candidates)


def draw_policy_arrow(ax: Any, env: Any, sid: int, action: int, color: str = "black") -> None:
    nxt = most_likely_next_state_id(env, sid, action)
    if nxt is None or int(nxt) == int(sid):
        return

    x0, y0 = grid_center(env, sid)
    x1, y1 = grid_center(env, nxt)
    dx, dy = x1 - x0, y1 - y0
    dist = max((dx * dx + dy * dy) ** 0.5, 1e-9)
    pad = 0.18
    sx = x0 + pad * dx / dist
    sy = y0 + pad * dy / dist
    ex = x1 - pad * dx / dist
    ey = y1 - pad * dy / dist

    patch = FancyArrowPatch(
        (sx, sy),
        (ex, ey),
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=1.6,
        color=color,
        zorder=4,
    )
    ax.add_patch(patch)


def visualize_grid_world_desert_manual(
    env: Any,
    *,
    values: Optional[Mapping[int, float]] = None,
    policy: Optional[Mapping[int, Optional[int]]] = None,
    ax: Any = None,
    title: Optional[str] = None,
    show_legend: bool = True,
    value_fmt: str = "{:.1f}",
):
    """Notebook-style desert visualizer with the earlier arrow-to-wall fix."""
    H, W = env.grid_height, env.grid_width
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, W), max(4, H)))

    grid_vals = None
    if values is not None:
        grid_vals = np.zeros((H, W), dtype=float)
        for sid in range(H * W):
            row, col = divmod(sid, W)
            grid_vals[row, col] = float(values.get(sid, 0.0))
        vmin, vmax = float(np.min(grid_vals)), float(np.max(grid_vals))
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = None

    quicksand = set(int(x) for x in getattr(env, "quicksand_ids", set()))
    start_sid = int(state_id(getattr(env, "start_state", None)))
    goal_sid = int(state_id(getattr(env, "goal_state", None)))

    for row in range(H):
        for col in range(W):
            sid = row * W + col
            if sid == start_sid:
                face = START_COLOR
            elif sid == goal_sid:
                face = GOAL_COLOR
            elif sid in quicksand:
                face = QUICKSAND_COLOR
            else:
                if grid_vals is not None and norm is not None:
                    face = DESERT_VALUE_CMAP(norm(grid_vals[row, col]))
                else:
                    face = OPEN_CELL_COLOR

            rect = Rectangle((col, H - 1 - row), 1, 1, facecolor=face, edgecolor="black", linewidth=1.0, zorder=1)
            ax.add_patch(rect)

            if grid_vals is not None and sid not in quicksand:
                ax.text(
                    col + 0.5,
                    H - 1 - row + 0.5,
                    value_fmt.format(grid_vals[row, col]),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    zorder=3,
                )

            if policy is not None and sid in policy and policy[sid] is not None:
                draw_policy_arrow(ax, env, sid, int(policy[sid]), color="black")

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.grid(True, alpha=0.35)
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if title is None:
        if hasattr(env, "p_intended") and hasattr(env, "p_perp"):
            title = f"Windy Desert (move {env.p_intended:.2f}, drift ⟂ {env.p_perp:.2f})"
        else:
            title = "Windy Desert"
    ax.set_title(title, fontsize=12)

    if show_legend:
        legend_items = [
            Rectangle((0, 0), 1, 1, facecolor=START_COLOR, edgecolor="black", label="Start"),
            Rectangle((0, 0), 1, 1, facecolor=GOAL_COLOR, edgecolor="black", label="Oasis"),
            Rectangle((0, 0), 1, 1, facecolor=QUICKSAND_COLOR, edgecolor="black", label="Quicksand"),
            Rectangle((0, 0), 1, 1, facecolor=OPEN_CELL_COLOR, edgecolor="black", label="Dry Sand"),
        ]
        ax.legend(handles=legend_items, loc="upper right", frameon=True)

    return ax


def snapshot_items(snaps: Any, requested: tuple[int, ...]) -> list[tuple[int, Any]]:
    """Normalize control snapshots into ordered (episode, snapshot) pairs."""
    if isinstance(snaps, Mapping):
        items: list[tuple[int, Any]] = []
        for episode in requested:
            if episode in snaps:
                items.append((episode, snaps[episode]))
            elif str(episode) in snaps:
                items.append((episode, snaps[str(episode)]))
        if items:
            return items

        fallback: list[tuple[int, Any]] = []
        for key, value in snaps.items():
            try:
                fallback.append((int(key), value))
            except Exception:
                continue
        return sorted(fallback, key=lambda pair: pair[0])

    if isinstance(snaps, (list, tuple)):
        if len(snaps) == len(requested):
            return list(zip(requested, list(snaps)))
        pairs: list[tuple[int, Any]] = []
        for item in snaps:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    pairs.append((int(item[0]), item[1]))
                except Exception:
                    continue
        if pairs:
            return sorted(pairs, key=lambda pair: pair[0])
        return list(zip(requested[: len(snaps)], list(snaps)[: len(requested)]))

    return [(requested[-1], snaps)]


def snapshot_to_policy(snapshot: Any, env: Any, extract_greedy_policy_fn: Any) -> dict[int, Optional[int]]:
    """Turn a snapshot into a policy, handling both stored policies and stored Q-tables."""
    if isinstance(snapshot, Mapping):
        keys = list(snapshot.keys())[:8]
        values = list(snapshot.values())[:8]
        looks_like_policy_keys = all(not isinstance(key, tuple) for key in keys)
        looks_like_policy_vals = all(
            value is None or isinstance(value, (int, np.integer))
            for value in values
        )
        if looks_like_policy_keys and looks_like_policy_vals:
            return {int(k): (None if v is None else int(v)) for k, v in snapshot.items()}

    return extract_greedy_policy_fn(snapshot, env)


def plot_policy_evolution_manual(
    env: Any,
    snaps: Any,
    snapshot_episodes: tuple[int, ...],
    *,
    extract_greedy_policy_fn: Any,
    title: str,
) -> plt.Figure:
    items = snapshot_items(snaps, snapshot_episodes)
    if not items:
        raise RuntimeError("No policy snapshots were available to plot.")

    fig, axes = plt.subplots(1, len(items), figsize=(4 * len(items), 4))
    if len(items) == 1:
        axes = [axes]

    for ax, (episode, snapshot) in zip(axes, items):
        policy = snapshot_to_policy(snapshot, env, extract_greedy_policy_fn)
        visualize_grid_world_desert_manual(
            env,
            policy=policy,
            ax=ax,
            title=f"iter {episode}",
            show_legend=False,
        )

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def compute_mode_path(env: Any, policy: Mapping[int, Optional[int]], max_steps: Optional[int] = None) -> list[int]:
    """Notebook-style representative path: follow the most likely next state under the policy."""
    if max_steps is None:
        max_steps = env.grid_height * env.grid_width * 2

    current = getattr(env, "start_state")
    path = [int(state_id(current))]
    seen = {int(state_id(current))}

    for _ in range(max_steps):
        action = policy.get(int(state_id(current)), None)
        if action is None:
            break

        nexts = getattr(env, "transitions", {}).get((int(state_id(current)), int(action)), {})
        if not nexts:
            break

        best_prob = max(nexts.values())
        candidates = [ns for ns, prob in nexts.items() if prob == best_prob]
        nxt = random.choice(candidates)

        nxt_id = int(state_id(nxt))
        path.append(nxt_id)

        if getattr(nxt, "terminal", False) or nxt_id in seen:
            break

        seen.add(nxt_id)
        current = nxt

    return path


def visualize_policy_path_manual(
    env: Any,
    policy: Mapping[int, Optional[int]],
    *,
    color: str = "C0",
    ax: Any = None,
    title: Optional[str] = None,
    lw: float = 3.0,
):
    H, W = env.grid_height, env.grid_width
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, W), max(4, H)))

    quicksand = set(int(x) for x in getattr(env, "quicksand_ids", set()))
    start_sid = int(state_id(getattr(env, "start_state", None)))
    goal_sid = int(state_id(getattr(env, "goal_state", None)))

    for row in range(H):
        for col in range(W):
            sid = row * W + col
            if sid == start_sid:
                face = START_COLOR
            elif sid == goal_sid:
                face = GOAL_COLOR
            elif sid in quicksand:
                face = QUICKSAND_COLOR
            else:
                face = OPEN_CELL_COLOR
            ax.add_patch(Rectangle((col, H - 1 - row), 1, 1, facecolor=face, edgecolor="black"))

    path = compute_mode_path(env, policy)

    for i in range(len(path) - 1):
        x0, y0 = grid_center(env, path[i])
        x1, y1 = grid_center(env, path[i + 1])
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=lw, color=color, shrinkA=6, shrinkB=6),
        )

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title or "Mode path")
    return ax, path


def evaluate_policy_supported(fn: Any, env: Any, policy: Any, episodes: int, max_steps: int):
    sig = _signature(fn)
    kwargs: dict[str, Any] = {}
    if sig is None or "episodes" in sig.parameters:
        kwargs["episodes"] = episodes
    elif "num_episodes" in sig.parameters:
        kwargs["num_episodes"] = episodes

    if sig is None or "max_steps" in sig.parameters:
        kwargs["max_steps"] = max_steps

    return fn(env, policy, **kwargs)


def train_control(
    fn: Any,
    *,
    env: Any,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    snapshots: tuple[int, ...],
    max_steps: int,
):
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

    result = fn(env, **kwargs)
    if not (isinstance(result, (list, tuple)) and len(result) == 2):
        raise RuntimeError(
            f"{getattr(fn, '__name__', 'control algorithm')} did not return the expected (Q, snapshots) tuple."
        )
    return result[0], result[1]


def main() -> int:
    args = build_parser().parse_args()
    control_snapshots = parse_snapshot_list(args.control_snapshots)
    modules = _import_rlbook()

    np.random.seed(args.seed)
    random.seed(args.seed)

    create_windy_desert_env = modules["create_windy_desert_env"]
    pkg_apply_reward_shaping = modules["apply_reward_shaping"]
    value_iteration = modules["value_iteration"]
    extract_policy = modules["extract_policy"]
    make_epsilon_greedy_policy = modules["make_epsilon_greedy_policy"]
    q_learning = modules["q_learning"]
    sarsa = modules["sarsa"]
    extract_greedy_policy = modules["extract_greedy_policy"]
    evaluate_policy = modules["evaluate_policy"]
    rmse = modules["rmse"]
    mc_td_traces = modules["mc_td_traces"]

    save_dir = args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "seed": args.seed,
        "save_dir": str(save_dir),
        "figures": [],
        "parts": {},
    }

    print("\n=============================================")
    print("Chapter 3 Case Study: Classical RL Algorithms")
    print("=============================================")
    print("MC vs TD(0) Prediction and Q-Learning vs SARSA on the Windy Desert")
    print(f"Seed: {args.seed}")
    print(f"Output directory: {save_dir}")

    # ------------------------------------------------------------------
    # Part 1/2: MC vs TD(0) Prediction
    # ------------------------------------------------------------------
    print("\n--- Part 1/2 - MC vs TD(0) Prediction ---")
    env_pred = call_with_fallbacks(
        create_windy_desert_env,
        [
            ((), {"gamma": args.gamma}),
            ((), {}),
        ],
    )

    # Re-apply the notebook's default rewards so the CLI mirrors the colab.
    apply_reward_shaping_local(env_pred, step_cost=-0.01, hole_penalty=-1.0, goal_reward=1.0)

    print(
        f"Prediction env: {getattr(env_pred, 'grid_height', '?')}x{getattr(env_pred, 'grid_width', '?')}, "
        f"states={len(getattr(env_pred, 'states', []))}"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    visualize_grid_world_desert_manual(env_pred, ax=ax, title="Windy Desert Layout", show_legend=True)
    path = save_dir / "01_windy_desert_layout.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    V_star = call_supported(value_iteration, env_pred)
    greedy_pi = call_supported(extract_policy, env_pred, V_star)
    eps_policy = call_supported(
        make_epsilon_greedy_policy,
        greedy_pi,
        getattr(env_pred, "actions", None),
        epsilon=args.epsilon_prediction,
    )

    print(
        f"Prediction config: epsilon={args.epsilon_prediction}, episodes={args.prediction_episodes}, "
        f"snapshot_every={args.prediction_snapshot_every}, alpha_td={args.alpha_td}, gamma={args.gamma}"
    )

    mc_snapshots, td_snapshots = call_supported(
        mc_td_traces,
        eps_policy,
        env_pred,
        total_episodes=args.prediction_episodes,
        snapshot_every=args.prediction_snapshot_every,
        alpha_td=args.alpha_td,
        gamma=args.gamma,
    )

    V_mc_final = mc_snapshots[-1]
    V_td_final = td_snapshots[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    visualize_grid_world_desert_manual(
        env_pred,
        ax=axes[0],
        title="Windy Desert Layout",
        show_legend=True,
    )
    visualize_grid_world_desert_manual(
        env_pred,
        values=V_mc_final,
        ax=axes[1],
        title=f"MC Value (ε={args.epsilon_prediction})",
        show_legend=False,
    )
    visualize_grid_world_desert_manual(
        env_pred,
        values=V_td_final,
        ax=axes[2],
        title=f"TD(0) Value (ε={args.epsilon_prediction})",
        show_legend=False,
    )
    path = save_dir / "02_mc_vs_td_values.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    episodes_axis = list(
        range(
            args.prediction_snapshot_every,
            args.prediction_episodes + 1,
            args.prediction_snapshot_every,
        )
    )
    mc_err = [float(rmse(V, V_star)) for V in mc_snapshots]
    td_err = [float(rmse(V, V_star)) for V in td_snapshots]
    if len(episodes_axis) != len(mc_err):
        episodes_axis = list(range(1, len(mc_err) + 1))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(episodes_axis, mc_err, marker="o", label="MC")
    ax.plot(episodes_axis, td_err, marker="s", label="TD(0)")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("RMSE to V*")
    ax.set_title(f"MC vs TD(0) Convergence (ε={args.epsilon_prediction})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = save_dir / "03_mc_vs_td_rmse.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    mc_final_rmse = float(rmse(V_mc_final, V_star))
    td_final_rmse = float(rmse(V_td_final, V_star))
    print(f"Final MC RMSE:  {mc_final_rmse:.4f}")
    print(f"Final TD RMSE:  {td_final_rmse:.4f}")

    summary["parts"]["prediction"] = {
        "environment": env_metadata(env_pred),
        "epsilon": args.epsilon_prediction,
        "gamma": args.gamma,
        "alpha_td": args.alpha_td,
        "episodes": args.prediction_episodes,
        "snapshot_every": args.prediction_snapshot_every,
        "mc_snapshot_count": len(mc_snapshots),
        "td_snapshot_count": len(td_snapshots),
        "mc_final_rmse": mc_final_rmse,
        "td_final_rmse": td_final_rmse,
    }

    # ------------------------------------------------------------------
    # Part 2/2: Q-Learning vs SARSA control
    # ------------------------------------------------------------------
    print("\n--- Part 2/2 - Q-Learning vs SARSA Control ---")
    env_ctrl = call_with_fallbacks(
        create_windy_desert_env,
        [
            ((), {"p_intended": 0.90, "p_perp": 0.05, "gamma": args.gamma}),
            ((), {"p_intended": 0.90, "p_perp": 0.05}),
        ],
    )

    if pkg_apply_reward_shaping is not None:
        call_supported(
            pkg_apply_reward_shaping,
            env_ctrl,
            step_cost=-1.0,
            hole_penalty=-100.0,
            goal_reward=10.0,
        )
    else:
        apply_reward_shaping_local(env_ctrl, step_cost=-1.0, hole_penalty=-100.0, goal_reward=10.0)

    print(
        f"Control env: {getattr(env_ctrl, 'grid_height', '?')}x{getattr(env_ctrl, 'grid_width', '?')}, "
        f"states={len(getattr(env_ctrl, 'states', []))}, p_intended=0.90, p_perp=0.05"
    )
    print(
        f"Control config: episodes={args.control_episodes}, alpha={args.alpha_control}, "
        f"gamma={args.gamma}, epsilon={args.epsilon_control}, "
        f"snapshots={control_snapshots}, max_steps={args.max_steps}"
    )

    Q_q, snaps_q = train_control(
        q_learning,
        env=env_ctrl,
        episodes=args.control_episodes,
        alpha=args.alpha_control,
        gamma=args.gamma,
        epsilon=args.epsilon_control,
        snapshots=control_snapshots,
        max_steps=args.max_steps,
    )
    Q_s, snaps_s = train_control(
        sarsa,
        env=env_ctrl,
        episodes=args.control_episodes,
        alpha=args.alpha_control,
        gamma=args.gamma,
        epsilon=args.epsilon_control,
        snapshots=control_snapshots,
        max_steps=args.max_steps,
    )

    fig = plot_policy_evolution_manual(
        env_ctrl,
        snaps_q,
        control_snapshots,
        extract_greedy_policy_fn=extract_greedy_policy,
        title=f"Q-learning policy evolution (ε={args.epsilon_control})",
    )
    path = save_dir / "04_q_learning_policy_evolution.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    fig = plot_policy_evolution_manual(
        env_ctrl,
        snaps_s,
        control_snapshots,
        extract_greedy_policy_fn=extract_greedy_policy,
        title=f"SARSA policy evolution (ε={args.epsilon_control})",
    )
    path = save_dir / "05_sarsa_policy_evolution.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    pi_q = call_supported(extract_greedy_policy, Q_q, env_ctrl)
    pi_s = call_supported(extract_greedy_policy, Q_s, env_ctrl)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _, q_path = visualize_policy_path_manual(
        env_ctrl,
        pi_q,
        color="C0",
        ax=axes[0],
        title="Q-learning: final path",
    )
    _, s_path = visualize_policy_path_manual(
        env_ctrl,
        pi_s,
        color="C1",
        ax=axes[1],
        title="SARSA: final path",
    )
    path = save_dir / "06_q_vs_sarsa_final_paths.png"
    save_figure(fig, path, args.show)
    summary["figures"].append(path.name)

    ret_q, steps_q, success_q = evaluate_policy_supported(
        evaluate_policy,
        env_ctrl,
        pi_q,
        args.eval_episodes,
        args.max_steps,
    )
    ret_s, steps_s, success_s = evaluate_policy_supported(
        evaluate_policy,
        env_ctrl,
        pi_s,
        args.eval_episodes,
        args.max_steps,
    )

    print(
        f"Q-Learning ({args.control_episodes:,} eps): "
        f"return={float(ret_q):.2f}, steps={float(steps_q):.1f}, success={float(success_q):.0f}%"
    )
    print(
        f"SARSA      ({args.control_episodes:,} eps): "
        f"return={float(ret_s):.2f}, steps={float(steps_s):.1f}, success={float(success_s):.0f}%"
    )

    summary["parts"]["control"] = {
        "environment": env_metadata(env_ctrl),
        "reward_shaping": {
            "step_cost": -1.0,
            "hole_penalty": -100.0,
            "goal_reward": 10.0,
        },
        "episodes": args.control_episodes,
        "alpha": args.alpha_control,
        "gamma": args.gamma,
        "epsilon": args.epsilon_control,
        "max_steps": args.max_steps,
        "snapshot_episodes": list(control_snapshots),
        "q_learning_path": q_path,
        "sarsa_path": s_path,
        "q_learning_path_hits_goal": bool(q_path and q_path[-1] == state_id(getattr(env_ctrl, "goal_state", None))),
        "sarsa_path_hits_goal": bool(s_path and s_path[-1] == state_id(getattr(env_ctrl, "goal_state", None))),
        "evaluation_episodes": args.eval_episodes,
        "evaluation": {
            "q_learning": {
                "mean_return": float(ret_q),
                "mean_steps": float(steps_q),
                "success_rate": float(success_q),
            },
            "sarsa": {
                "mean_return": float(ret_s),
                "mean_steps": float(steps_s),
                "success_rate": float(success_s),
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
