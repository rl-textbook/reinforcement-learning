"""
Windy Desert visualization helpers for Chapter 3.
"""

from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from rlbook.core.analysis import ValueTrajectory, VisitationCounter


def visualize_grid_world_desert(
    values=None,
    policy=None,
    env=None,
    ax=None,
    title=None,
    show_legend: bool = True,
    value_fmt: str = "{:.1f}",
):
    """
    Grid visualizer for the Windy Desert environment.
    """
    assert env is not None, "env is required"

    H, W = env.grid_height, env.grid_width
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, W), max(4, H)))

    grid_vals = None
    cmap = None
    norm = None

    if values is not None:
        grid_vals = np.zeros((H, W), dtype=float)
        for state_id in range(H * W):
            r, c = divmod(state_id, W)
            grid_vals[r, c] = values.get(state_id, 0.0)

        vmin, vmax = grid_vals.min(), grid_vals.max()
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = LinearSegmentedColormap.from_list(
            "value_cmap",
            [(0.8, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.6, 0.0)],
            N=100,
        )

    quicksand_ids = getattr(env, "quicksand_ids", set())
    arrows = {0: (0, 0.35), 1: (0.35, 0), 2: (0, -0.35), 3: (-0.35, 0)}

    for r in range(H):
        for c in range(W):
            state_id = r * W + c
            if state_id == env.start_state.id:
                face = "#cfe8ff"
            elif state_id == env.goal_state.id:
                face = "#ffd700"
            elif state_id in quicksand_ids:
                face = "#f5deb3"
            elif grid_vals is not None:
                face = cmap(norm(grid_vals[r, c]))
            else:
                face = "white"

            ax.add_patch(
                Rectangle(
                    (c, H - 1 - r),
                    1,
                    1,
                    facecolor=face,
                    edgecolor="black",
                    lw=0.8,
                )
            )

            if grid_vals is not None and state_id not in quicksand_ids:
                ax.text(
                    c + 0.5,
                    H - 1 - r + 0.75,
                    value_fmt.format(grid_vals[r, c]),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

            if policy is not None and policy.get(state_id) is not None:
                dx, dy = arrows.get(policy[state_id], (0, 0))
                if dx != 0 or dy != 0:
                    ax.annotate(
                        "",
                        xy=(c + 0.5 + dx, H - 1 - r + 0.5 + dy),
                        xytext=(c + 0.5, H - 1 - r + 0.5),
                        arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
                    )

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.grid(True)

    if show_legend:
        legend_patches = [
            Rectangle((0, 0), 1, 1, fc="#cfe8ff", ec="k", label="Start"),
            Rectangle((0, 0), 1, 1, fc="#ffd700", ec="k", label="Oasis (Goal)"),
            Rectangle((0, 0), 1, 1, fc="#f5deb3", ec="k", label="Quicksand"),
        ]
        ax.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            fontsize=8,
            frameon=True,
        )

    if title:
        ax.set_title(title, fontsize=12)

    return ax


def plot_policy_evolution(env, snaps: Dict[int, Dict[int, Optional[int]]], title: str):
    """
    Plot policy snapshots at selected episode counts.
    """
    ks = sorted(snaps.keys())
    n = len(ks)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axs = [axs]

    for ax, k in zip(axs, ks):
        visualize_grid_world_desert(
            policy=snaps[k],
            env=env,
            ax=ax,
            title=f"iter {k}",
            show_legend=False,
        )

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


def most_likely_next_distinct_state(env, state_id: int, action: int):
    """
    Pick the most likely successor that changes state.

    This is for visualization only: it compresses wall-bump self-loops in the
    representative policy path, without changing the learned policy or MDP.
    """
    next_states = env.transitions.get((state_id, action), {})
    if not next_states:
        return None

    distinct = [(next_state, prob) for next_state, prob in next_states.items() if next_state.id != state_id]
    pool = distinct if distinct else list(next_states.items())

    best_prob = max(prob for _, prob in pool)
    candidates = [next_state for next_state, prob in pool if abs(prob - best_prob) <= 1e-12]
    return min(candidates, key=lambda next_state: next_state.id)


def compute_mode_path(env, policy, max_steps=None):
    """
    Trace the representative path induced by a deterministic policy.
    """
    if max_steps is None:
        max_steps = env.grid_height * env.grid_width * 2

    state = env.start_state
    path = [state.id]
    seen = {state.id}

    for _ in range(max_steps):
        action = policy.get(state.id, None)
        if action is None:
            break

        next_state = most_likely_next_distinct_state(env, state.id, action)
        if next_state is None:
            break

        path.append(next_state.id)

        if next_state.terminal or next_state.id in seen:
            break

        seen.add(next_state.id)
        state = next_state

    return path


def _draw_policy_path_base(ax, env):
    H, W = env.grid_height, env.grid_width
    quicksand_ids = getattr(env, "quicksand_ids", set())

    for r in range(H):
        for c in range(W):
            state_id = r * W + c
            if state_id == env.start_state.id:
                face = "#cfe8ff"
            elif state_id == env.goal_state.id:
                face = "#ffd700"
            elif state_id in quicksand_ids:
                face = "#e8c07d"
            else:
                face = "white"

            ax.add_patch(
                Rectangle((c, H - 1 - r), 1, 1, facecolor=face, edgecolor="black")
            )


def visualize_policy_path(env, policy, color="C0", ax=None, title=None, lw: int = 3):
    """
    Overlay a representative policy path on the Windy Desert grid.

    If the axis already contains the desert grid, the base map is not redrawn,
    which keeps multi-path overlays visible.
    """
    H, W = env.grid_height, env.grid_width
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, W), max(4, H)))

    if len(ax.patches) == 0:
        _draw_policy_path_base(ax, env)

    path = compute_mode_path(env, policy)

    def ctr(state_id: int):
        r, c = divmod(state_id, W)
        return c + 0.5, H - 1 - r + 0.5

    for i in range(len(path) - 1):
        x0, y0 = ctr(path[i])
        x1, y1 = ctr(path[i + 1])
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                lw=lw,
                color=color,
                shrinkA=6,
                shrinkB=6,
            ),
        )

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.grid(True)
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    if title:
        ax.set_title(title)

    return ax, path


def plot_visitation_mc_vs_td(
    mc_visit: VisitationCounter,
    td_visit: VisitationCounter,
    env,
    mode: str = "freq",
    cmap_name: str = "magma",
    equal_scale: bool = True,
):
    """
    Side-by-side visitation heatmaps for Monte Carlo and TD(0).
    """
    mc_grid = mc_visit.as_grid(env, mode=mode)
    td_grid = td_visit.as_grid(env, mode=mode)

    vmin = min(mc_grid.min(), td_grid.min()) if equal_scale else None
    vmax = max(mc_grid.max(), td_grid.max()) if equal_scale else None

    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axs[0].imshow(mc_grid, cmap=cmap_name, vmin=vmin, vmax=vmax)
    axs[0].set_title(f"MC visitation ({mode})")
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    im2 = axs[1].imshow(td_grid, cmap=cmap_name, vmin=vmin, vmax=vmax)
    axs[1].set_title(f"TD(0) visitation ({mode})")
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    label_map = {
        "raw": "raw visits",
        "freq": "visits / episode",
        "prob": "visit probability",
        "log": "log1p(visits)",
    }
    fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.9, label=label_map.get(mode, mode))
    plt.show()


def plot_value_snapshots(
    rec_mc: ValueTrajectory,
    rec_td: ValueTrajectory,
    checkpoints: List[int] = None,
    cmap_name: str = "Greens",
):
    """
    Plot static MC and TD(0) value snapshots at selected episode counts.
    """
    if checkpoints is None:
        iterations = [episode_idx for episode_idx, _ in rec_mc.snaps]
        if not iterations:
            return
        k = min(6, len(iterations))
        idxs = np.linspace(0, len(iterations) - 1, k, dtype=int)
        checkpoints = [iterations[i] for i in idxs]

    mc_map = {episode_idx: grid for episode_idx, grid in rec_mc.snaps}
    td_map = {episode_idx: grid for episode_idx, grid in rec_td.snaps}

    all_vals = np.stack([*mc_map.values(), *td_map.values()], axis=0)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    def closest(snapshot_map, episode):
        eligible = [idx for idx in snapshot_map if idx <= episode]
        return snapshot_map[max(eligible)] if eligible else next(iter(snapshot_map.values()))

    cols = len(checkpoints)
    fig, axs = plt.subplots(2, cols, figsize=(3.2 * cols, 6), constrained_layout=True)

    for j, episode in enumerate(checkpoints):
        mc_grid = closest(mc_map, episode)
        td_grid = closest(td_map, episode)

        im1 = axs[0, j].imshow(mc_grid, vmin=vmin, vmax=vmax, cmap=cmap_name)
        axs[0, j].set_title(f"MC (ep {episode})")
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])

        axs[1, j].imshow(td_grid, vmin=vmin, vmax=vmax, cmap=cmap_name)
        axs[1, j].set_title(f"TD(0) (ep {episode})")
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])

    fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.85, label="State value")
    plt.show()


def animate_value_evolution(
    rec_mc: ValueTrajectory,
    rec_td: ValueTrajectory,
    interval_ms: int = 50,
    cmap_name: str = "Greens",
    save_path=None,
):
    """
    Animate MC and TD(0) value evolution side by side.
    """
    from matplotlib.animation import FuncAnimation

    T = min(len(rec_mc.snaps), len(rec_td.snaps))
    mc_frames = [grid for _, grid in rec_mc.snaps[:T]]
    td_frames = [grid for _, grid in rec_td.snaps[:T]]
    mc_iters = [episode_idx for episode_idx, _ in rec_mc.snaps[:T]]
    td_iters = [episode_idx for episode_idx, _ in rec_td.snaps[:T]]

    all_vals = np.stack(mc_frames + td_frames, axis=0)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    im1 = ax1.imshow(mc_frames[0], vmin=vmin, vmax=vmax, cmap=cmap_name)
    im2 = ax2.imshow(td_frames[0], vmin=vmin, vmax=vmax, cmap=cmap_name)

    ax1.set_title(f"MC (ep {mc_iters[0]})")
    ax2.set_title(f"TD(0) (ep {td_iters[0]})")
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.9, label="State value")

    def update(frame_idx):
        im1.set_data(mc_frames[frame_idx])
        im2.set_data(td_frames[frame_idx])
        ax1.set_title(f"MC (ep {mc_iters[frame_idx]})")
        ax2.set_title(f"TD(0) (ep {td_iters[frame_idx]})")
        return [im1, im2]

    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    plt.close(fig)

    if save_path:
        anim.save(save_path, dpi=150)

    return anim
