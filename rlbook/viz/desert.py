"""Windy Desert visualization functions (Chapter 3)."""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Arrow
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from rlbook.core.analysis import VisitationCounter, ValueTrajectory


def visualize_grid_world_desert(values=None, policy=None, env=None,
                                ax=None, title=None, show_legend=True,
                                value_fmt="{:.1f}"):
    """Windy Desert grid visualizer.

    Shows Start (light blue), Oasis (gold), Quicksand (sand), Safe sand (white).
    Supports value heatmaps and policy arrows.

    Args:
        values: Dict {state_id -> V(s)} or None.
        policy: Dict {state_id -> action} or None.
        env: Environment (required).
        ax: Matplotlib axis (created if None).
        title: Plot title override.
        show_legend: Whether to draw legend blocks.
        value_fmt: Format string for value text.

    Returns:
        Matplotlib axis.
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
        for i in range(H * W):
            r, c = divmod(i, W)
            grid_vals[r, c] = values.get(i, 0.0)
        vmin, vmax = grid_vals.min(), grid_vals.max()
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = LinearSegmentedColormap.from_list(
            'value_cmap', [(0.8, 0.0, 0.0), (1, 1, 1), (0.0, 0.6, 0.0)], N=100)

    quicksand_ids = getattr(env, 'quicksand_ids', set())
    arrows = {0: (0, 0.35), 1: (0.35, 0), 2: (0, -0.35), 3: (-0.35, 0)}

    for r in range(H):
        for c in range(W):
            sid = r * W + c
            if sid == env.start_state.id:
                face = '#cfe8ff'
            elif sid == env.goal_state.id:
                face = '#ffd700'
            elif sid in quicksand_ids:
                face = '#f5deb3'
            elif grid_vals is not None:
                face = cmap(norm(grid_vals[r, c]))
            else:
                face = 'white'

            ax.add_patch(Rectangle((c, H - 1 - r), 1, 1,
                                   facecolor=face, edgecolor='black', lw=0.8))

            if grid_vals is not None and sid not in quicksand_ids:
                ax.text(c + 0.5, H - 1 - r + 0.75,
                        value_fmt.format(grid_vals[r, c]),
                        ha='center', va='center', fontsize=8)

            if policy is not None and policy.get(sid) is not None:
                dx, dy = arrows.get(policy[sid], (0, 0))
                if dx != 0 or dy != 0:
                    ax.annotate("", xy=(c + 0.5 + dx, H - 1 - r + 0.5 + dy),
                                xytext=(c + 0.5, H - 1 - r + 0.5),
                                arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    ax.grid(True)

    if show_legend:
        patches = [
            Rectangle((0, 0), 1, 1, fc='#cfe8ff', ec='k', label='Start'),
            Rectangle((0, 0), 1, 1, fc='#ffd700', ec='k', label='Oasis (Goal)'),
            Rectangle((0, 0), 1, 1, fc='#f5deb3', ec='k', label='Quicksand'),
        ]
        ax.legend(handles=patches, loc='upper left', bbox_to_anchor=(1.01, 1),
                  fontsize=8, frameon=True)

    if title:
        ax.set_title(title, fontsize=12)
    return ax


# ---------------------------------------------------------------------------
# Policy evolution
# ---------------------------------------------------------------------------
def plot_policy_evolution(env, snaps: Dict[int, Dict[int, Optional[int]]], title):
    """Plot a row of policy snapshots at different episode counts."""
    ks = sorted(snaps.keys())
    n = len(ks)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axs = [axs]
    for ax, k in zip(axs, ks):
        visualize_grid_world_desert(policy=snaps[k], env=env, ax=ax,
                                    title=f"iter {k}", show_legend=False)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()


# ---------------------------------------------------------------------------
# Mode-probability path overlay
# ---------------------------------------------------------------------------
def compute_mode_path(env, policy, max_steps=None):
    """Trace the most-likely path under a deterministic policy."""
    if max_steps is None:
        max_steps = env.grid_height * env.grid_width * 2
    s = env.start_state
    path = [s.id]
    seen = {s.id}
    for _ in range(max_steps):
        a = policy.get(s.id, None)
        if a is None:
            break
        nexts = env.transitions.get((s.id, a), {})
        if not nexts:
            break
        m = max(nexts.values())
        cands = [ns for ns, p in nexts.items() if p == m]
        nxt = random.choice(cands)
        path.append(nxt.id)
        if nxt.terminal or nxt.id in seen:
            break
        seen.add(nxt.id)
        s = nxt
    return path


def visualize_policy_path(env, policy, color='C0', ax=None, title=None, lw=3):
    """Overlay the mode-probability path on the desert grid."""
    H, W = env.grid_height, env.grid_width
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, W), max(4, H)))

    quicksand_ids = getattr(env, 'quicksand_ids', set())
    for r in range(H):
        for c in range(W):
            sid = r * W + c
            if sid == env.start_state.id:
                face = '#cfe8ff'
            elif sid == env.goal_state.id:
                face = '#ffd700'
            elif sid in quicksand_ids:
                face = '#ffb3b3'
            else:
                face = 'white'
            ax.add_patch(Rectangle((c, H - 1 - r), 1, 1,
                                   facecolor=face, edgecolor='black'))

    path = compute_mode_path(env, policy)

    def ctr(sid):
        r, c = divmod(sid, W)
        return c + 0.5, H - 1 - r + 0.5

    for i in range(len(path) - 1):
        x0, y0 = ctr(path[i])
        x1, y1 = ctr(path[i + 1])
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", lw=lw, color=color,
                                     shrinkA=6, shrinkB=6))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks(range(W + 1))
    ax.set_yticks(range(H + 1))
    ax.grid(True)
    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title or "Mode path")
    return ax, path


# ---------------------------------------------------------------------------
# MC / TD visitation and value evolution plots
# ---------------------------------------------------------------------------
def plot_visitation_mc_vs_td(mc_visit: VisitationCounter,
                             td_visit: VisitationCounter,
                             env, mode="freq", cmap_name="magma",
                             equal_scale=True):
    """Side-by-side MC vs TD(0) visitation heatmaps."""
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

    label_map = {"raw": "raw visits", "freq": "visits / episode",
                 "prob": "visit probability", "log": "log1p(visits)"}
    fig.colorbar(im2, ax=axs.ravel().tolist(), shrink=0.9, label=label_map.get(mode, mode))
    plt.show()


def plot_value_snapshots(rec_mc: ValueTrajectory, rec_td: ValueTrajectory,
                         checkpoints: List[int] = None, cmap_name="Greens"):
    """Static value-function snapshots at given episodes for MC and TD."""
    if checkpoints is None:
        its = [i for i, _ in rec_mc.snaps]
        if not its:
            return
        k = min(6, len(its))
        idxs = np.linspace(0, len(its) - 1, k, dtype=int)
        checkpoints = [its[i] for i in idxs]

    mc_map = {i: G for i, G in rec_mc.snaps}
    td_map = {i: G for i, G in rec_td.snaps}

    all_vals = np.stack([*mc_map.values(), *td_map.values()], axis=0)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    def closest(d, k):
        ks = [x for x in d.keys() if x <= k]
        return d[max(ks)] if ks else next(iter(d.values()))

    cols = len(checkpoints)
    fig, axs = plt.subplots(2, cols, figsize=(3.2 * cols, 6), constrained_layout=True)
    for j, ep in enumerate(checkpoints):
        Gmc = closest(mc_map, ep)
        Gtd = closest(td_map, ep)
        im1 = axs[0, j].imshow(Gmc, vmin=vmin, vmax=vmax, cmap=cmap_name)
        axs[0, j].set_title(f"MC (ep {ep})")
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
        axs[1, j].imshow(Gtd, vmin=vmin, vmax=vmax, cmap=cmap_name)
        axs[1, j].set_title(f"TD(0) (ep {ep})")
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])

    fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.85, label="State value")
    plt.show()


def animate_value_evolution(rec_mc: ValueTrajectory, rec_td: ValueTrajectory,
                            interval_ms=50, cmap_name="Greens", save_path=None):
    """Animate MC vs TD(0) value function evolution side by side."""
    from matplotlib.animation import FuncAnimation

    T = min(len(rec_mc.snaps), len(rec_td.snaps))
    mc_frames = [G for _, G in rec_mc.snaps[:T]]
    td_frames = [G for _, G in rec_td.snaps[:T]]
    mc_its = [i for i, _ in rec_mc.snaps[:T]]
    td_its = [i for i, _ in rec_td.snaps[:T]]

    all_vals = np.stack(mc_frames + td_frames, axis=0)
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    if vmin == vmax:
        vmin, vmax = vmin - 1.0, vmax + 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    im1 = ax1.imshow(mc_frames[0], vmin=vmin, vmax=vmax, cmap=cmap_name)
    im2 = ax2.imshow(td_frames[0], vmin=vmin, vmax=vmax, cmap=cmap_name)
    ax1.set_title(f"MC (ep {mc_its[0]})")
    ax2.set_title(f"TD(0) (ep {td_its[0]})")
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im2, ax=[ax1, ax2], shrink=0.9, label="State value")

    def update(t):
        im1.set_data(mc_frames[t])
        im2.set_data(td_frames[t])
        ax1.set_title(f"MC (ep {mc_its[t]})")
        ax2.set_title(f"TD(0) (ep {td_its[t]})")
        return [im1, im2]

    anim = FuncAnimation(fig, update, frames=T, interval=interval_ms, blit=False)
    plt.close(fig)
    if save_path:
        anim.save(save_path, dpi=150)
    return anim
