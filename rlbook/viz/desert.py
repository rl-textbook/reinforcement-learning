"""Visualization helpers for the Windy Desert case study."""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from analysis import ValueTrajectory, VisitationCounter

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


def visualize_grid_world_desert(
    values=None,
    policy=None,
    env=None,
    ax=None,
    title: Optional[str] = None,
    show_legend: bool = True,
    value_fmt: str = "{:.1f}",
):
    """Render the Windy Desert with optional values and policy arrows."""
    assert env is not None, "env is required"

    H, W = env.grid_height, env.grid_width
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, W), max(4, H)))

    grid_vals = None
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

    quicksand_ids = getattr(env, "quicksand_ids", set())
    arrows = {0: (0.0, 0.35), 1: (0.35, 0.0), 2: (0.0, -0.35), 3: (-0.35, 0.0)}

    for r in range(H):
        for c in range(W):
            sid = r * W + c
            if sid == env.start_state.id:
                face = "#cfe8ff"
            elif sid == env.goal_state.id:
                face = "#ffd700"
            elif sid in quicksand_ids:
                face = "#e8c07d"
            elif grid_vals is not None:
                face = DESERT_VALUE_CMAP(norm(grid_vals[r, c]))
            else:
                face = "white"

            ax.add_patch(Rectangle((c, H - 1 - r), 1, 1, facecolor=face, edgecolor="black", lw=0.8))

            if grid_vals is not None and sid not in quicksand_ids:
                ax.text(
                    c + 0.5,
                    H - 1 - r + 0.5,
                    value_fmt.format(grid_vals[r, c]),
                    ha="center",
                    va="center",
                    fontsize=9,
                )

            if policy is not None and policy.get(sid) is not None:
                dx, dy = arrows[policy[sid]]
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

    if title is None:
        if hasattr(env, "p_intended") and hasattr(env, "p_perp"):
            title = (
                f"Windy Desert (move {env.p_intended:.2f}, "
                f"drift ⟂ {env.p_perp:.2f} each)"
            )
        else:
            title = "Windy Desert"
    ax.set_title(title, fontsize=12)

    if show_legend:
        patches = [
            Rectangle((0, 0), 1, 1, fc="#cfe8ff", ec="k", label="Start"),
            Rectangle((0, 0), 1, 1, fc="#ffd700", ec="k", label="Oasis"),
            Rectangle((0, 0), 1, 1, fc="#e8c07d", ec="k", label="Quicksand"),
            Rectangle((0, 0), 1, 1, fc="white", ec="k", label="Dry sand"),
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8, frameon=True)

    return ax


def plot_policy_evolution(env, snaps: Dict[int, Dict[int, Optional[int]]], title: str):
    """Plot policy snapshots across training."""
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
    """Pick the most likely successor that changes the state, if possible."""
    nexts = env.transitions.get((state_id, action), {})
    if not nexts:
        return None

    distinct = [(ns, p) for ns, p in nexts.items() if ns.id != state_id]
    pool = distinct if distinct else list(nexts.items())
    best_p = max(p for _, p in pool)
    cands = [ns for ns, p in pool if abs(p - best_p) <= 1e-12]
    return min(cands, key=lambda ns: ns.id)


def compute_mode_path(env, policy, max_steps: Optional[int] = None):
    """Trace a representative path while compressing wall-bump self-loops."""
    if max_steps is None:
        max_steps = env.grid_height * env.grid_width * 2

    s = env.start_state
    path = [s.id]
    seen = {s.id}
    for _ in range(max_steps):
        a = policy.get(s.id, None)
        if a is None:
            break
        nxt = most_likely_next_distinct_state(env, s.id, a)
        if nxt is None:
            break
        path.append(nxt.id)
        if nxt.terminal or nxt.id in seen:
            break
        seen.add(nxt.id)
        s = nxt
    return path


def visualize_policy_path(env, policy, color: str = "C0", ax=None, title: Optional[str] = None, lw: int = 3):
    """Overlay a representative policy path on the grid."""
    H, W = env.grid_height, env.grid_width
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, W), max(4, H)))

    for r in range(H):
        for c in range(W):
            sid = r * W + c
            if sid == env.start_state.id:
                face = "#cfe8ff"
            elif sid == env.goal_state.id:
                face = "#ffd700"
            elif sid in getattr(env, "quicksand_ids", set()):
                face = "#e8c07d"
            else:
                face = "white"
            ax.add_patch(Rectangle((c, H - 1 - r), 1, 1, facecolor=face, edgecolor="black"))

    path = compute_mode_path(env, policy)

    def ctr(sid: int):
        r, c = divmod(sid, W)
        return c + 0.5, H - 1 - r + 0.5

    for i in range(len(path) - 1):
        x0, y0 = ctr(path[i])
        x1, y1 = ctr(path[i + 1])
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
    ax.set_title(title or "Representative path")
    return ax, path


def plot_visitation_mc_vs_td(
    mc_visit: VisitationCounter,
    td_visit: VisitationCounter,
    env,
    mode: str = "freq",
    cmap_name: str = "magma",
    equal_scale: bool = True,
):
    """Compare MC and TD(0) visitation heatmaps."""
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
    checkpoints: Optional[List[int]] = None,
    cmap_name: str = "Greens",
):
    """Show static value-function snapshots at selected episodes."""
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

    def closest_pair(d, k):
        ks = [x for x in d.keys() if x <= k]
        actual = max(ks) if ks else min(d.keys())
        return actual, d[actual]

    cols = len(checkpoints)
    fig, axs = plt.subplots(2, cols, figsize=(3.2 * cols, 6), constrained_layout=True)
    if cols == 1:
        axs = np.array(axs).reshape(2, 1)
    for j, ep in enumerate(checkpoints):
        mc_ep, Gmc = closest_pair(mc_map, ep)
        td_ep, Gtd = closest_pair(td_map, ep)
        im1 = axs[0, j].imshow(Gmc, vmin=vmin, vmax=vmax, cmap=cmap_name)
        axs[0, j].set_title(f"MC (ep {mc_ep})")
        axs[0, j].set_xticks([])
        axs[0, j].set_yticks([])
        axs[1, j].imshow(Gtd, vmin=vmin, vmax=vmax, cmap=cmap_name)
        axs[1, j].set_title(f"TD(0) (ep {td_ep})")
        axs[1, j].set_xticks([])
        axs[1, j].set_yticks([])

    fig.colorbar(im1, ax=axs.ravel().tolist(), shrink=0.85, label="State value")
    plt.show()


def animate_value_evolution(
    rec_mc: ValueTrajectory,
    rec_td: ValueTrajectory,
    interval_ms: int = 50,
    cmap_name: str = "Greens",
    save_path: Optional[str] = None,
):
    """Animate MC and TD(0) value evolution side by side."""
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

    def update(t: int):
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
