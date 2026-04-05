"""Visualization helpers for the Chapter 2 MDP notebook."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Arrow, Rectangle

START_COLOR = "#cfe8ff"
HOLE_COLOR = "#e8c07d"
GOAL_COLOR = "#ffd700"
OPEN_CELL_COLOR = "white"
REWARD_NEUTRAL_COLOR = "#D9D9D9"

VALUE_NEG_COLOR = "#E69F00"
VALUE_ZERO_COLOR = "#FFFFFF"
VALUE_POS_COLOR = "#0072B2"

PATH_LINE_COLOR = "#0072B2"
PATH_START_MARKER = "#4D4D4D"
PATH_INTERMEDIATE = "#CC79A7"
PATH_END_MARKER = GOAL_COLOR

HEATMAP_CMAP = "cividis"
BAR_COLORS = [VALUE_POS_COLOR, VALUE_NEG_COLOR, "#7F7F7F"]

SEMANTIC_STATE_LABELS = {"start", "goal", "end", "hole", "holes"}


def normalize_holes(holes, grid_size: int) -> set[int]:
    if holes is None:
        return set()

    hole_ids: set[int] = set()
    for item in holes:
        if isinstance(item, tuple) and len(item) == 2:
            row, col = item
            hole_ids.add(int(row) * grid_size + int(col))
        else:
            hole_ids.add(int(item))
    return hole_ids


def is_semantic_state_label(label) -> bool:
    return str(label).strip().lower() in SEMANTIC_STATE_LABELS


def make_state_legend_handles(
    include_regular: bool = True,
    regular_facecolor: str = OPEN_CELL_COLOR,
    regular_label: str = "Empty Cell",
):
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=START_COLOR, edgecolor="black", label="Start"),
        Rectangle((0, 0), 1, 1, facecolor=GOAL_COLOR, edgecolor="black", label="Goal"),
        Rectangle((0, 0), 1, 1, facecolor=HOLE_COLOR, edgecolor="black", label="Hole"),
    ]
    if include_regular:
        handles.append(
            Rectangle((0, 0), 1, 1, facecolor=regular_facecolor, edgecolor="black", label=regular_label)
        )
    return handles


def add_state_legend(
    ax,
    include_regular: bool = True,
    regular_facecolor: str = OPEN_CELL_COLOR,
    regular_label: str = "Empty Cell",
    extra_handles=None,
    loc: str = "upper right",
    fontsize: int = 10,
    **legend_kwargs,
):
    handles = make_state_legend_handles(include_regular, regular_facecolor, regular_label)
    if extra_handles:
        handles.extend(extra_handles)
    return ax.legend(handles=handles, loc=loc, frameon=True, fontsize=fontsize, **legend_kwargs)


def visualize_grid_world(
    values: Optional[Mapping[int, float]] = None,
    policy: Optional[Mapping[int, Optional[int]]] = None,
    grid_size: int = 4,
    state_labels: Optional[Mapping[int, str]] = None,
    holes=None,
    goal_state: Optional[int] = None,
    start_state: int = 0,
    ax=None,
    title: Optional[str] = None,
    show_legend: bool = True,
):
    """Visualize a grid world with values and/or a policy."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    goal_state = grid_size * grid_size - 1 if goal_state is None else goal_state
    hole_ids = normalize_holes(holes, grid_size)

    value_grid = None
    cmap = None
    norm = None

    if values is not None:
        value_grid = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                state_idx = i * grid_size + j
                value_grid[i, j] = values.get(state_idx, 0.0)

        vmin, vmax = float(np.min(value_grid)), float(np.max(value_grid))
        if vmin == vmax:
            vmin, vmax = vmin - 1.0, vmax + 1.0

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = LinearSegmentedColormap.from_list(
            "value_cmap",
            [VALUE_NEG_COLOR, VALUE_ZERO_COLOR, VALUE_POS_COLOR],
            N=100,
        )

    for i in range(grid_size):
        for j in range(grid_size):
            state_idx = i * grid_size + j
            is_start = state_idx == start_state
            is_hole = state_idx in hole_ids
            is_goal = state_idx == goal_state

            if is_start:
                facecolor = START_COLOR
            elif is_goal:
                facecolor = GOAL_COLOR
            elif is_hole:
                facecolor = HOLE_COLOR
            elif value_grid is not None and cmap is not None and norm is not None:
                facecolor = cmap(norm(value_grid[i, j]))
            else:
                facecolor = OPEN_CELL_COLOR

            rect = Rectangle((j, grid_size - 1 - i), 1, 1, facecolor=facecolor, edgecolor="black", linewidth=1)
            ax.add_patch(rect)

            if value_grid is not None and not is_hole:
                ax.text(
                    j + 0.5,
                    grid_size - 1 - i + 0.72,
                    f"{value_grid[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                )

            if state_labels is not None and state_idx in state_labels:
                label = state_labels[state_idx]
                if not is_semantic_state_label(label):
                    label_y = 0.24 if value_grid is not None else 0.5
                    ax.text(
                        j + 0.5,
                        grid_size - 1 - i + label_y,
                        label,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )

            if (
                policy is not None
                and state_idx in policy
                and policy[state_idx] is not None
                and not is_hole
                and not is_goal
            ):
                action = policy[state_idx]
                dx, dy = 0.0, 0.0
                if action == 0:
                    dx, dy = 0.0, 0.4
                elif action == 1:
                    dx, dy = 0.4, 0.0
                elif action == 2:
                    dx, dy = 0.0, -0.4
                elif action == 3:
                    dx, dy = -0.4, 0.0

                if dx != 0 or dy != 0:
                    ax.add_patch(Arrow(j + 0.5, grid_size - 1 - i + 0.5, dx, dy, width=0.2, color="black"))

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(True, color="black", linewidth=1)
    ax.set_axisbelow(True)

    if values is not None and cmap is not None and norm is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("State Value")

    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=14)

    if show_legend:
        add_state_legend(
            ax,
            include_regular=True,
            regular_facecolor=OPEN_CELL_COLOR,
            regular_label="Empty Cell",
            fontsize=9,
        )

    return ax


def visualize_agent_path(env, path, policy=None, values=None, holes=None, title: Optional[str] = None):
    """Visualize a trajectory over the grid."""
    grid_size = int(np.sqrt(len(env.states)))
    _, ax = plt.subplots(figsize=(10, 10))
    goal_state = env.states[-1].id
    hole_ids = normalize_holes(holes, grid_size)

    visualize_grid_world(
        values=values,
        policy=policy,
        grid_size=grid_size,
        holes=hole_ids,
        goal_state=goal_state,
        start_state=0,
        ax=ax,
        show_legend=False,
    )

    for i, state_id in enumerate(path):
        row = state_id // grid_size
        col = state_id % grid_size
        y = grid_size - 1 - row
        x = col

        if i == 0:
            marker, color, size = "o", PATH_START_MARKER, 200
        elif i == len(path) - 1:
            marker, color, size = "*", "#1f77b4", 300
        else:
            marker, color, size = ".", PATH_INTERMEDIATE, 100

        if i == 0 or i == len(path) - 1 or i % 2 == 0:
            ax.scatter(x + 0.5, y + 0.5, marker=marker, color=color, s=size, zorder=10)

    for start_id, end_id in zip(path[:-1], path[1:]):
        start_row, start_col = divmod(start_id, grid_size)
        end_row, end_col = divmod(end_id, grid_size)
        start_x, start_y = start_col + 0.5, grid_size - 1 - start_row + 0.5
        end_x, end_y = end_col + 0.5, grid_size - 1 - end_row + 0.5
        ax.plot([start_x, end_x], [start_y, end_y], color=PATH_LINE_COLOR, alpha=0.6, linewidth=2, zorder=5)

    if title:
        ax.set_title(title, fontsize=16)

    semantic_legend = add_state_legend(
        ax,
        include_regular=True,
        regular_facecolor=OPEN_CELL_COLOR,
        regular_label="Empty Cell",
        loc="upper right",
        fontsize=10,
    )
    ax.add_artist(semantic_legend)

    path_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PATH_START_MARKER, markeredgecolor=PATH_START_MARKER, markersize=10, label="Path Start"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#1f77b4", markeredgecolor="#1f77b4", markersize=14, label="Path End"),
        Line2D([0], [0], color=PATH_LINE_COLOR, linewidth=2, label="Path"),
    ]
    ax.legend(handles=path_handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    plt.tight_layout()
    return ax


def plot_basic_grid_world_figure(
    grid_size: int = 5,
    holes=((1, 1), (2, 3), (3, 1)),
    start=(0, 0),
    goal=(4, 4),
    save_path=None,
    dpi: int = 300,
):
    """Plot the grid layout used in the case study."""
    holes = set(holes)
    fig, ax = plt.subplots(figsize=(10, 10))

    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) == start:
                face = START_COLOR
            elif (r, c) == goal:
                face = GOAL_COLOR
            elif (r, c) in holes:
                face = HOLE_COLOR
            else:
                face = OPEN_CELL_COLOR

            ax.add_patch(
                Rectangle((c, grid_size - 1 - r), 1, 1, facecolor=face, edgecolor="black", linewidth=2)
            )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("5×5 Grid World", fontsize=26, pad=18)
    add_state_legend(
        ax,
        include_regular=True,
        regular_facecolor=OPEN_CELL_COLOR,
        regular_label="Empty Cell",
        fontsize=14,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


def plot_reward_structure_figure(
    grid_size: int = 5,
    holes=((1, 1), (2, 3), (3, 1)),
    start=(0, 0),
    goal=(4, 4),
    step_reward: float = -1.0,
    hole_reward: float = -10.0,
    goal_reward: float = 10.0,
    save_path=None,
    dpi: int = 300,
):
    """Plot the immediate-reward structure used in the case study."""
    holes = set(holes)
    fig, ax = plt.subplots(figsize=(10, 10))

    for r in range(grid_size):
        for c in range(grid_size):
            if (r, c) == start:
                face, value = START_COLOR, step_reward
            elif (r, c) in holes:
                face, value = HOLE_COLOR, hole_reward
            elif (r, c) == goal:
                face, value = GOAL_COLOR, goal_reward
            else:
                face, value = REWARD_NEUTRAL_COLOR, step_reward

            ax.add_patch(
                Rectangle((c, grid_size - 1 - r), 1, 1, facecolor=face, edgecolor="black", linewidth=2)
            )
            ax.text(
                c + 0.5,
                grid_size - 1 - r + 0.5,
                f"{value:g}",
                ha="center",
                va="center",
                fontsize=22,
                fontweight="bold",
                color="#2B2B2B",
            )

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Reward Structure: Immediate Rewards", fontsize=26, pad=18)
    add_state_legend(
        ax,
        include_regular=True,
        regular_facecolor=REWARD_NEUTRAL_COLOR,
        regular_label="Regular State",
        fontsize=14,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


__all__ = [
    "START_COLOR",
    "HOLE_COLOR",
    "GOAL_COLOR",
    "OPEN_CELL_COLOR",
    "REWARD_NEUTRAL_COLOR",
    "VALUE_NEG_COLOR",
    "VALUE_ZERO_COLOR",
    "VALUE_POS_COLOR",
    "HEATMAP_CMAP",
    "BAR_COLORS",
    "normalize_holes",
    "is_semantic_state_label",
    "make_state_legend_handles",
    "add_state_legend",
    "visualize_grid_world",
    "visualize_agent_path",
    "plot_basic_grid_world_figure",
    "plot_reward_structure_figure",
]
