"""Grid world visualization functions (Chapter 2)."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Arrow
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Optional, List


def visualize_grid_world(values=None, policy=None, grid_size=4, state_labels=None,
                         holes=None, goal_state=None, ax=None, title=None):
    """Visualize a grid world with value function, policy, and holes.

    Args:
        values: Dict mapping state IDs to values.
        policy: Dict mapping state IDs to actions (0=up, 1=right, 2=down, 3=left).
        grid_size: Size of the grid.
        state_labels: Optional dict of labels for specific states.
        holes: Set of state IDs that are holes.
        goal_state: State ID of the goal.
        ax: Matplotlib axis (created if None).
        title: Title for the plot.

    Returns:
        Matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    cmap = None
    norm = None
    if values is not None:
        value_grid = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                state_idx = i * grid_size + j
                value_grid[i, j] = values.get(state_idx, 0)

        vmin, vmax = np.min(value_grid), np.max(value_grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colors = [(0.8, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.6, 0.0)]
        cmap = LinearSegmentedColormap.from_list('value_cmap', colors, N=100)

    for i in range(grid_size):
        for j in range(grid_size):
            state_idx = i * grid_size + j

            if holes and state_idx in holes:
                facecolor = 'red'
            elif goal_state and state_idx == goal_state:
                facecolor = 'gold'
            elif values is not None:
                facecolor = cmap(norm(value_grid[i, j]))
            else:
                facecolor = 'white'

            rect = Rectangle((j, grid_size - 1 - i), 1, 1,
                              facecolor=facecolor, edgecolor='black', linewidth=1)
            ax.add_patch(rect)

            if values is not None and not (holes and state_idx in holes):
                ax.text(j + 0.5, grid_size - 1 - i + 0.75, f"{value_grid[i, j]:.1f}",
                        ha='center', va='center', fontsize=12)

            if state_labels is not None and state_idx in state_labels:
                ax.text(j + 0.5, grid_size - 1 - i + 0.25, state_labels[state_idx],
                        ha='center', va='center', fontsize=10, fontweight='bold')

            if (policy is not None and state_idx in policy
                    and policy[state_idx] is not None
                    and not (holes and state_idx in holes)
                    and state_idx != goal_state):
                action = policy[state_idx]
                dx, dy = 0, 0
                if action == 0:    dx, dy = 0, 0.4
                elif action == 1:  dx, dy = 0.4, 0
                elif action == 2:  dx, dy = 0, -0.4
                elif action == 3:  dx, dy = -0.4, 0
                if dx != 0 or dy != 0:
                    arrow = Arrow(j + 0.5, grid_size - 1 - i + 0.5, dx, dy,
                                  width=0.2, color='black')
                    ax.add_patch(arrow)

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_xticks(np.arange(0, grid_size + 1, 1))
    ax.set_yticks(np.arange(0, grid_size + 1, 1))
    ax.grid(True, color='black', linewidth=1)
    ax.set_axisbelow(True)

    if values is not None and cmap is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='State Value')

    ax.set_aspect('equal')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title:
        ax.set_title(title, fontsize=14)

    return ax


def visualize_agent_path(env, path, policy=None, values=None, holes=None, title=None):
    """Visualize the path taken by an agent on a grid world.

    Args:
        env: The grid environment.
        path: List of state IDs visited.
        policy: Optional policy dict.
        values: Optional value function dict.
        holes: Set of hole state IDs.
        title: Title for the plot.

    Returns:
        Matplotlib axis.
    """
    grid_size = int(np.sqrt(len(env.states)))
    fig, ax = plt.subplots(figsize=(10, 10))
    goal_state = len(env.states) - 1

    state_labels = {}
    for s in env.states:
        if s.terminal:
            if holes and s.id in holes:
                state_labels[s.id] = "HOLE"
            elif s.id == goal_state:
                state_labels[s.id] = "GOAL"

    visualize_grid_world(values=values, policy=policy, grid_size=grid_size,
                         state_labels=state_labels, holes=holes,
                         goal_state=goal_state, ax=ax)

    for i, state_id in enumerate(path):
        row = state_id // grid_size
        col = state_id % grid_size
        y = grid_size - 1 - row
        x = col

        if i == 0:
            marker, color, label, size = 'o', 'blue', 'Start', 200
        elif i == len(path) - 1:
            marker, color, label, size = '*', 'green', 'End', 300
        else:
            marker, color, label, size = '.', 'purple', None, 100

        if i == 0 or i == len(path) - 1 or i % 2 == 0:
            ax.scatter(x + 0.5, y + 0.5, marker=marker, color=color, s=size,
                       label=label if i <= 1 else None, zorder=10)

    for i in range(len(path) - 1):
        s_row, s_col = divmod(path[i], grid_size)
        e_row, e_col = divmod(path[i + 1], grid_size)
        sx, sy = s_col + 0.5, grid_size - 1 - s_row + 0.5
        ex, ey = e_col + 0.5, grid_size - 1 - e_row + 0.5
        ax.plot([sx, ex], [sy, ey], 'b-', alpha=0.6, linewidth=2, zorder=5)

    if title:
        ax.set_title(title, fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left',
              bbox_to_anchor=(1.05, 1), fontsize=12)
    plt.tight_layout()
    return ax
