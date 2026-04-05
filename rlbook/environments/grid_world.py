"""
Grid World Environment for Chapter 2 case studies.

A configurable grid world MDP with optional transition noise and holes.
See Chapter 2 (Fundamentals of Reinforcement Learning) for the
complete description and case studies using this environment.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from .core.mdp import Environment, State


def create_grid_world(
    grid_size: int = 4,
    transition_noise: float = 0.0,
    holes: Sequence[Tuple[int, int]] | None = None,
    gamma: float = 0.9,
) -> Environment:
    """Create a square grid-world MDP.

    The start state is the top-left cell and the goal is the bottom-right cell.
    Optional ``holes`` are terminal penalty states that the agent can enter.
    Regular transitions have reward -1, holes have reward -10, and the goal
    has reward +10.
    """
    holes = list(holes or [])
    hole_indices = {
        row * grid_size + col
        for row, col in holes
        if 0 <= row < grid_size and 0 <= col < grid_size
    }

    states: List[State] = []
    for row in range(grid_size):
        for col in range(grid_size):
            state_id = row * grid_size + col
            is_goal = state_id == grid_size * grid_size - 1
            is_hole = state_id in hole_indices
            states.append(
                State(
                    id=state_id,
                    name=(row, col),
                    terminal=is_goal or is_hole,
                )
            )

    actions = [0, 1, 2, 3]  # up, right, down, left
    env = Environment(states, actions, gamma=gamma)

    def clamp_position(row: int, col: int) -> tuple[int, int]:
        return max(0, min(grid_size - 1, row)), max(0, min(grid_size - 1, col))

    def move(row: int, col: int, action: int) -> tuple[int, int]:
        if action == 0:
            return clamp_position(row - 1, col)
        if action == 1:
            return clamp_position(row, col + 1)
        if action == 2:
            return clamp_position(row + 1, col)
        if action == 3:
            return clamp_position(row, col - 1)
        raise ValueError(f"Unknown action: {action}")

    for row in range(grid_size):
        for col in range(grid_size):
            state_id = row * grid_size + col
            state = states[state_id]
            if state.terminal:
                continue

            for action in actions:
                intended_row, intended_col = move(row, col, action)
                intended_idx = intended_row * grid_size + intended_col
                intended_state = states[intended_idx]

                if transition_noise == 0:
                    transition_probs = {intended_state: 1.0}
                else:
                    transition_probs = {intended_state: 1.0 - transition_noise}
                    other_actions = [a for a in actions if a != action]
                    noise_prob = transition_noise / len(other_actions)

                    for other_action in other_actions:
                        noise_row, noise_col = move(row, col, other_action)
                        noise_idx = noise_row * grid_size + noise_col
                        noise_state = states[noise_idx]
                        transition_probs[noise_state] = transition_probs.get(noise_state, 0.0) + noise_prob

                env.set_transition_probs(state, action, transition_probs)

                for next_state in transition_probs:
                    next_state_idx = next_state.id
                    if next_state_idx == grid_size * grid_size - 1:
                        reward = 10.0
                    elif next_state_idx in hole_indices:
                        reward = -10.0
                    else:
                        reward = -1.0
                    env.set_reward(state, action, next_state, reward)

    return env


__all__ = ["create_grid_world"]
