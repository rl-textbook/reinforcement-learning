"""
Grid World Environment for Chapter 2 case studies.

A configurable grid world MDP with optional transition noise and holes.
See Chapter 2 (Fundamentals of Reinforcement Learning) for the
complete description and case studies using this environment.
"""

from typing import List, Dict, Tuple, Optional
from rlbook.core.mdp import State, Environment


def create_grid_world(grid_size=4, transition_noise=0.0, holes=None):
    """
    Create a grid world MDP with optional transition noise and holes.

    Args:
        grid_size: Size of the grid
        transition_noise: Probability of a random transition (0 means deterministic)
        holes: List of (row, col) tuples representing non-traversable cells
               If None, no holes are created

    Returns:
        Environment object
    """
    # Default holes if none specified (can be empty list for no holes)
    if holes is None:
        holes = []

    # Convert holes to set of indices for faster lookup
    hole_indices = set()
    for row, col in holes:
        if 0 <= row < grid_size and 0 <= col < grid_size:
            hole_indices.add(row * grid_size + col)

    # Create states
    states = []
    for i in range(grid_size * grid_size):
        is_goal = (i == grid_size * grid_size - 1)  # Bottom-right is goal
        is_hole = i in hole_indices
        terminal = is_goal or is_hole  # Both goal and holes are terminal
        states.append(State(id=i, name=f"S{i}", terminal=terminal))

    # Define actions: 0=up, 1=right, 2=down, 3=left
    actions = [0, 1, 2, 3]

    # Create environment
    env = Environment(states, actions)

    # Helper function to get state index from grid position
    def get_state_idx(row, col):
        if row < 0 or row >= grid_size or col < 0 or col >= grid_size:
            return None
        return row * grid_size + col

    # Helper function to get grid position from state index
    def get_grid_pos(state_idx):
        row = state_idx // grid_size
        col = state_idx % grid_size
        return row, col

    # Define transitions and rewards
    for row in range(grid_size):
        for col in range(grid_size):
            state_idx = get_state_idx(row, col)
            state = states[state_idx]

            # Skip if this state is terminal (goal or hole)
            if state.terminal:
                continue

            # For each action
            for action in actions:
                # Determine intended next position
                next_row, next_col = row, col

                if action == 0:  # Up
                    next_row = max(0, row - 1)
                elif action == 1:  # Right
                    next_col = min(grid_size - 1, col + 1)
                elif action == 2:  # Down
                    next_row = min(grid_size - 1, row + 1)
                elif action == 3:  # Left
                    next_col = max(0, col - 1)

                intended_next_state_idx = get_state_idx(next_row, next_col)
                intended_next_state = states[intended_next_state_idx]

                # If no noise, transition is deterministic
                if transition_noise == 0:
                    env.set_transition_probs(state, action, {intended_next_state: 1.0})
                else:
                    # With noise, we might end up in any adjacent cell
                    transition_probs = {}

                    # Add intended transition with higher probability
                    transition_probs[intended_next_state] = 1.0 - transition_noise

                    # Add possible noisy transitions to adjacent cells
                    valid_noise_actions = [a for a in actions if a != action]
                    noise_prob = transition_noise / len(valid_noise_actions)

                    for noise_action in valid_noise_actions:
                        noise_row, noise_col = row, col

                        if noise_action == 0:  # Up
                            noise_row = max(0, row - 1)
                        elif noise_action == 1:  # Right
                            noise_col = min(grid_size - 1, col + 1)
                        elif noise_action == 2:  # Down
                            noise_row = min(grid_size - 1, row + 1)
                        elif noise_action == 3:  # Left
                            noise_col = max(0, col - 1)

                        noise_state_idx = get_state_idx(noise_row, noise_col)
                        noise_state = states[noise_state_idx]

                        if noise_state in transition_probs:
                            transition_probs[noise_state] += noise_prob
                        else:
                            transition_probs[noise_state] = noise_prob

                    env.set_transition_probs(state, action, transition_probs)

                # Define rewards based on next state type
                for next_state in states:
                    next_state_idx = next_state.id
                    if next_state_idx == grid_size * grid_size - 1:
                        # Goal state - big positive reward
                        env.set_reward(state, action, next_state, 10)
                    elif next_state_idx in hole_indices:
                        # Hole state - big negative reward (penalty for falling in)
                        env.set_reward(state, action, next_state, -10)
                    else:
                        # Regular step - small negative reward
                        env.set_reward(state, action, next_state, -1)

    return env
