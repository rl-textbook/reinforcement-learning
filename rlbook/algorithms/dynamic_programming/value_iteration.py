"""
Value Iteration Algorithm.

Solves an MDP by iteratively applying the Bellman optimality backup
until convergence. See Chapter 2, Algorithm 2.1 for pseudocode.
"""

from typing import Dict
from rlbook.core.mdp import Environment


def value_iteration(env: Environment, theta: float = 0.001, max_iterations: int = 100) -> Dict:
    """
    Solve an MDP using value iteration.

    Args:
        env: The MDP environment
        theta: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        Optimal value function
    """
    # Initialize value function
    V = {state.id: 0 for state in env.states}

    for i in range(max_iterations):
        delta = 0

        # Update each state
        for state in env.states:
            if state.terminal:
                continue

            v = V[state.id]

            # Find maximum action value
            max_value = float('-inf')

            for action in env.actions:
                if (state.id, action) not in env.transitions:
                    continue

                action_value = 0
                for next_state, prob in env.transitions[(state.id, action)].items():
                    reward = env.rewards.get((state.id, action, next_state.id), 0)
                    action_value += prob * (reward + env.gamma * V[next_state.id])

                max_value = max(max_value, action_value)

            # Update value function
            V[state.id] = max_value
            delta = max(delta, abs(v - V[state.id]))

        # Check convergence
        if delta < theta:
            print(f"Value iteration converged after {i+1} iterations")
            break

    return V


def extract_policy(env: Environment, V: Dict) -> Dict:
    """
    Extract the optimal policy from a value function.

    Args:
        env: The MDP environment
        V: The value function

    Returns:
        Optimal policy
    """
    policy = {}

    for state in env.states:
        if state.terminal:
            policy[state.id] = None
            continue

        # Find the best action
        best_action = None
        best_value = float('-inf')

        for action in env.actions:
            if (state.id, action) not in env.transitions:
                continue

            action_value = 0
            for next_state, prob in env.transitions[(state.id, action)].items():
                reward = env.rewards.get((state.id, action, next_state.id), 0)
                action_value += prob * (reward + env.gamma * V[next_state.id])

            if action_value > best_value:
                best_value = action_value
                best_action = action

        policy[state.id] = best_action

    return policy
