"""
Policy Iteration Algorithm.

Alternates between policy evaluation and policy improvement until
convergence. See Chapter 2, Algorithm 2.2 for pseudocode.
"""

import random
from typing import Dict, Tuple
from rlbook.core.mdp import Environment


def policy_evaluation(env: Environment, policy: Dict, theta: float = 0.001, max_iterations: int = 100) -> Dict:
    """
    Evaluate a given policy by computing its value function.

    Args:
        env: The MDP environment
        policy: The policy to evaluate
        theta: Convergence threshold
        max_iterations: Maximum number of iterations

    Returns:
        Value function for the given policy
    """
    V = {state.id: 0 for state in env.states}

    for i in range(max_iterations):
        delta = 0
        for state in env.states:
            if state.terminal:
                continue

            v = V[state.id]
            action = policy[state.id]

            # Compute value based on the current policy
            value = 0
            for next_state, prob in env.transitions.get((state.id, action), {}).items():
                reward = env.rewards.get((state.id, action, next_state.id), 0)
                value += prob * (reward + env.gamma * V[next_state.id])

            V[state.id] = value
            delta = max(delta, abs(v - V[state.id]))

        if delta < theta:
            break

    return V


def policy_iteration(env: Environment, theta: float = 0.001, max_iterations: int = 1000) -> Tuple[Dict, Dict]:
    """
    Solve an MDP using policy iteration.

    Args:
        env: The MDP environment
        theta: Convergence threshold for policy evaluation
        max_iterations: Maximum number of policy iteration steps

    Returns:
        Tuple containing optimal policy and its value function
    """
    # Initialize a random policy
    policy = {state.id: random.choice(env.actions) for state in env.states if not state.terminal}

    for i in range(max_iterations):
        # Step 1: Policy Evaluation
        V = policy_evaluation(env, policy, theta)

        # Step 2: Policy Improvement
        policy_stable = True

        for state in env.states:
            if state.terminal:
                continue

            old_action = policy[state.id]
            best_action = None
            best_value = float('-inf')

            # Evaluate all actions to find the best one
            for action in env.actions:
                action_value = 0
                for next_state, prob in env.transitions.get((state.id, action), {}).items():
                    reward = env.rewards.get((state.id, action, next_state.id), 0)
                    action_value += prob * (reward + env.gamma * V[next_state.id])

                if action_value > best_value:
                    best_value = action_value
                    best_action = action

            # Update policy if a better action is found
            policy[state.id] = best_action

            if old_action != best_action:
                policy_stable = False

        # Check for policy stability
        if policy_stable:
            print(f"Policy iteration converged after {i+1} iterations")
            break

    return policy, V
