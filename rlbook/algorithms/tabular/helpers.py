"""
Shared helper functions for Chapter 3 tabular algorithms.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


def make_epsilon_greedy_policy(
    greedy_policy: Dict[int, Optional[int]],
    actions: List[int],
    epsilon: float,
) -> Dict[int, Dict[int, float]]:
    """
    Convert a deterministic policy {state_id -> action or None}
    into a stochastic epsilon-greedy policy {state_id -> {action -> prob}}.
    """
    nA = len(actions)
    policy: Dict[int, Dict[int, float]] = {}

    for state_id, greedy_action in greedy_policy.items():
        if greedy_action is None:
            policy[state_id] = {}
            continue

        probs = {action: epsilon / nA for action in actions}
        probs[greedy_action] += 1.0 - epsilon
        policy[state_id] = probs

    return policy


def self_loop_probability(env, state_id: int, action: int) -> float:
    """
    Probability that action leaves the agent in the same state.
    """
    return sum(
        prob
        for next_state, prob in env.transitions.get((state_id, action), {}).items()
        if next_state.id == state_id
    )


def greedy_action_from_Q(
    Q: Dict[Tuple[int, int], float],
    env,
    state_id: int,
    tie_tol: float = 1e-12,
    prefer_progress: bool = True,
) -> int:
    """
    Deterministic greedy action selection.

    For exact or near-exact ties, optionally prefer actions with lower self-loop
    probability so the displayed greedy policy is more informative in the windy grid.
    """
    q_values = {action: Q[(state_id, action)] for action in env.actions}
    best_q = max(q_values.values())
    best_actions = [
        action for action, value in q_values.items()
        if abs(value - best_q) <= tie_tol
    ]

    if prefer_progress and len(best_actions) > 1:
        loop_probs = {
            action: self_loop_probability(env, state_id, action)
            for action in best_actions
        }
        min_loop = min(loop_probs.values())
        best_actions = [
            action for action in best_actions
            if abs(loop_probs[action] - min_loop) <= 1e-12
        ]

    return min(best_actions)


def epsilon_greedy(
    Q: Dict[Tuple[int, int], float],
    env,
    state_id: int,
    epsilon: float = 0.1,
    tie_tol: float = 1e-12,
    prefer_progress: bool = True,
) -> int:
    if random.random() < epsilon:
        return random.choice(env.actions)
    return greedy_action_from_Q(
        Q, env, state_id, tie_tol=tie_tol, prefer_progress=prefer_progress
    )


def extract_greedy_policy(
    Q: Dict[Tuple[int, int], float],
    env,
    tie_tol: float = 1e-12,
    prefer_progress: bool = True,
) -> Dict[int, Optional[int]]:
    policy: Dict[int, Optional[int]] = {}
    for state in env.states:
        if state.terminal:
            policy[state.id] = None
        else:
            policy[state.id] = greedy_action_from_Q(
                Q, env, state.id, tie_tol=tie_tol, prefer_progress=prefer_progress
            )
    return policy


def evaluate_policy(env, policy: Dict[int, Optional[int]], episodes: int = 100, max_steps: int = 500):
    """
    Roll out a deterministic policy and return:
        (mean return, mean episode length, success rate in percent)
    """
    total_returns = []
    total_steps = []
    successes = []

    for _ in range(episodes):
        state = env.reset()
        episode_return = 0.0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action = policy.get(state.id, None)
            if action is None:
                break
            state, reward, done = env.step(action)
            episode_return += reward
            steps += 1

        total_returns.append(episode_return)
        total_steps.append(steps)
        successes.append(1 if state == env.goal_state else 0)

    return (
        float(np.mean(total_returns)),
        float(np.mean(total_steps)),
        float(100.0 * np.mean(successes)),
    )
