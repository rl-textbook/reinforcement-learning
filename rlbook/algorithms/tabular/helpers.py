"""Shared tabular RL helper functions."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


def make_epsilon_greedy_policy(
    greedy_policy: Dict[int, Optional[int]],
    actions: List[int],
    epsilon: float,
) -> Dict[int, Dict[int, float]]:
    """Convert a deterministic greedy policy into a stochastic ε-greedy policy."""
    nA = len(actions)
    pi: Dict[int, Dict[int, float]] = {}
    for s, a_star in greedy_policy.items():
        if a_star is None:
            pi[s] = {}
            continue
        probs = {a: epsilon / nA for a in actions}
        probs[a_star] += 1.0 - epsilon
        pi[s] = probs
    return pi


def self_loop_probability(env, state_id: int, action: int) -> float:
    """Probability that action a leaves the agent in the same state."""
    return sum(
        p
        for nxt, p in env.transitions.get((state_id, action), {}).items()
        if nxt.id == state_id
    )


def greedy_action_from_Q(
    Q: Dict[Tuple[int, int], float],
    env,
    state_id: int,
    tie_tol: float = 1e-12,
    prefer_progress: bool = True,
) -> int:
    """Deterministic greedy action selection with an optional self-loop tie-break."""
    q_vals = {a: Q[(state_id, a)] for a in env.actions}
    best_q = max(q_vals.values())
    best_actions = [a for a, q in q_vals.items() if abs(q - best_q) <= tie_tol]

    if prefer_progress and len(best_actions) > 1:
        loop_probs = {a: self_loop_probability(env, state_id, a) for a in best_actions}
        min_loop = min(loop_probs.values())
        best_actions = [a for a in best_actions if abs(loop_probs[a] - min_loop) <= 1e-12]

    return min(best_actions)


def epsilon_greedy(
    Q: Dict[Tuple[int, int], float],
    env,
    state_id: int,
    epsilon: float = 0.1,
    tie_tol: float = 1e-12,
    prefer_progress: bool = True,
) -> int:
    """ε-greedy action selection for control."""
    if random.random() < epsilon:
        return random.choice(env.actions)
    return greedy_action_from_Q(
        Q,
        env,
        state_id,
        tie_tol=tie_tol,
        prefer_progress=prefer_progress,
    )


def extract_greedy_policy(
    Q: Dict[Tuple[int, int], float],
    env,
    tie_tol: float = 1e-12,
    prefer_progress: bool = True,
) -> Dict[int, Optional[int]]:
    """Extract a deterministic greedy policy from an action-value function."""
    pi: Dict[int, Optional[int]] = {}
    for s in env.states:
        if s.terminal:
            pi[s.id] = None
        else:
            pi[s.id] = greedy_action_from_Q(
                Q,
                env,
                s.id,
                tie_tol=tie_tol,
                prefer_progress=prefer_progress,
            )
    return pi


def evaluate_policy(env, policy: Dict[int, Optional[int]], episodes: int = 100, max_steps: int = 500):
    """Run a deterministic policy and report mean return, mean steps, and success rate."""
    total_returns, total_steps, successes = [], [], []
    for _ in range(episodes):
        s = env.reset()
        ret, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            a = policy.get(s.id, None)
            if a is None:
                break
            s, r, done = env.step(a)
            ret += r
            steps += 1
        total_returns.append(ret)
        total_steps.append(steps)
        successes.append(1 if s == env.goal_state else 0)
    return (
        float(np.mean(total_returns)),
        float(np.mean(total_steps)),
        float(100.0 * np.mean(successes)),
    )
