"""
Shared helper functions for tabular RL algorithms.

Includes epsilon-greedy action selection, greedy policy extraction,
and policy evaluation utilities used across Chapter 3.
"""

import random
from typing import Dict, List, Tuple, Optional
import numpy as np


def make_epsilon_greedy_policy(
    greedy_policy: Dict[int, Optional[int]],
    actions: List[int],
    epsilon: float
) -> Dict[int, Dict[int, float]]:
    """
    Convert a deterministic greedy policy {state_id -> action or None}
    into a stochastic ε-greedy policy {state_id -> {action -> prob}}.
    Terminal states map to {}.
    """
    nA = len(actions)
    pi: Dict[int, Dict[int,float]] = {}

    for s, a_star in greedy_policy.items():
        if a_star is None:
            pi[s] = {}
            continue
        # Base prob on all actions
        probs = {a: epsilon / nA for a in actions}
        # Add (1 - epsilon) mass to greedy action
        probs[a_star] += (1.0 - epsilon)
        pi[s] = probs
    return pi


def epsilon_greedy(Q, state_id, actions, epsilon=0.1):
    if random.random() < epsilon:
        return random.choice(actions)
    q = [Q[(state_id, a)] for a in actions]
    m = max(q)
    best = [a for a,qa in zip(actions, q) if qa == m]
    return random.choice(best)


def extract_greedy_policy(Q: Dict[Tuple[int,int], float], env) -> Dict[int, Optional[int]]:
    pi = {}
    for s in env.states:
        if s.terminal:
            pi[s.id] = None
        else:
            q = [Q[(s.id, a)] for a in env.actions]
            m = max(q)
            best = [a for a,qa in zip(env.actions, q) if qa == m]
            pi[s.id] = random.choice(best)
    return pi


def evaluate_policy(env, policy: Dict[int, Optional[int]], episodes=100, max_steps=500):
    """Run deterministic greedy policy; return mean return, mean steps, success rate."""
    total_returns, total_steps, successes = [], [], []
    for _ in range(episodes):
        s = env.reset()
        ret, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            a = policy.get(s.id, None)
            if a is None:  # terminal
                break
            s, r, done = env.step(a)
            ret += r
            steps += 1
        total_returns.append(ret)
        total_steps.append(steps)
        successes.append(1 if s == env.goal_state else 0)
    return (float(np.mean(total_returns)),
            float(np.mean(total_steps)),
            float(100.0*np.mean(successes)))
