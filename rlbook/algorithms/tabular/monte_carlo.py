"""First-visit Monte Carlo prediction."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def generate_episode(
    env,
    policy: Dict[int, Dict[int, float]],
    max_steps: int = 500,
    visit: "VisitationCounter | None" = None,
):
    """Sample one episode from the environment under a stochastic policy."""
    episode: List[Tuple[int, int, float]] = []
    state = env.reset()
    if visit is not None:
        visit.step(state.id)

    done = False
    steps = 0
    while not done and steps < max_steps:
        probs_dict = policy.get(state.id, {})
        if not probs_dict:
            break
        acts, ps = zip(*probs_dict.items())
        a = int(np.random.choice(acts, p=ps))
        nxt, r, done = env.step(a)
        episode.append((state.id, a, r))
        state = nxt
        if visit is not None:
            visit.step(state.id)
        steps += 1

    if visit is not None:
        visit.end_episode()
    return episode


def first_visit_mc_prediction(
    policy,
    env,
    num_episodes: int,
    gamma: float = 1.0,
    max_steps: int = 500,
    visit: "VisitationCounter | None" = None,
):
    """Estimate V under a fixed policy using true first-visit returns."""
    V = {s.id: 0.0 for s in env.states}
    returns_sum = {s.id: 0.0 for s in env.states}
    returns_count = {s.id: 0 for s in env.states}

    for _ in range(num_episodes):
        episode = generate_episode(env, policy, max_steps=max_steps, visit=visit)

        first_visit_index: dict[int, int] = {}
        for t, (s_t, _, _) in enumerate(episode):
            if s_t not in first_visit_index:
                first_visit_index[s_t] = t

        G = 0.0
        returns = [0.0] * len(episode)
        for t in reversed(range(len(episode))):
            _, _, r_t = episode[t]
            G = gamma * G + r_t
            returns[t] = G

        for s_t, t_first in first_visit_index.items():
            returns_sum[s_t] += returns[t_first]
            returns_count[s_t] += 1
            V[s_t] = returns_sum[s_t] / returns_count[s_t]

    return V
