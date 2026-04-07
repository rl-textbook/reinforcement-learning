"""
First-visit Monte Carlo prediction for Chapter 3.
"""

from typing import Dict, List, Tuple

import numpy as np


def generate_episode(
    env,
    policy: Dict[int, Dict[int, float]],
    max_steps: int = 500,
    visit: "VisitationCounter | None" = None,
) -> List[Tuple[int, int, float]]:
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

        actions, probs = zip(*probs_dict.items())
        action = np.random.choice(actions, p=probs)
        next_state, reward, done = env.step(action)

        episode.append((state.id, action, reward))
        state = next_state

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
    """
    True first-visit MC prediction.

    This matches the patched Chapter 3 notebook: we identify the first forward
    occurrence of each state in the sampled episode, then update from the return
    starting at that first visit.
    """
    V = {state.id: 0.0 for state in env.states}
    returns_sum = {state.id: 0.0 for state in env.states}
    returns_count = {state.id: 0 for state in env.states}

    for _ in range(num_episodes):
        episode = generate_episode(env, policy, max_steps=max_steps, visit=visit)

        first_visit_index = {}
        for t, (state_id, _, _) in enumerate(episode):
            if state_id not in first_visit_index:
                first_visit_index[state_id] = t

        G = 0.0
        returns = [0.0] * len(episode)
        for t in reversed(range(len(episode))):
            _, _, reward_t = episode[t]
            G = gamma * G + reward_t
            returns[t] = G

        for state_id, t_first in first_visit_index.items():
            returns_sum[state_id] += returns[t_first]
            returns_count[state_id] += 1
            V[state_id] = returns_sum[state_id] / returns_count[state_id]

    return V
