"""
First-Visit Monte Carlo Prediction.

Estimates state values by averaging returns from first visits
in sampled episodes. See Chapter 3, Algorithm 3.1 for pseudocode.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


def generate_episode(env, policy: Dict[int, Dict[int, float]], max_steps: int = 500,
                     visit: 'VisitationCounter|None' = None):   # NEW
    episode: List[Tuple[int,int,float]] = []
    state = env.reset()
    if visit is not None: visit.step(state.id)                  # NEW
    done = False; steps = 0
    while not done and steps < max_steps:
        probs_dict = policy.get(state.id, {})
        if not probs_dict: break
        acts, ps = zip(*probs_dict.items())
        a = np.random.choice(acts, p=ps)
        nxt, r, done = env.step(a)
        episode.append((state.id, a, r))
        state = nxt
        if visit is not None: visit.step(state.id)              # NEW
        steps += 1
    if visit is not None: visit.end_episode()                   # NEW
    return episode


def first_visit_mc_prediction(policy, env, num_episodes: int, gamma: float = 1.0,
                              max_steps: int = 500, visit: 'VisitationCounter|None' = None):
    V = {s.id: 0.0 for s in env.states}
    returns_sum = {s.id: 0.0 for s in env.states}
    returns_count = {s.id: 0 for s in env.states}

    for _ in range(num_episodes):
        episode = generate_episode(env, policy, max_steps=max_steps, visit=visit)  # pass counter
        G = 0.0
        visited = set()
        for t in reversed(range(len(episode))):
            s_t, _, r_t = episode[t]
            G = gamma * G + r_t
            if s_t not in visited:
                visited.add(s_t)
                returns_sum[s_t] += G
                returns_count[s_t] += 1
                V[s_t] = returns_sum[s_t] / max(1, returns_count[s_t])
    return V
