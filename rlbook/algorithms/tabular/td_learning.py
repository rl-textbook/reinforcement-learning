"""TD(0) prediction."""

from __future__ import annotations

import numpy as np


def td0_prediction(
    policy,
    env,
    num_episodes: int,
    gamma: float = 1.0,
    max_steps: int = 500,
    alpha0: float = 1.0,
    visit: "VisitationCounter | None" = None,
):
    """Estimate V online with a 1/N(s) step size."""
    V = {s.id: 0.0 for s in env.states}
    N = {s.id: 0 for s in env.states}

    for _ in range(num_episodes):
        state = env.reset()
        if visit is not None:
            visit.step(state.id)

        done = False
        steps = 0
        while not done and steps < max_steps:
            probs = policy.get(state.id, {})
            if not probs:
                break
            acts, ps = zip(*probs.items())
            a = int(np.random.choice(acts, p=ps))
            nxt, r, done = env.step(a)

            N[state.id] += 1
            alpha = alpha0 / N[state.id]
            V[state.id] += alpha * (r + gamma * V[nxt.id] - V[state.id])

            state = nxt
            if visit is not None:
                visit.step(state.id)
            steps += 1

        if visit is not None:
            visit.end_episode()

    return V
