"""
TD(0) prediction for Chapter 3.
"""

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
    # Step 1: Initialize values and per - state visit counts
    V = {state.id: 0.0 for state in env.states}
    visit_count = {state.id: 0 for state in env.states}

    for _ in range(num_episodes):
        # Step 2: Start a new episode
        state = env.reset()
        if visit is not None:
            visit.step(state.id)
        done = False
        steps = 0

        # Step 3: Follow the policy and update after each transition
        while not done and steps < max_steps:
            probs = policy.get(state.id, {})
            if not probs:
                break
            actions, ps = zip(*probs.items())
            action = np.random.choice(actions, p=ps)

            # Step 4: Sample the next state and reward
            next_state, reward, done = env.step(action)

            # Step 5: Use a diminishing per-state step size: α_t(s) = α0 / N_t(s)
            visit_count[state.id] += 1
            alpha = alpha0 / visit_count[state.id]

            # Step 6: Apply the one - step TD update
            V[state.id] += alpha * (reward + gamma * V[next_state.id] - V[state.id])

            # Step 7: Advance the rollout
            state = next_state
            if visit is not None:
                visit.step(state.id)

            steps += 1

        if visit is not None:
            visit.end_episode()

    return V
