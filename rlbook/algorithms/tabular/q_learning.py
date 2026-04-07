"""
Q-learning control for Chapter 3.
"""

from rlbook.algorithms.tabular.helpers import epsilon_greedy, extract_greedy_policy


def q_learning(
    env,
    episodes: int = 4000,
    alpha: float = 0.3,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    snapshots=(0, 50, 200, 1000, 4000),
    max_steps: int = 400,
):
    # Step 1: Initialize the Q - table for every state - action pair
    Q = {(state.id, action): 0.0 for state in env.states for action in env.actions}
    snap_policies = {0: extract_greedy_policy(Q, env)} if 0 in snapshots else {} # Collecting snapshots for displaying diagnostics later on

    for ep in range(1, episodes + 1):
        # Step 2: Start a fresh episode
        state = env.reset()
        done = False
        steps = 0

        # Step 3: Interact until termination or the step cap
        while not done and steps < max_steps:
            # Step 3 a : Choose the behavior action using epsilon - greedy exploration
            action = epsilon_greedy(Q, env, state.id, epsilon=epsilon)
            
            # Step 3 b : Observe the stochastic transition and reward
            next_state, reward, done = env.step(action)

            # Step 3 c : Bootstrap from the best next action
            td_target = reward if done else reward + gamma * max(
                Q[(next_state.id, next_action)] for next_action in env.actions
            )
            Q[(state.id, action)] += alpha * (td_target - Q[(state.id, action)])

            # Step 3 d : Advance to the next state
            state = next_state
            steps += 1

        if ep in snapshots:
            snap_policies[ep] = extract_greedy_policy(Q, env) # Optional snapshots for diagnostics

    return Q, snap_policies
