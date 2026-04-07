"""
SARSA control for Chapter 3.
"""

from rlbook.algorithms.tabular.helpers import epsilon_greedy, extract_greedy_policy


def sarsa(
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
    snap_policies = {0: extract_greedy_policy(Q, env)} if 0 in snapshots else {} # Optional snapshots for capturing diagnostics

    for ep in range(1, episodes + 1):
        # Step 2: Start a fresh episode and choose the initial behavior action
        state = env.reset()
        action = epsilon_greedy(Q, env, state.id, epsilon=epsilon)
        done = False
        steps = 0

        # Step 3: Roll the SARSA tuple (S , A , R , S ’, A ’) forward
        while not done and steps < max_steps:
            # Step 3 a : Take the current action and observe the next transition
            next_state, reward, done = env.step(action)

            # Step 3 b : Terminal transitions use the immediate reward only
            if done:
                td_target = reward
            else:
                # Step 3 c : Otherwise choose the next behavior action and bootstrap from it
                next_action = epsilon_greedy(Q, env, next_state.id, epsilon=epsilon)
                td_target = reward + gamma * Q[(next_state.id, next_action)]

            Q[(state.id, action)] += alpha * (td_target - Q[(state.id, action)])

            if done:
                break

            # Step 3 d : Shift to the next state - action pair
            state, action = next_state, next_action
            steps += 1

        if ep in snapshots:
            snap_policies[ep] = extract_greedy_policy(Q, env)

    return Q, snap_policies
