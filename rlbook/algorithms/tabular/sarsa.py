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
    Q = {(state.id, action): 0.0 for state in env.states for action in env.actions}
    snap_policies = {0: extract_greedy_policy(Q, env)} if 0 in snapshots else {}

    for ep in range(1, episodes + 1):
        state = env.reset()
        action = epsilon_greedy(Q, env, state.id, epsilon=epsilon)
        done = False
        steps = 0

        while not done and steps < max_steps:
            next_state, reward, done = env.step(action)

            if done:
                td_target = reward
            else:
                next_action = epsilon_greedy(Q, env, next_state.id, epsilon=epsilon)
                td_target = reward + gamma * Q[(next_state.id, next_action)]

            Q[(state.id, action)] += alpha * (td_target - Q[(state.id, action)])

            if done:
                break

            state, action = next_state, next_action
            steps += 1

        if ep in snapshots:
            snap_policies[ep] = extract_greedy_policy(Q, env)

    return Q, snap_policies
