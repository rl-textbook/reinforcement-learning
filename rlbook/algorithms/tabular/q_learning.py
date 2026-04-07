"""Q-learning control."""

from __future__ import annotations

from helpers import epsilon_greedy, extract_greedy_policy


def q_learning(
    env,
    episodes: int = 4000,
    alpha: float = 0.3,
    gamma: float = 1.0,
    epsilon: float = 0.1,
    snapshots=(0, 50, 200, 1000, 4000),
    max_steps: int = 400,
):
    """Off-policy TD control with periodic greedy-policy snapshots."""
    Q = {(s.id, a): 0.0 for s in env.states for a in env.actions}
    snap_policies = {0: extract_greedy_policy(Q, env)} if 0 in snapshots else {}

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        t = 0
        while not done and t < max_steps:
            a = epsilon_greedy(Q, env, s.id, epsilon=epsilon)
            s2, r, done = env.step(a)
            td_target = r if done else r + gamma * max(Q[(s2.id, aa)] for aa in env.actions)
            Q[(s.id, a)] += alpha * (td_target - Q[(s.id, a)])
            s = s2
            t += 1
        if ep in snapshots:
            snap_policies[ep] = extract_greedy_policy(Q, env)

    return Q, snap_policies
