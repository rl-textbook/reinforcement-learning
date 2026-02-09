"""
SARSA: On-Policy TD Control.

Learns the action-value function using on-policy updates following
the current epsilon-greedy policy. See Chapter 3, Algorithm 3.4
for pseudocode.
"""

from rlbook.algorithms.tabular.helpers import epsilon_greedy, extract_greedy_policy


def sarsa(env, episodes=4000, alpha=0.3, gamma=1.0, epsilon=0.1,
          snapshots=(0, 50, 200, 1000, 4000), max_steps=400):
    Q = {(s.id,a): 0.0 for s in env.states for a in env.actions}
    snap_policies = {0: extract_greedy_policy(Q, env)} if 0 in snapshots else {}
    for ep in range(1, episodes+1):
        s = env.reset()
        a = epsilon_greedy(Q, s.id, env.actions, epsilon)
        done = False; t = 0
        while not done and t < max_steps:
            s2, r, done = env.step(a)
            if done:
                target = r
                Q[(s.id,a)] += alpha * (target - Q[(s.id,a)])
                break
            a2 = epsilon_greedy(Q, s2.id, env.actions, epsilon)
            target = r + gamma * Q[(s2.id, a2)]
            Q[(s.id,a)] += alpha * (target - Q[(s.id,a)])
            s, a = s2, a2; t += 1
        if ep in snapshots:
            snap_policies[ep] = extract_greedy_policy(Q, env)
    return Q, snap_policies
