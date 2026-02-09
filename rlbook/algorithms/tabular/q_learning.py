"""
Q-Learning: Off-Policy TD Control.

Learns the optimal action-value function using off-policy updates
with the max operator. See Chapter 3, Algorithm 3.3 for pseudocode.
"""

from rlbook.algorithms.tabular.helpers import epsilon_greedy, extract_greedy_policy


def q_learning(env, episodes=4000, alpha=0.3, gamma=1.0, epsilon=0.1,
               snapshots=(0, 50, 200, 1000, 4000), max_steps=400):
    Q = {(s.id,a): 0.0 for s in env.states for a in env.actions}
    snap_policies = {0: extract_greedy_policy(Q, env)} if 0 in snapshots else {}
    for ep in range(1, episodes+1):
        s = env.reset(); done = False; t = 0
        while not done and t < max_steps:
            a = epsilon_greedy(Q, s.id, env.actions, epsilon)
            s2, r, done = env.step(a)
            next_max = max(Q[(s2.id, aa)] for aa in env.actions)
            Q[(s.id,a)] += alpha * (r + gamma*next_max - Q[(s.id,a)])
            s = s2; t += 1
        if ep in snapshots:
            snap_policies[ep] = extract_greedy_policy(Q, env)
    return Q, snap_policies
