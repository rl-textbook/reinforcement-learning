"""Value Iteration vs Policy Iteration comparison on Grid World."""

from rlbook.environments import create_grid_world
from rlbook.algorithms.dynamic_programming import value_iteration, extract_policy, policy_iteration

# Deterministic 4x4 grid (basic sanity check)
env = create_grid_world(grid_size=4)
V_vi = value_iteration(env, theta=0.001)
pi_vi = extract_policy(env, V_vi)
pi_pi, V_pi = policy_iteration(env, theta=0.001)

# Compare only non-terminal states (extract_policy includes terminal states
# with None, while policy_iteration excludes them)
pi_vi_nonterminal = {s: a for s, a in pi_vi.items() if a is not None}

print("Value Iteration policy:", pi_vi)
print("Policy Iteration policy:", pi_pi)
print("Policies match (non-terminal):", pi_vi_nonterminal == pi_pi)

# Value function agreement
max_v_diff = max(abs(V_vi[s] - V_pi[s]) for s in V_vi)
print(f"Max value difference: {max_v_diff:.8f}")

# 5x5 deterministic grid with 3 holes (matches notebook cells 9, 12, 18, 20, 22)
env_hard = create_grid_world(grid_size=5, transition_noise=0.0, holes=[(1,1), (2,3), (3,1)])
V_vi_hard = value_iteration(env_hard, theta=0.001, max_iterations=100)
pi_vi_hard = extract_policy(env_hard, V_vi_hard)
pi_pi_hard, V_pi_hard = policy_iteration(env_hard, theta=0.001, max_iterations=1000)

pi_vi_hard_nt = {s: a for s, a in pi_vi_hard.items() if a is not None}

print("\n5x5 deterministic grid with 3 holes:")
print("Policies match (non-terminal):", pi_vi_hard_nt == pi_pi_hard)
max_v_diff_hard = max(abs(V_vi_hard[s] - V_pi_hard[s]) for s in V_vi_hard)
print(f"Max value difference: {max_v_diff_hard:.8f}")
