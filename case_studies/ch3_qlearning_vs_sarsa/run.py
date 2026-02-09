"""Q-Learning vs SARSA comparison on Windy Desert."""

from rlbook.environments import create_windy_desert_env, apply_reward_shaping
from rlbook.algorithms.tabular import q_learning, sarsa, evaluate_policy, extract_greedy_policy

# Notebook cell 29: stochastic env with reward shaping
env = create_windy_desert_env(p_intended=0.90, p_perp=0.05)
apply_reward_shaping(env, step_cost=-1.0, hole_penalty=-100.0, goal_reward=+10.0)

# Notebook cell 32: Q-learning 10k episodes, SARSA 1k episodes, alpha=0.3
Q_ql, snaps_ql = q_learning(env, episodes=10000, alpha=0.3, gamma=1.0, epsilon=0.1,
                             max_steps=400)
Q_sa, snaps_sa = sarsa(env, episodes=1000, alpha=0.3, gamma=1.0, epsilon=0.1,
                        max_steps=400)

pi_ql = extract_greedy_policy(Q_ql, env)
pi_sa = extract_greedy_policy(Q_sa, env)

ret_ql, steps_ql, success_ql = evaluate_policy(env, pi_ql, episodes=100)
ret_sa, steps_sa, success_sa = evaluate_policy(env, pi_sa, episodes=100)

print(f"Q-Learning (10k eps): return={ret_ql:.2f}, steps={steps_ql:.1f}, success={success_ql:.0f}%")
print(f"SARSA      (1k eps):  return={ret_sa:.2f}, steps={steps_sa:.1f}, success={success_sa:.0f}%")
