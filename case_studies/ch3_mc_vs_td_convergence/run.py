"""MC vs TD(0) convergence comparison on Windy Desert."""

import math
import numpy as np

from rlbook.environments import create_windy_desert_env
from rlbook.algorithms.dynamic_programming import value_iteration, extract_policy
from rlbook.algorithms.tabular import (
    make_epsilon_greedy_policy, first_visit_mc_prediction, td0_prediction
)

env = create_windy_desert_env()
V_opt = value_iteration(env, theta=1e-3, max_iterations=500)
greedy = extract_policy(env, V_opt)

# Notebook cell 16: epsilon = 0.0
policy = make_epsilon_greedy_policy(greedy, env.actions, epsilon=0.0)

# Notebook cell 17: total_episodes = 5000
V_mc = first_visit_mc_prediction(policy, env, num_episodes=5000, gamma=1.0)
V_td = td0_prediction(policy, env, num_episodes=5000, gamma=1.0)

# Compare RMSE to optimal
states = [s.id for s in env.states if not s.terminal]
rmse_mc = math.sqrt(np.mean([(V_mc[s] - V_opt[s])**2 for s in states]))
rmse_td = math.sqrt(np.mean([(V_td[s] - V_opt[s])**2 for s in states]))

print(f"MC RMSE to optimal: {rmse_mc:.4f}")
print(f"TD RMSE to optimal: {rmse_td:.4f}")
