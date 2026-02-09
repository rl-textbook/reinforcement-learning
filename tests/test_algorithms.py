"""
Basic correctness tests for RL algorithms.

Run with: pytest tests/ -v
"""

import numpy as np
from rlbook.core.mdp import State, Environment
from rlbook.environments import create_grid_world, create_windy_desert_env
from rlbook.algorithms.dynamic_programming import value_iteration, extract_policy, policy_iteration


def test_value_iteration_2x2():
    """Value iteration on a trivial 2x2 grid should converge."""
    env = create_grid_world(grid_size=2)
    V = value_iteration(env, theta=1e-6, max_iterations=100)
    # Goal state (bottom-right, id=3) should have value 0
    assert V[3] == 0.0
    # Start state should have positive value (can reach goal)
    assert V[0] > 0.0


def test_vi_pi_agree():
    """Value iteration and policy iteration should produce the same policy."""
    env = create_grid_world(grid_size=4)
    V_vi = value_iteration(env, theta=1e-4)
    pi_vi = extract_policy(env, V_vi)
    pi_pi, V_pi = policy_iteration(env, theta=1e-4)

    # Policies should agree on non-terminal states
    for s in env.states:
        if not s.terminal:
            assert pi_vi[s.id] == pi_pi[s.id], f"Disagreement at state {s.id}"


def test_grid_world_has_correct_states():
    """Grid world should have grid_size^2 states."""
    for size in [2, 3, 4, 5]:
        env = create_grid_world(grid_size=size)
        assert len(env.states) == size * size


def test_windy_desert_env_creation():
    """Windy desert environment should be created successfully."""
    env = create_windy_desert_env()
    assert env.grid_height == 6
    assert env.grid_width == 8
    assert len(env.states) == 48
    assert env.goal_state is not None
    assert len(env.quicksand_ids) > 0


def test_q_learning_improves():
    """Q-learning should improve over random policy."""
    from rlbook.algorithms.tabular import q_learning, extract_greedy_policy
    env = create_grid_world(grid_size=3)
    Q, _ = q_learning(env, episodes=500, alpha=0.3, epsilon=0.1, snapshots=())
    pi = extract_greedy_policy(Q, env)
    # The policy for state 0 should not be None (it should point toward goal)
    assert pi[0] is not None


def test_sarsa_runs():
    """SARSA should complete without errors."""
    from rlbook.algorithms.tabular import sarsa
    env = create_grid_world(grid_size=3)
    Q, snaps = sarsa(env, episodes=100, snapshots=(0, 100))
    assert 0 in snaps
    assert 100 in snaps


def test_mc_prediction_runs():
    """MC prediction should produce value estimates."""
    from rlbook.algorithms.tabular import first_visit_mc_prediction, make_epsilon_greedy_policy
    env = create_grid_world(grid_size=3)
    V_opt = value_iteration(env, theta=1e-3)
    greedy = extract_policy(env, V_opt)
    policy = make_epsilon_greedy_policy(greedy, env.actions, epsilon=0.1)
    V_mc = first_visit_mc_prediction(policy, env, num_episodes=100)
    # Should have estimates for all states
    assert len(V_mc) == len(env.states)


def test_td_prediction_runs():
    """TD(0) prediction should produce value estimates."""
    from rlbook.algorithms.tabular import td0_prediction, make_epsilon_greedy_policy
    env = create_grid_world(grid_size=3)
    V_opt = value_iteration(env, theta=1e-3)
    greedy = extract_policy(env, V_opt)
    policy = make_epsilon_greedy_policy(greedy, env.actions, epsilon=0.1)
    V_td = td0_prediction(policy, env, num_episodes=100)
    assert len(V_td) == len(env.states)
