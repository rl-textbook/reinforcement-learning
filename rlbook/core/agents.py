"""Agents for tabular MDP environments."""

import random
import numpy as np
from typing import Dict, Optional, Tuple, List

from rlbook.core.mdp import Environment, State


class PolicyAgent:
    """Agent that follows a deterministic policy."""

    def __init__(self, env: Environment, policy: Dict[int, Optional[int]]):
        self.env = env
        self.policy = policy

    def choose_action(self, state: State) -> int:
        if state.terminal or state.id not in self.policy or self.policy[state.id] is None:
            return random.choice(self.env.actions)
        return self.policy[state.id]


class RandomWalkAgent:
    """Agent that takes random actions."""

    def __init__(self, env: Environment):
        self.env = env

    def choose_action(self, state: State) -> int:
        return random.choice(self.env.actions)


def run_episode(env: Environment, agent, max_steps: int = 100) -> Tuple[float, int, List[int]]:
    """Run a single episode, returning (total_reward, steps, path)."""
    state = env.reset()
    total_reward = 0.0
    done = False
    steps = 0
    path = [state.id]

    while not done and steps < max_steps:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(next_state.id)
        state = next_state
        steps += 1

    return total_reward, steps, path


def compare_agents(env, rl_policy, random_agent, policy_agent, num_episodes=100, holes=None):
    """
    Compare RL agent with random walk on the same environment.

    Args:
        env: The environment
        rl_policy: Policy dictionary from RL
        random_agent: Random walk agent
        num_episodes: Number of episodes to run
        holes: Set of state IDs that are holes

    Returns:
        Comparison statistics
    """
    goal_state = len(env.states) - 1

    # Run episodes with both agents
    rl_rewards = []
    rl_steps = []
    rl_paths = []

    random_rewards = []
    random_steps = []
    random_paths = []

    for _ in range(num_episodes):
        # RL agent
        reward, steps, path = run_episode(env, policy_agent)
        rl_rewards.append(reward)
        rl_steps.append(steps)
        rl_paths.append(path)

        # Random agent
        reward, steps, path = run_episode(env, random_agent)
        random_rewards.append(reward)
        random_steps.append(steps)
        random_paths.append(path)

    # Calculate success rates (reaching goal, not falling in hole)
    rl_success_rate = np.mean([1 if path[-1] == goal_state else 0 for path in rl_paths])
    random_success_rate = np.mean([1 if path[-1] == goal_state else 0 for path in random_paths])

    results = {
        "rl_mean_reward": np.mean(rl_rewards),
        "rl_mean_steps": np.mean(rl_steps),
        "rl_success_rate": rl_success_rate,
        "random_mean_reward": np.mean(random_rewards),
        "random_mean_steps": np.mean(random_steps),
        "random_success_rate": random_success_rate,
        "rl_best_path": rl_paths[np.argmax(rl_rewards)],
        "random_best_path": random_paths[np.argmax(random_rewards)]
    }

    return results
