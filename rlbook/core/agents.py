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


def run_episode(env, agent, max_steps=100, render=False, delay=0.5, holes=None):
    """
    Run a single episode with the given agent.

    Args:
        env: The environment to run in
        agent: The agent to control actions
        max_steps: Maximum steps before termination
        render: Whether to visualize steps
        delay: Delay between rendered frames
        holes: Hole states as either state IDs or (row, col) tuples

    Returns:
        Tuple of (total_reward, steps, path)
    """
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    path = [state.id]

    if render:
        grid_size = int(np.sqrt(len(env.states)))
        _, ax = plt.subplots(figsize=(8, 8))
        goal_state = len(env.states) - 1
        hole_ids = normalize_holes(holes, grid_size)

    while not done and steps < max_steps:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        path.append(next_state.id)
        state = next_state
        steps += 1

        if render:
            ax.clear()
            state_labels = {path[-1]: "Agent"}
            visualize_grid_world(
                grid_size=grid_size,
                state_labels=state_labels,
                holes=hole_ids,
                goal_state=goal_state,
                start_state=0,
                ax=ax,
                title=f"Step {steps}, Reward: {total_reward:.1f}"
            )
            plt.pause(delay)

    if render:
        plt.close()

    return total_reward, steps, path


def compute_metrics(env, policy, num_episodes=100, max_steps=100):
    agent = PolicyAgent(env, policy) if policy is not None else RandomWalkAgent(env)
    rewards, steps, paths = [], [], []
    goal_state = env.states[-1].id

    for _ in range(num_episodes):
        total_reward, num_steps, path = run_episode(env, agent, max_steps=max_steps)
        rewards.append(float(total_reward))
        steps.append(int(num_steps))
        paths.append(path)

    success_rate = np.mean([1 if path[-1] == goal_state else 0 for path in paths])

    return {
        "mean_reward": float(np.mean(rewards)),
        "mean_steps": float(np.mean(steps)),
        "success_rate": float(success_rate),
        "best_path": paths[int(np.argmax(rewards))],
    }

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
    rl_results = compute_metrics(env, rl_policy, num_episodes=num_episodes)
    random_results = compute_metrics(env, None, num_episodes=num_episodes)

    results = {
        "rl_mean_reward": rl_results["mean_reward"],
        "rl_mean_steps": rl_results["mean_steps"],
        "rl_success_rate": rl_results["success_rate"],
        "random_mean_reward": random_results["mean_reward"],
        "random_mean_steps": random_results["mean_steps"],
        "random_success_rate": random_results["success_rate"],
        "rl_best_path": rl_results["best_path"],
        "random_best_path": random_results["best_path"],
    }


    return results

