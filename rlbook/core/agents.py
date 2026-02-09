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


def compare_agents(env: Environment, policies: Dict[str, Optional[Dict]],
                   episodes: int = 100, max_steps: int = 100) -> Dict[str, Dict]:
    """Compare multiple agents by running episodes and collecting metrics.

    Args:
        env: The environment.
        policies: Dict mapping label -> policy dict (None for random agent).
        episodes: Number of evaluation episodes.
        max_steps: Max steps per episode.

    Returns:
        Dict mapping label -> {'mean_reward', 'mean_steps', 'success_rate'}.
    """
    goal_state_id = env.states[-1].id
    results = {}

    for label, policy in policies.items():
        if policy is None:
            agent = RandomWalkAgent(env)
        else:
            agent = PolicyAgent(env, policy)

        rewards, steps, successes = [], [], []
        for _ in range(episodes):
            total_reward, num_steps, path = run_episode(env, agent, max_steps)
            rewards.append(total_reward)
            steps.append(num_steps)
            successes.append(1 if path[-1] == goal_state_id else 0)

        results[label] = {
            'mean_reward': float(np.mean(rewards)),
            'mean_steps': float(np.mean(steps)),
            'success_rate': float(100.0 * np.mean(successes)),
        }

    return results
