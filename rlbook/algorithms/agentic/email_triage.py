"""
Agentic RL Components for Customer Support Email Triage.

Includes the LLM-based support agent, persistent customer memory system,
and baseline agents. The full progressive environment and scenario data
are in the Chapter 7 notebook.
See Chapter 7 (Agentic Architectures in Reinforcement Learning).
"""

import random
from typing import List, Dict
from collections import defaultdict

import numpy as np


class CustomerMemorySystem:
    """Global memory system that persists across episodes."""

    def __init__(self):
        self.customer_history = defaultdict(list)  # customer_id -> list of interactions
        self.interaction_count = defaultdict(int)  # customer_id -> count

    def record_interaction(self, customer_id: str, scenario_id: int,
                          actions_taken: List[int], outcome: str):
        """Record a customer interaction."""
        self.customer_history[customer_id].append({
            'scenario_id': scenario_id,
            'actions': actions_taken,
            'outcome': outcome,
            'interaction_num': self.interaction_count[customer_id]
        })
        self.interaction_count[customer_id] += 1

    def get_customer_history(self, customer_id: str) -> Dict:
        """Get customer's interaction history."""
        return {
            'previous_interactions': len(self.customer_history[customer_id]),
            'history': self.customer_history[customer_id],
            'is_repeat': len(self.customer_history[customer_id]) > 0
        }

    def encode_history(self, customer_id: str) -> np.ndarray:
        """Encode customer history as features."""
        history = self.customer_history[customer_id]

        if not history:
            return np.zeros(5)

        # Features: [has_history, num_interactions, most_common_action, escalation, recent_outcome]
        num_interactions = min(len(history) / 5.0, 1.0)  # Normalize

        # Most common action in history
        all_actions = [a for h in history for a in h['actions']]
        if all_actions:
            from collections import Counter
            common_action = Counter(all_actions).most_common(1)[0][0] / 11.0
        else:
            common_action = 0.0

        # Escalation (repeat customer)
        escalation = 1.0 if len(history) >= 2 else 0.0

        # Recent outcome encoding
        recent_outcome = 1.0 if history[-1]['outcome'] == 'success' else 0.0

        return np.array([1.0, num_interactions, common_action, escalation, recent_outcome])


class Level1RandomAgent:
    """Level 1: Pure random baseline."""

    def __init__(self, env):
        self.env = env

    def evaluate(self, num_episodes=100):
        results = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            terminated = False
            steps = 0

            while not terminated:
                action = np.random.randint(0, 11)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                steps += 1

            results.append({
                'reward': episode_reward,
                'tools_used': len(self.env.tools_used)
            })

        return {
            'mean_reward': np.mean([r['reward'] for r in results]),
            'success_rate': np.mean([r['reward'] > 0.5 for r in results]) * 100,
            'mean_tools': np.mean([r['tools_used'] for r in results])
        }


class Level2ClassificationAgent:
    """Level 2: LLM classification guides initial action."""

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def evaluate(self, num_episodes=100):
        results = []

        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            email = self.env.current_scenario['email']

            initial_action = self.agent.suggest_initial_action(email)

            episode_reward = 0
            terminated = False

            obs, reward, terminated, truncated, _ = self.env.step(initial_action)
            episode_reward += reward

            while not terminated:
                action = np.random.randint(0, 11)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward

            results.append({
                'reward': episode_reward,
                'tools_used': len(self.env.tools_used)
            })

        return {
            'mean_reward': np.mean([r['reward'] for r in results]),
            'success_rate': np.mean([r['reward'] > 0.5 for r in results]) * 100,
            'mean_tools': np.mean([r['tools_used'] for r in results])
        }
