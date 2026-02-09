"""Progressive email triage environment and evaluation (Chapter 7)."""

import random
from typing import Dict, List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EmailTriageEnvProgressive(gym.Env):
    """Environment supporting progressive capability levels with RL-learned tool use.

    Following ToolRL (Zeng et al., 2024) paradigm, tools are discrete actions
    in the RL policy space. The agent learns through PPO:
    - Which tools to invoke for each scenario
    - Optimal tool sequencing
    - When tool use is necessary vs. when to act directly

    Args:
        level: Capability level (1-6).
        memory_system: CustomerMemorySystem instance.
        agent: SupportAgent instance (required for level >= 2).
        scenarios: List of scenario dicts.
        investigation_plans: Dict mapping scenario_id -> plan text (required for level >= 5).
    """

    ACTION_NAMES = [
        'resolve',           # 0
        'check_product',     # 1
        'fraud_check',       # 2
        'search_orders',     # 3
        'check_shipping',    # 4
        'check_subscription',# 5
        'refund',            # 6
        'replacement',       # 7
        'cancel_subscription',# 8
        'access_memory',     # 9
        'security_lock',     # 10
    ]

    def __init__(self, level: int = 6, memory_system=None,
                 agent=None, scenarios=None, investigation_plans=None):
        super().__init__()

        from rlbook.algorithms.agentic.email_triage import CustomerMemorySystem
        from rlbook.algorithms.agentic.scenarios import EMAIL_SCENARIOS

        self.level = level
        self.memory_system = memory_system or CustomerMemorySystem()
        self.agent = agent
        self.scenarios = scenarios or EMAIL_SCENARIOS
        self.investigation_plans = investigation_plans or {}

        obs_dims = {1: 11, 2: 13, 3: 24, 4: 35, 5: 55, 6: 60}
        obs_size = obs_dims[level]
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(obs_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))

        self.current_scenario = None
        self.current_plan = {}
        self.tools_used = set()
        self.action_history = []
        self.steps_taken = 0
        self.max_steps = 5
        self.accessed_memory = False

    def _encode_plan(self, plan_text: str) -> np.ndarray:
        keywords = ['check', 'verify', 'search', 'order', 'shipping', 'refund',
                     'replace', 'fraud', 'cancel', 'subscription', 'product',
                     'payment', 'status', 'customer', 'urgent', 'angry',
                     'damaged', 'wrong', 'delay', 'issue']
        encoding = np.zeros(20)
        plan_lower = plan_text.lower()
        for i, keyword in enumerate(keywords):
            encoding[i] = min(plan_lower.count(keyword), 5) / 5.0
        return encoding

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_scenario = random.choice(self.scenarios)
        self.customer_id = self.current_scenario['customer_id']

        if self.level >= 2 and self.agent is not None:
            self.urgency = 1.0 if self.agent.classify_urgency(
                self.current_scenario['email']) == 'high' else 0.0
            categories = ['general', 'refund', 'replacement',
                          'cancellation', 'shipping', 'fraud']
            category = self.agent.classify_category(self.current_scenario['email'])
            self.category_encoding = categories.index(category) / len(categories)

        if self.level >= 5:
            plan_text = self.investigation_plans.get(
                self.current_scenario['id'], '')
            self.plan_encoding = self._encode_plan(plan_text)

            self.current_plan = {}
            plan_lower = plan_text.lower()
            action_keywords = {
                3: ['order', 'search'], 4: ['shipping', 'track'],
                6: ['refund'], 2: ['fraud'], 7: ['replace'],
                9: ['history', 'previous', 'before', 'again'],
            }
            for action_id, keywords in action_keywords.items():
                if any(kw in plan_lower for kw in keywords):
                    self.current_plan[action_id] = 0.1

        if self.level >= 6:
            self.memory_encoding = self.memory_system.encode_history(self.customer_id)
            self.customer_history = self.memory_system.get_customer_history(self.customer_id)

        self.tools_used = set()
        self.action_history = []
        self.steps_taken = 0
        self.accessed_memory = False

        return self._get_observation(), {}

    def _get_observation(self):
        obs_parts = [np.zeros(11)]

        if self.level >= 2:
            obs_parts.append(np.array([
                getattr(self, 'urgency', 0.0),
                getattr(self, 'category_encoding', 0.0)]))

        if self.level >= 3:
            tools_vector = np.zeros(11)
            for tool in self.tools_used:
                tools_vector[tool] = 1.0
            obs_parts.append(tools_vector)

        if self.level >= 4:
            action_vector = np.zeros(11)
            for a in self.action_history:
                action_vector[a] = min(action_vector[a] + 0.2, 1.0)
            obs_parts.append(action_vector)

        if self.level >= 5:
            obs_parts.append(getattr(self, 'plan_encoding', np.zeros(20)))

        if self.level >= 6:
            obs_parts.append(getattr(self, 'memory_encoding', np.zeros(5)))

        return np.concatenate(obs_parts).astype(np.float32)

    def step(self, action: int):
        self.steps_taken += 1
        self.action_history.append(action)

        reward = 0
        terminated = False

        if action in range(1, 11):
            self.tools_used.add(action)
            optimal_tools = set(self.current_scenario['optimal_actions'])
            if action in optimal_tools:
                reward += 0.05

            if self.level >= 5 and action in self.current_plan:
                reward += self.current_plan[action]

            if self.level >= 6 and action == 9:
                self.accessed_memory = True
                if self.current_scenario.get('is_repeat', False):
                    reward += 0.3
                elif hasattr(self, 'customer_history') and self.customer_history['is_repeat']:
                    reward += 0.15

        if action == 0 or self.steps_taken >= self.max_steps:
            terminated = True

            optimal = set(self.current_scenario['optimal_actions'])
            used = self.tools_used

            if len(optimal) == 0:
                quality = 1.0 if action == 0 else 0.0
            else:
                relevant = used & optimal
                irrelevant = used - optimal
                precision = len(relevant) / len(used) if used else 0
                recall = len(relevant) / len(optimal) if optimal else 0
                quality = (precision + recall) / 2
                quality -= 0.1 * len(irrelevant)

            if self.level >= 6:
                if self.current_scenario.get('is_repeat', False) and not self.accessed_memory:
                    quality -= 0.2

            reward += max(0, quality)

            if self.level >= 6:
                outcome = 'success' if quality > 0.5 else 'failure'
                self.memory_system.record_interaction(
                    self.customer_id, self.current_scenario['id'],
                    list(self.action_history), outcome)

        obs = self._get_observation()
        return obs, reward, terminated, False, {}


def evaluate_rl_model(model, env, num_episodes=100):
    """Evaluate a trained RL model on the email triage environment.

    Args:
        model: Trained SB3 model with .predict() method.
        env: EmailTriageEnvProgressive instance.
        num_episodes: Number of evaluation episodes.

    Returns:
        Dict with mean_reward, success_rate, mean_tools, memory_usage,
        memory_on_repeats.
    """
    results = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        terminated = False
        accessed_memory = False
        is_repeat = env.current_scenario.get('is_repeat', False)

        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            if int(action) == 9:
                accessed_memory = True
            obs, reward, terminated, truncated, _ = env.step(int(action))
            episode_reward += reward

        results.append({
            'reward': episode_reward,
            'tools_used': len(env.tools_used),
            'accessed_memory': accessed_memory,
            'is_repeat': is_repeat,
        })

    repeat_scenarios = [r for r in results if r['is_repeat']]
    memory_on_repeats = (sum(1 for r in repeat_scenarios if r['accessed_memory'])
                         if repeat_scenarios else 0)

    return {
        'mean_reward': np.mean([r['reward'] for r in results]),
        'success_rate': np.mean([r['reward'] > 0.5 for r in results]) * 100,
        'mean_tools': np.mean([r['tools_used'] for r in results]),
        'memory_usage': sum(1 for r in results if r['accessed_memory']) / len(results) * 100,
        'memory_on_repeats': (memory_on_repeats / len(repeat_scenarios) * 100
                              if repeat_scenarios else 0),
    }
