"""Instrumented versions of VI and PI that record convergence traces."""

import random
from typing import Dict, List, Tuple, Optional

import numpy as np

from rlbook.core.mdp import Environment
from rlbook.algorithms.dynamic_programming import (
    extract_policy, policy_evaluation
)


def value_iteration_trace(env: Environment, theta: float = 0.001,
                          max_iterations: int = 100):
    """Value iteration with per-sweep delta and policy recording.

    Returns:
        (deltas, policies) — deltas[i] is the max Bellman error at sweep i;
        policies[i] is the greedy policy *before* sweep i (plus the final one).
    """
    V = {s.id: 0.0 for s in env.states}
    deltas = []
    policies = []

    for i in range(max_iterations):
        policies.append(extract_policy(env, V))
        delta = 0.0
        for state in env.states:
            if state.terminal:
                continue
            v_old = V[state.id]
            max_value = float('-inf')
            for action in env.actions:
                if (state.id, action) not in env.transitions:
                    continue
                action_value = 0.0
                for nxt, p in env.transitions[(state.id, action)].items():
                    r = env.rewards.get((state.id, action, nxt.id), 0)
                    action_value += p * (r + env.gamma * V[nxt.id])
                max_value = max(max_value, action_value)
            V[state.id] = max_value
            delta = max(delta, abs(v_old - max_value))
        deltas.append(delta)
        if delta < theta:
            policies.append(extract_policy(env, V))
            break

    return deltas, policies


def policy_iteration_trace(env: Environment, theta: float = 0.001,
                           max_iterations: int = 1000):
    """Policy iteration with per-step change count and policy recording.

    Returns:
        (policy_changes, policies) — policy_changes[i] is the number of
        states whose action changed at improvement step i;
        policies[i] is the policy before step i.
    """
    policy = {s.id: random.choice(env.actions)
              for s in env.states if not s.terminal}
    policy_changes = []
    policies = [policy.copy()]

    for i in range(max_iterations):
        V = policy_evaluation(env, policy, theta)
        changes = 0
        for state in env.states:
            if state.terminal:
                continue
            old_action = policy[state.id]
            best_action = None
            best_value = float('-inf')
            for action in env.actions:
                val = 0.0
                for nxt, p in env.transitions.get((state.id, action), {}).items():
                    r = env.rewards.get((state.id, action, nxt.id), 0)
                    val += p * (r + env.gamma * V[nxt.id])
                if val > best_value:
                    best_value = val
                    best_action = action
            policy[state.id] = best_action
            if best_action != old_action:
                changes += 1
        policy_changes.append(changes)
        policies.append(policy.copy())
        if changes == 0:
            break

    return policy_changes, policies


def run_value_iter_steps(env: Environment, k: int, theta: float = 1e-3) -> Dict:
    """Run k sweeps of value iteration and return the resulting value function."""
    V = {s.id: 0.0 for s in env.states}
    for _ in range(k):
        new_V = V.copy()
        for state in env.states:
            if state.terminal:
                continue
            best_val = float('-inf')
            for action in env.actions:
                val = sum(
                    prob * (env.rewards.get((state.id, action, ns.id), 0) + env.gamma * V[ns.id])
                    for ns, prob in env.transitions.get((state.id, action), {}).items()
                )
                best_val = max(best_val, val)
            new_V[state.id] = best_val
        V = new_V
    return V


def run_policy_iter_steps(env: Environment, k: int, theta: float = 1e-3) -> Dict:
    """Run k steps of policy iteration and return the resulting policy."""
    policy = {s.id: np.random.choice(env.actions)
              for s in env.states if not s.terminal}
    for _ in range(k):
        V = policy_evaluation(env, policy, theta=theta)
        stable = True
        for state in env.states:
            if state.terminal:
                policy[state.id] = None
                continue
            old_a = policy[state.id]
            best_a, best_val = None, float('-inf')
            for action in env.actions:
                val = sum(
                    prob * (env.rewards.get((state.id, action, ns.id), 0) + env.gamma * V[ns.id])
                    for ns, prob in env.transitions.get((state.id, action), {}).items()
                )
                if val > best_val:
                    best_val, best_a = val, action
            policy[state.id] = best_a
            if best_a != old_a:
                stable = False
        if stable:
            break
    return policy
