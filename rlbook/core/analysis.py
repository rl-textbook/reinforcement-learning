"""
Analysis utilities for Chapter 3: RMSE, visitation counts, and value trajectories.
"""

import math
from typing import Dict, List, Tuple

import numpy as np


def rmse(V_est: Dict[int, float], V_ref: Dict[int, float]) -> float:
    """
    Root mean squared error between two value functions.
    """
    keys = V_ref.keys()
    return math.sqrt(np.mean([(V_est[key] - V_ref[key]) ** 2 for key in keys]))


class VisitationCounter:
    """
    Counts state visits across episodes.
    """

    def __init__(self, nS: int):
        self.counts = np.zeros(nS, dtype=np.int64)
        self.episodes = 0

    def step(self, state_id: int) -> None:
        self.counts[state_id] += 1

    def end_episode(self) -> None:
        self.episodes += 1

    def as_grid(self, env, mode: str = "freq", eps: float = 1e-9):
        H, W = env.grid_height, env.grid_width
        x = self.counts.astype(np.float64).copy()

        if mode == "raw":
            pass
        elif mode == "freq":
            x /= max(self.episodes, 1)
        elif mode == "prob":
            x /= max(x.sum(), eps)
        elif mode == "log":
            x = np.log1p(x)
        else:
            raise ValueError(f"Unknown visitation mode: {mode}")

        return x.reshape(H, W)


class ValueTrajectory:
    """
    Stores (episode_index, value_grid) snapshots for plotting and animation.
    """

    def __init__(self, env):
        self.env = env
        self.snaps: List[Tuple[int, np.ndarray]] = []

    def add(self, V: Dict[int, float], episode_idx: int) -> None:
        H, W = self.env.grid_height, self.env.grid_width
        grid = np.zeros((H, W), dtype=float)

        for state_id in range(H * W):
            r, c = divmod(state_id, W)
            grid[r, c] = V.get(state_id, 0.0)

        self.snaps.append((episode_idx, grid.copy()))


def _first_visit_update_from_episode(episode, gamma, running_sum, running_count, V):
    first_visit_index = {}
    for t, (state_id, _, _) in enumerate(episode):
        if state_id not in first_visit_index:
            first_visit_index[state_id] = t

    G = 0.0
    returns = [0.0] * len(episode)
    for t in reversed(range(len(episode))):
        _, _, reward_t = episode[t]
        G = gamma * G + reward_t
        returns[t] = G

    for state_id, t_first in first_visit_index.items():
        running_sum[state_id] += returns[t_first]
        running_count[state_id] += 1
        V[state_id] = running_sum[state_id] / running_count[state_id]


def mc_td_traces(
    policy,
    env,
    total_episodes: int = 500,
    snapshot_every: int = 10,
    alpha_td: float = 0.1,
    gamma: float = 1.0,
    max_steps: int = 500,
):
    """
    Run Monte Carlo and TD(0) under the same fixed policy, recording periodic snapshots.
    """
    from rlbook.algorithms.tabular.monte_carlo import generate_episode

    mc_snaps = []
    td_snaps = []

    V_mc = {state.id: 0.0 for state in env.states}
    ret_sum = {state.id: 0.0 for state in env.states}
    ret_count = {state.id: 0 for state in env.states}

    V_td = {state.id: 0.0 for state in env.states}

    for ep in range(1, total_episodes + 1):
        episode = generate_episode(env, policy, max_steps=max_steps)
        _first_visit_update_from_episode(episode, gamma, ret_sum, ret_count, V_mc)

        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            probs = policy.get(state.id, {})
            if not probs:
                break

            actions, ps = zip(*probs.items())
            action = np.random.choice(actions, p=ps)
            next_state, reward, done = env.step(action)

            V_td[state.id] += alpha_td * (reward + gamma * V_td[next_state.id] - V_td[state.id])

            state = next_state
            steps += 1

        if ep % snapshot_every == 0:
            mc_snaps.append(V_mc.copy())
            td_snaps.append(V_td.copy())

    return mc_snaps, td_snaps


def run_value_evolution(
    env,
    policy,
    total_episodes: int = 2000,
    snapshot_every: int = 50,
    gamma: float = 1.0,
    alpha_td: float = 0.1,
    max_steps: int = 500,
):
    """
    Interleave MC and TD(0), recording ValueTrajectory objects.
    """
    from rlbook.algorithms.tabular.monte_carlo import generate_episode

    V_mc = {state.id: 0.0 for state in env.states}
    ret_sum = {state.id: 0.0 for state in env.states}
    ret_count = {state.id: 0 for state in env.states}
    V_td = {state.id: 0.0 for state in env.states}

    rec_mc = ValueTrajectory(env)
    rec_td = ValueTrajectory(env)

    for ep in range(1, total_episodes + 1):
        episode = generate_episode(env, policy, max_steps=max_steps)
        _first_visit_update_from_episode(episode, gamma, ret_sum, ret_count, V_mc)

        state = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            probs = policy.get(state.id, {})
            if not probs:
                break

            actions, ps = zip(*probs.items())
            action = np.random.choice(actions, p=ps)
            next_state, reward, done = env.step(action)

            V_td[state.id] += alpha_td * (reward + gamma * V_td[next_state.id] - V_td[state.id])

            state = next_state
            steps += 1

        if ep % snapshot_every == 0:
            rec_mc.add(V_mc, ep)
            rec_td.add(V_td, ep)

    return rec_mc, rec_td
