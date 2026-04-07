"""Analysis utilities for the Chapter 3 notebook."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from monte_carlo import generate_episode


def rmse(V_est: Dict[int, float], V_ref: Dict[int, float]) -> float:
    """Root mean squared error between two value functions."""
    keys = V_ref.keys()
    return math.sqrt(np.mean([(V_est[k] - V_ref[k]) ** 2 for k in keys]))


class VisitationCounter:
    """Counts state visits across episodes."""

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
            x /= max(1, self.episodes)
        elif mode == "prob":
            total = x.sum()
            x = x / max(total, eps)
        elif mode == "log":
            x = np.log1p(x)
        else:
            raise ValueError(f"Unsupported visitation mode: {mode}")
        return x.reshape(H, W)


class ValueTrajectory:
    """Keeps (episode_index, V_grid) snapshots for animation."""

    def __init__(self, env):
        self.env = env
        self.snaps: List[Tuple[int, np.ndarray]] = []

    def add(self, V: Dict[int, float], episode_idx: int) -> None:
        H, W = self.env.grid_height, self.env.grid_width
        G = np.zeros((H, W), dtype=float)
        for i in range(H * W):
            r, c = divmod(i, W)
            G[r, c] = V.get(i, 0.0)
        self.snaps.append((episode_idx, G.copy()))


def _true_first_visit_update(episode, gamma: float):
    """Return first-visit returns for the sampled episode."""
    first_visit_index: dict[int, int] = {}
    for t, (s_t, _, _) in enumerate(episode):
        if s_t not in first_visit_index:
            first_visit_index[s_t] = t

    G = 0.0
    returns = [0.0] * len(episode)
    for t in reversed(range(len(episode))):
        _, _, r_t = episode[t]
        G = gamma * G + r_t
        returns[t] = G

    return {s_t: returns[t_first] for s_t, t_first in first_visit_index.items()}


def mc_td_traces(
    policy,
    env,
    total_episodes: int = 500,
    snapshot_every: int = 10,
    alpha_td: float = 0.1,
    gamma: float = 1.0,
    max_steps: int = 500,
):
    """Run MC and TD(0) side by side under the same policy."""
    mc_snaps, td_snaps = [], []

    V_mc = {s.id: 0.0 for s in env.states}
    ret_sum = {s.id: 0.0 for s in env.states}
    ret_cnt = {s.id: 0 for s in env.states}
    V_td = {s.id: 0.0 for s in env.states}

    for ep in range(1, total_episodes + 1):
        episode = generate_episode(env, policy, max_steps=max_steps)
        fv_returns = _true_first_visit_update(episode, gamma)
        for s_t, G in fv_returns.items():
            ret_sum[s_t] += G
            ret_cnt[s_t] += 1
            V_mc[s_t] = ret_sum[s_t] / ret_cnt[s_t]

        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            probs = policy.get(state.id, {})
            if not probs:
                break
            acts, ps = zip(*probs.items())
            a = int(np.random.choice(acts, p=ps))
            nxt, r, done = env.step(a)
            V_td[state.id] += alpha_td * (r + gamma * V_td[nxt.id] - V_td[state.id])
            state = nxt
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
    """Interleave MC and TD(0) updates while recording value snapshots."""
    V_mc = {s.id: 0.0 for s in env.states}
    R_sum = {s.id: 0.0 for s in env.states}
    R_cnt = {s.id: 0 for s in env.states}
    V_td = {s.id: 0.0 for s in env.states}

    rec_mc = ValueTrajectory(env)
    rec_td = ValueTrajectory(env)

    for ep in range(1, total_episodes + 1):
        episode = generate_episode(env, policy, max_steps=max_steps)
        fv_returns = _true_first_visit_update(episode, gamma)
        for s_t, G in fv_returns.items():
            R_sum[s_t] += G
            R_cnt[s_t] += 1
            V_mc[s_t] = R_sum[s_t] / R_cnt[s_t]

        s = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            probs = policy.get(s.id, {})
            if not probs:
                break
            acts, ps = zip(*probs.items())
            a = int(np.random.choice(acts, p=ps))
            s2, r, done = env.step(a)
            V_td[s.id] += alpha_td * (r + gamma * V_td[s2.id] - V_td[s.id])
            s = s2
            steps += 1

        if ep % snapshot_every == 0:
            rec_mc.add(V_mc, ep)
            rec_td.add(V_td, ep)

    return rec_mc, rec_td
