"""Analysis utilities: convergence tracking, visitation counting, RMSE."""

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------
def rmse(V_est: Dict[int, float], V_ref: Dict[int, float]) -> float:
    """Root mean squared error between two value functions."""
    keys = V_ref.keys()
    return math.sqrt(np.mean([(V_est[k] - V_ref[k]) ** 2 for k in keys]))


# ---------------------------------------------------------------------------
# Visitation counter (MC / TD comparison)
# ---------------------------------------------------------------------------
class VisitationCounter:
    """Counts state visits across episodes."""

    def __init__(self, nS: int):
        self.counts = np.zeros(nS, dtype=np.int64)
        self.episodes = 0

    def step(self, state_id: int):
        self.counts[state_id] += 1

    def end_episode(self):
        self.episodes += 1

    def as_grid(self, env, mode="freq", eps=1e-9):
        H, W = env.grid_height, env.grid_width
        x = self.counts.astype(np.float64)
        if mode == "freq":
            x = x / max(self.episodes, 1)
        elif mode == "prob":
            x = x / (x.sum() + eps)
        elif mode == "log":
            x = np.log1p(x)
        return x.reshape(H, W)


# ---------------------------------------------------------------------------
# Value trajectory recorder (for animation / snapshots)
# ---------------------------------------------------------------------------
class ValueTrajectory:
    """Keeps (episode_index, V_grid) snapshots for animation."""

    def __init__(self, env):
        self.env = env
        self.snaps: List[Tuple[int, np.ndarray]] = []

    def add(self, V: Dict[int, float], episode_idx: int):
        H, W = self.env.grid_height, self.env.grid_width
        G = np.zeros((H, W), dtype=float)
        for i in range(H * W):
            r, c = divmod(i, W)
            G[r, c] = V.get(i, 0.0)
        self.snaps.append((episode_idx, G.copy()))


# ---------------------------------------------------------------------------
# Joint MC / TD trace runner (interleaved, same policy)
# ---------------------------------------------------------------------------
def mc_td_traces(policy, env, total_episodes=500, snapshot_every=10,
                 alpha_td=0.1, gamma=1.0, max_steps=500):
    """Run MC and TD(0) simultaneously under the same policy, recording snapshots.

    Args:
        policy: Stochastic policy dict {state_id: {action: prob}}.
        env: Environment.
        total_episodes: How many episodes to run.
        snapshot_every: Save V snapshots every N episodes.
        alpha_td: Fixed learning rate for TD(0).
        gamma: Discount factor.
        max_steps: Max steps per episode.

    Returns:
        (mc_snapshots, td_snapshots) — each a list of V dicts.
    """
    from rlbook.algorithms.tabular.monte_carlo import generate_episode

    mc_snaps, td_snaps = [], []

    V_mc = {s.id: 0.0 for s in env.states}
    ret_sum = {s.id: 0.0 for s in env.states}
    ret_cnt = {s.id: 0 for s in env.states}

    V_td = {s.id: 0.0 for s in env.states}

    for ep in range(1, total_episodes + 1):
        # MC update
        episode = generate_episode(env, policy, max_steps=max_steps)
        G = 0.0
        seen = set()
        for (s, a, r) in reversed(episode):
            G = gamma * G + r
            if s not in seen:
                seen.add(s)
                ret_sum[s] += G
                ret_cnt[s] += 1
                V_mc[s] = ret_sum[s] / ret_cnt[s]

        # TD(0) episode
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            probs = policy.get(state.id, {})
            if not probs:
                break
            acts, ps = zip(*probs.items())
            a = np.random.choice(acts, p=ps)
            nxt, r, done = env.step(a)
            V_td[state.id] += alpha_td * (r + gamma * V_td[nxt.id] - V_td[state.id])
            state = nxt
            steps += 1

        if ep % snapshot_every == 0:
            mc_snaps.append(V_mc.copy())
            td_snaps.append(V_td.copy())

    return mc_snaps, td_snaps


def run_value_evolution(env, policy, total_episodes=2000, snapshot_every=50,
                        gamma=1.0, alpha_td=0.1, max_steps=500):
    """Interleave MC and TD(0), recording ValueTrajectory objects.

    Returns:
        (rec_mc, rec_td) — ValueTrajectory instances.
    """
    from rlbook.algorithms.tabular.monte_carlo import generate_episode

    V_mc = {s.id: 0.0 for s in env.states}
    R_sum = {s.id: 0.0 for s in env.states}
    R_cnt = {s.id: 0 for s in env.states}
    V_td = {s.id: 0.0 for s in env.states}

    rec_mc = ValueTrajectory(env)
    rec_td = ValueTrajectory(env)

    for ep in range(1, total_episodes + 1):
        # MC episode
        episode = generate_episode(env, policy, max_steps=max_steps)
        G = 0.0
        seen = set()
        for t in reversed(range(len(episode))):
            s_t, _, r_t = episode[t]
            G = gamma * G + r_t
            if s_t not in seen:
                seen.add(s_t)
                R_sum[s_t] += G
                R_cnt[s_t] += 1
                V_mc[s_t] = R_sum[s_t] / R_cnt[s_t]

        # TD(0) episode
        s = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            probs = policy.get(s.id, {})
            if not probs:
                break
            acts, ps = zip(*probs.items())
            a = np.random.choice(acts, p=ps)
            s2, r, done = env.step(a)
            V_td[s.id] += alpha_td * (r + gamma * V_td[s2.id] - V_td[s.id])
            s = s2
            steps += 1

        if ep % snapshot_every == 0:
            rec_mc.add(V_mc, ep)
            rec_td.add(V_td, ep)

    return rec_mc, rec_td
