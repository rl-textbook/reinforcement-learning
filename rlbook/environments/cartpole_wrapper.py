"""CartPole environment wrapper for Chapter 4 deep RL case studies."""

from __future__ import annotations

import random

import gymnasium as gym
import numpy as np
import torch


class CartPoleWrapper:
    """Small convenience wrapper around Gymnasium's CartPole-v1 environment.

    Contract:
      s = env.reset(seed=seed)         -> np.ndarray shape (D,)
      s_next, r, done = env.step(a)    -> np.ndarray, float, bool
      env.state_dim / env.obs_dim      -> int
      env.n_actions                    -> int

    Notes
    -----
    The constructor seed is used for the *first* reset only. Subsequent
    ``reset()`` calls without an explicit seed do not re-seed the environment,
    which avoids accidentally replaying the same evaluation episode over and
    over again.
    """

    def __init__(self, seed: int = 0):
        self.env = gym.make("CartPole-v1")
        self._pending_seed = int(seed)
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.state_dim = int(obs_space.shape[0])
        self.obs_dim = self.state_dim  # backwards-compatible alias
        self.n_actions = int(act_space.n)

    def reset(self, seed: int | None = None):
        actual_seed = self._pending_seed if seed is None else int(seed)
        if actual_seed is None:
            out = self.env.reset()
        else:
            out = self.env.reset(seed=actual_seed)
            self.env.action_space.seed(actual_seed)
        self._pending_seed = None
        s = out[0] if isinstance(out, tuple) else out
        return np.asarray(s, dtype=np.float32)

    def step(self, a: int):
        s_next, r, terminated, truncated, _ = self.env.step(int(a))
        done = bool(terminated or truncated)
        return np.asarray(s_next, dtype=np.float32), float(r), done

    def close(self):
        self.env.close()


def set_global_seed(seed: int):
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
