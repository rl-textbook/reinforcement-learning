"""
CartPole Environment Wrapper for Chapter 4 case studies.

Provides a simplified interface to the Gymnasium CartPole-v1 environment
with reproducible seeding. See Chapter 4 (Deep Reinforcement Learning)
for the complete description and case studies.
"""

import random
import numpy as np
import gymnasium as gym
import torch


class CartPoleWrapper:
    """
    Contract:
      s = env.reset(seed=seed)         -> np.ndarray shape (D,)
      s_next, r, done = env.step(a)    -> np.ndarray, float, bool
      env.state_dim, env.n_actions     -> ints
    """
    def __init__(self, seed: int = 0):
        self.env = gym.make("CartPole-v1")
        self._seed = seed
        obs_space = self.env.observation_space
        act_space = self.env.action_space
        self.state_dim = int(obs_space.shape[0])  # D
        self.n_actions = int(act_space.n)         # A

    def reset(self, seed: int = None):
        if seed is None:
            seed = self._seed
        out = self.env.reset(seed=seed)
        s = out[0] if isinstance(out, tuple) else out
        return np.asarray(s, dtype=np.float32)

    def step(self, a: int):
        s_next, r, terminated, truncated, _ = self.env.step(int(a))
        done = bool(terminated or truncated)
        return (np.asarray(s_next, dtype=np.float32), float(r), done)

# --- Seeding helper to make experiments reproducible ---
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
