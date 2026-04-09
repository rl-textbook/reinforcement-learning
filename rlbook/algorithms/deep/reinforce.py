"""REINFORCE (Monte Carlo policy gradient) helpers."""

from __future__ import annotations

from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from rlbook.environments.cartpole_wrapper import CartPoleWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_actions)] # logits for each discrete action
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> pi_theta(.| s): (B, A)
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


def sample_action_and_logp(policy: PolicyNet, s, n_actions: int | None = None):
    """Sample an action and return its differentiable log-probability."""
    s_t = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    probs = policy(s_t).squeeze(0) # probabilities used for sampling
    dist = Categorical(probs=probs)
    a = int(dist.sample().item())
    logp = dist.log_prob(torch.tensor(a, device=DEVICE, dtype=torch.int64))
    return a, logp


def evaluate(env: CartPoleWrapper, policy_fn: Callable, episodes: int = 10, seed_base: int | None = None):
    """Evaluate a policy on a *separate* environment instance.

    When ``seed_base`` is provided, evaluation episodes are reproducible and
    distinct from one another.
    """
    returns = []
    for ep in range(int(episodes)):
        reset_seed = None if seed_base is None else int(seed_base + ep)
        s = env.reset(seed=reset_seed)
        total = 0.0
        for _ in range(10_000):
            a = policy_fn(s)
            s, r, done = env.step(a)
            total += r
            if done:
                break
        returns.append(total)
    return np.asarray(returns, dtype=np.float32)


def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    G, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g # G_t = R_{t+1} + gamma G_{t+1}
        G.append(g)
    G.reverse()
    return G # aligned with t = 0, 1, ..., T -1


def reinforce_update(
    policy: PolicyNet,
    optimizer,
    logps: List[torch.Tensor],
    returns: List[float],
    baseline_mean: float | None = None, # baseline : scalar EMA of episode - return means
):
    """Performing a gradient step. Slightly different from the code listing, we computed returns before calling the update. 
    We input the returns as a parameter in this function, hence we don't need to recompute.
   """
    logp_t = torch.stack(logps) # [ log pi_theta ( A_t | S_t ) ] _t
    G_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    advantage = G_t - baseline_mean if baseline_mean is not None else G_t
    
    # Maximize expected return by minimizing the negative objective
    loss = -(logp_t * advantage.detach()).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())
