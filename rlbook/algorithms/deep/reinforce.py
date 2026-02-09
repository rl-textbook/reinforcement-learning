"""
REINFORCE (Monte Carlo Policy Gradient).

Learns a parameterized policy directly using episodic returns
as the gradient signal. See Chapter 4, Algorithm 4.2 for pseudocode.
"""

import random
from typing import List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlbook.environments.cartpole_wrapper import CartPoleWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_actions)]  # logits for A actions
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> probs: (B, A)   (rows sum to 1)
        logits = self.net(x)
        return F.softmax(logits, dim=-1)


# --- Sampler: pick action and compute log pi(a|s) for policy methods ---
def sample_action_and_logp(policy: PolicyNet, s, n_actions: int):
    # Make a batch of size 1 so the network sees shape (1, D)
    s_t = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1, D)
    with torch.no_grad():
        probs = policy(s_t)  # (1, A)
    p = probs.squeeze(0).detach().cpu().numpy()
    a = int(random.choices(range(n_actions), weights=p, k=1)[0])
    # Recompute with grad to get log-prob scalar
    probs = policy(s_t)  # (1, A) requires grad
    eps = 1e-8
    logp = torch.log(probs[0, a].clamp_min(eps))
    return a, logp


# --- Evaluation helper: run N episodes with a given policy/agent ---
def evaluate(env: CartPoleWrapper, policy_fn, episodes=10):
    # policy_fn: callable s -> a (greedy/near-greedy)
    returns = []
    for _ in range(episodes):
        s = env.reset()
        total = 0.0
        for _ in range(10000):
            a = policy_fn(s)
            s, r, done = env.step(a)
            total += r
            if done:
                break
        returns.append(total)
    return np.array(returns, dtype=np.float32)


# --- Returns for REINFORCE ---
def compute_returns(rewards: List[float], gamma: float) -> List[float]:
    G, g = [], 0.0
    for r in reversed(rewards):
        g = r + gamma * g
        G.append(g)
    G.reverse()
    return G

# --- REINFORCE step (per episode) ---
def reinforce_update(policy: PolicyNet, optimizer, logps: List[torch.Tensor], returns: List[float],
                     baseline_mean: float = None):
    logp_t = torch.stack(logps)  # (T,)
    G_t = torch.tensor(returns, dtype=torch.float32, device=DEVICE)
    if baseline_mean is not None:
        advantage = G_t - baseline_mean
    else:
        advantage = G_t
    loss = -(logp_t * advantage.detach()).mean()  # minimize -J
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())
