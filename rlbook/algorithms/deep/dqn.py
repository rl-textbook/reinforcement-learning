"""
Deep Q-Network (DQN) with optional Double DQN.

Uses experience replay and a target network for stable training.
See Chapter 4, Algorithm 4.1 for pseudocode.
"""

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rlbook.core.replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- QNet for DQN ---
class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_actions)]  # Q-values for A actions
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 2e-4
    replay_capacity: int = 100_000
    batch_size: int = 64
    target_sync_every: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    double_dqn: bool = True

class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: DQNConfig):
        self.n_actions = n_actions
        self.cfg = cfg
        self.online = QNet(state_dim, n_actions).to(DEVICE)
        self.target = QNet(state_dim, n_actions).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_capacity)
        self.steps = 0

    def epsilon(self):
        c = self.cfg
        frac = max(0.0, min(1.0, 1.0 - self.steps / float(c.eps_decay_steps)))
        return c.eps_end + (c.eps_start - c.eps_end) * frac

    def act(self, s, eval_mode=False):
        if (not eval_mode) and (random.random() < self.epsilon()):
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            q = self.online(x)  # (1, A)
            return int(q.argmax(dim=1).item())

    def push(self, s, a, r, s_next, done):
        self.replay.push(s, a, r, s_next, done)

    def update(self):
        c = self.cfg
        if len(self.replay) < c.batch_size:
            return None
        s, a, r, s_next, d = self.replay.sample(c.batch_size)
        S      = torch.as_tensor(s,      dtype=torch.float32, device=DEVICE)
        A      = torch.as_tensor(a,      dtype=torch.int64,   device=DEVICE).unsqueeze(1)
        R      = torch.as_tensor(r,      dtype=torch.float32, device=DEVICE).unsqueeze(1)
        S_next = torch.as_tensor(s_next, dtype=torch.float32, device=DEVICE)
        Dmask  = torch.as_tensor(d,      dtype=torch.float32, device=DEVICE).unsqueeze(1)

        q_sa = self.online(S).gather(1, A)  # (B, 1)
        with torch.no_grad():
            if c.double_dqn:
                a_star = self.online(S_next).argmax(dim=1, keepdim=True)   # (B,1)
                q_next = self.target(S_next).gather(1, a_star)             # (B,1)
            else:
                q_next = self.target(S_next).max(dim=1, keepdim=True).values
            y = R + c.gamma * (1.0 - Dmask) * q_next
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.steps += 1
        if self.steps % c.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
