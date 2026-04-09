"""Deep Q-Network (DQN) with optional Double DQN."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlbook.core.replay_buffer import ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QNet(nn.Module):
    def __init__(self, in_dim: int, n_actions: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, n_actions)] # one output per action
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) state batch -> Q(s, a) for all actions: (B, A)
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 2e-4
    replay_capacity: int = 100_000
    batch_size: int = 64
    target_sync_every: int = 1_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    double_dqn: bool = False


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: DQNConfig):
        self.n_actions = int(n_actions)
        self.cfg = cfg
        self.online = QNet(state_dim, n_actions).to(DEVICE)
        self.target = QNet(state_dim, n_actions).to(DEVICE)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_capacity)
        self.env_steps = 0
        self.update_steps = 0

    def epsilon(self):
        # Linearly interpolate epsilon from eps_start to eps_end over eps_decay_steps
        c = self.cfg
        if c.eps_decay_steps <= 0:
            return c.eps_end
        frac = min(1.0, self.update_steps / float(c.eps_decay_steps))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def act(self, s, eval_mode: bool = False):
        # Training : collect off - policy data with epsilon - greedy exploration
        if (not eval_mode) and (random.random() < self.epsilon()):
            action = random.randrange(self.n_actions)
        else:
            # Evaluation : take the greedy action argmax_a Q_theta (s , a )
            with torch.no_grad():
                x = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                q = self.online(x) # shape (1, A)
                action = int(q.argmax(dim=1).item())
        if not eval_mode:
            self.env_steps += 1
        return action

    def push(self, s, a, r, s_next, done):
        self.replay.push(s, a, r, s_next, done)

    def update(self):
        c = self.cfg
        if len(self.replay) < c.batch_size:
            return None # wait until replay has enough data for a minibatch

        s, a, r, s_next, d = self.replay.sample(c.batch_size)
        
        # Tensorize the sampled transitions
        S = torch.as_tensor(s, dtype=torch.float32, device=DEVICE)
        A = torch.as_tensor(a, dtype=torch.int64, device=DEVICE).unsqueeze(1)
        R = torch.as_tensor(r, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        S_next = torch.as_tensor(s_next, dtype=torch.float32, device=DEVICE)
        Dmask = torch.as_tensor(d, dtype=torch.float32, device=DEVICE).unsqueeze(1)

        # Left-hand side : current estimate Q_theta (S_t, A_t)
        q_sa = self.online(S).gather(1, A)
        
        # Right - hand side : fixed bootstrap target from Q_theta_minus
        with torch.no_grad():
            if c.double_dqn:
                a_star = self.online(S_next).argmax(dim=1, keepdim=True)
                q_next = self.target(S_next).gather(1, a_star)
            else:
                q_next = self.target(S_next).max(dim=1, keepdim=True).values
            y = R + c.gamma * (1.0 - Dmask) * q_next

        # Semi - gradient TD regression with a robust loss
        loss = F.smooth_l1_loss(q_sa, y)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()

        # Refresh the lagged target network every C step
        self.update_steps += 1
        if self.update_steps % c.target_sync_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
