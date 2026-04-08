"""Actor-Critic with one-step TD updates."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlbook.algorithms.deep.reinforce import PolicyNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)] # single scalar V(s)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> values: (B, 1)
        return self.net(x)


def actor_critic_update(
    actor: PolicyNet,
    critic: Critic,
    opt_actor,
    opt_critic,
    s,
    a,
    r,
    s_next,
    done,
    logp_a_s,
    gamma: float,
):
    # One transition (s, a, r, s_next, done) is enough
    S = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    S_next = torch.as_tensor(s_next, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    R = torch.tensor([[float(r)]], dtype=torch.float32, device=DEVICE)
    Dmask = torch.tensor([[1.0 if done else 0.0]], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        # Compute TD target : r + gamma * V(s_next)
        V_next = critic(S_next)
        target = R + gamma * (1.0 - Dmask) * V_next
    
    # Evaluate current value estimate
    V_s = critic(S)
    
    # Critic Update: Minimize (V(s) - target)^2
    critic_loss = F.mse_loss(V_s, target)
    opt_critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    opt_critic.step()

    delta = (target - V_s).detach()

    # Actor Update : Maximize log pi_theta(a|s) * delta
    actor_loss = -(logp_a_s * delta.squeeze())
    opt_actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    opt_actor.step()

    return float(actor_loss.item()), float(critic_loss.item()), float(delta.item())
