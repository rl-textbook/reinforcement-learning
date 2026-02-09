"""
Actor-Critic with one-step TD updates.

Combines a policy network (actor) with a value network (critic)
for lower-variance policy gradient updates.
See Chapter 4, Algorithm 4.3 for pseudocode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlbook.algorithms.deep.reinforce import PolicyNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Critic (scalar V(s)) used by Actor-Critic ---
class Critic(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, 1)]  # scalar V(s)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D) -> values: (B, 1)
        return self.net(x)


# --- Actor-Critic one-step update ---
def actor_critic_update(actor: PolicyNet, critic: Critic,
                        opt_actor, opt_critic,
                        s, a, r, s_next, done, logp_a_s, gamma: float):
    S      = torch.as_tensor(s,      dtype=torch.float32, device=DEVICE).unsqueeze(0)
    S_next = torch.as_tensor(s_next, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    R      = torch.tensor([[float(r)]], dtype=torch.float32, device=DEVICE)
    Dmask  = torch.tensor([[1.0 if done else 0.0]], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        V_next = critic(S_next)                               # (1,1)
        target = R + gamma * (1.0 - Dmask) * V_next           # (1,1)

    V_s = critic(S)                                           # (1,1)
    critic_loss = F.mse_loss(V_s, target)

    opt_critic.zero_grad(set_to_none=True)
    critic_loss.backward()
    opt_critic.step()

    with torch.no_grad():
        delta = target - V_s                                  # (1,1)

    actor_loss = -(logp_a_s * delta.squeeze())               # scalar
    opt_actor.zero_grad(set_to_none=True)
    actor_loss.backward()
    opt_actor.step()

    return float(actor_loss.item()), float(critic_loss.item()), float(delta.item())
