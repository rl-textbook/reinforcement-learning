"""
Experiment Runners for Deep RL Algorithms.

Training loops for REINFORCE, Actor-Critic, and DQN with
step-wise logging and periodic evaluation.
See Chapter 4 case studies for usage.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from rlbook.environments.cartpole_wrapper import CartPoleWrapper, set_global_seed
from rlbook.algorithms.deep.reinforce import (
    PolicyNet, sample_action_and_logp, evaluate,
    compute_returns, reinforce_update
)
from rlbook.algorithms.deep.actor_critic import Critic, actor_critic_update
from rlbook.algorithms.deep.dqn import DQNAgent, DQNConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Runners (each returns logs with step-wise training stats and periodic eval) ---
@dataclass
class RunConfig:
    gamma: float = 0.99
    episodes: int = 500
    steps_per_episode: int = 1000
    eval_every_steps: int = 5_000
    eval_episodes: int = 10
    total_step_budget: int = 200_000

def run_reinforce(seed: int, env: CartPoleWrapper, cfg: RunConfig, lr: float = 1e-3):
    set_global_seed(seed)
    policy = PolicyNet(env.state_dim, env.n_actions).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    logs, total_steps, baseline_mean = [], 0, None

    # Greedy policy for evaluation (argmax over probs)
    def greedy_fn(s):
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            p = policy(x)[0]
            return int(torch.argmax(p).item())

    for ep in range(cfg.episodes):
        s = env.reset(seed=seed + ep)
        rewards, logps = [], []
        for _ in range(cfg.steps_per_episode):
            a, logp = sample_action_and_logp(policy, s, env.n_actions)
            s_next, r, done = env.step(a)
            rewards.append(r); logps.append(logp)
            s = s_next; total_steps += 1
            if done or total_steps >= cfg.total_step_budget:
                break

        G = compute_returns(rewards, gamma=cfg.gamma)
        baseline_mean = 0.9 * (baseline_mean if baseline_mean is not None else np.mean(G)) + 0.1 * np.mean(G)
        loss = reinforce_update(policy, opt, logps, G, baseline_mean=baseline_mean)

        # evaluation cadence
        if total_steps % cfg.eval_every_steps == 0 or total_steps >= cfg.total_step_budget:
            rets = evaluate(env, greedy_fn, episodes=cfg.eval_episodes)
            logs.append({"algo": "REINFORCE", "seed": seed, "steps": total_steps,
                         "return_mean": float(np.mean(rets)), "return_iqr": float(np.subtract(*np.percentile(rets, [75,25]))),
                         "loss": loss})
        if total_steps >= cfg.total_step_budget:
            break

    return logs

def run_actor_critic(seed: int, env: CartPoleWrapper, cfg: RunConfig,
                     actor_lr: float = 3e-4, critic_lr: float = 1e-3):
    set_global_seed(seed)
    actor  = PolicyNet(env.state_dim, env.n_actions).to(DEVICE)
    critic = Critic(env.state_dim).to(DEVICE)
    opt_actor  = torch.optim.Adam(actor.parameters(),  lr=actor_lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    logs, total_steps = [], 0

    def greedy_fn(s):
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            p = actor(x)[0]
            return int(torch.argmax(p).item())

    for ep in range(cfg.episodes):
        s = env.reset(seed=seed + ep)
        for _ in range(cfg.steps_per_episode):
            a, logp = sample_action_and_logp(actor, s, env.n_actions)
            s_next, r, done = env.step(a)
            a_loss, c_loss, delta = actor_critic_update(actor, critic, opt_actor, opt_critic,
                                                        s, a, r, s_next, done, logp, cfg.gamma)
            s = s_next; total_steps += 1

            if total_steps % cfg.eval_every_steps == 0 or total_steps >= cfg.total_step_budget or done:
                rets = evaluate(env, greedy_fn, episodes=cfg.eval_episodes)
                logs.append({"algo": "ActorCritic", "seed": seed, "steps": total_steps,
                             "return_mean": float(np.mean(rets)), "return_iqr": float(np.subtract(*np.percentile(rets, [75,25]))),
                             "actor_loss": a_loss, "critic_loss": c_loss, "delta": delta})
            if done or total_steps >= cfg.total_step_budget:
                break
        if total_steps >= cfg.total_step_budget:
            break

    return logs

def run_dqn(seed: int, env: CartPoleWrapper, run_cfg: RunConfig, dqn_cfg: DQNConfig):
    set_global_seed(seed)
    agent = DQNAgent(env.state_dim, env.n_actions, dqn_cfg)
    logs, total_steps = [], 0
    warmup = 1000
    name = "DoubleDQN" if dqn_cfg.double_dqn else "DQN"

    # ✅ evaluate greedily (no exploration)
    def near_greedy(s):
        return agent.act(s, eval_mode=True)

    for ep in range(run_cfg.episodes):
        s = env.reset(seed=seed + ep)
        for _ in range(run_cfg.steps_per_episode):
            a = agent.act(s, eval_mode=False)              # training uses ε-greedy
            s_next, r, done = env.step(a)
            agent.push(s, a, r, s_next, done)
            s = s_next
            total_steps += 1

            if total_steps > warmup:
                _ = agent.update()

            if (total_steps % run_cfg.eval_every_steps == 0) or done or (total_steps >= run_cfg.total_step_budget):
                rets = evaluate(env, near_greedy, episodes=run_cfg.eval_episodes)  # ✅ use near_greedy
                logs.append({"algo": name, "seed": seed, "steps": total_steps,
                             "return_mean": float(np.mean(rets)),
                             "return_iqr": float(np.subtract(*np.percentile(rets, [75, 25])))})
            if done or total_steps >= run_cfg.total_step_budget:
                break
        if total_steps >= run_cfg.total_step_budget:
            break
    return logs
