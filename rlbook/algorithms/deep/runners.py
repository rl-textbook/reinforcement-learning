"""Experiment runners for Chapter 4 deep RL algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rlbook.environments.cartpole_wrapper import CartPoleWrapper, set_global_seed
from rlbook.algorithms.deep.reinforce import (
    PolicyNet,
    sample_action_and_logp,
    evaluate,
    compute_returns,
    reinforce_update,
)
from rlbook.algorithms.deep.actor_critic import Critic, actor_critic_update
from rlbook.algorithms.deep.dqn import DQNAgent, DQNConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class RunConfig:
    gamma: float = 0.99
    episodes: int = 500
    steps_per_episode: int = 1_000
    eval_every_steps: int = 5_000
    eval_episodes: int = 10
    total_step_budget: int = 200_000


def _make_eval_env(env: CartPoleWrapper, seed: int) -> CartPoleWrapper:
    env_cls = type(env)
    try:
        return env_cls(seed=seed)
    except TypeError:
        return CartPoleWrapper(seed=seed)


def _iqr(values) -> float:
    arr = np.asarray(values, dtype=np.float32)
    q25, q75 = np.percentile(arr, [25, 75])
    return float(q75 - q25)


def _eval_seed_base(train_seed: int, step_label: int) -> int:
    return 10_000 * (int(train_seed) + 1) + int(step_label)


def run_reinforce(seed: int, env: CartPoleWrapper, cfg: RunConfig, lr: float = 1e-3):
    set_global_seed(seed)
    eval_env = _make_eval_env(env, seed=seed + 50_000)
    policy = PolicyNet(env.state_dim, env.n_actions).to(DEVICE)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    logs, total_steps, baseline_mean = [], 0, None
    next_eval_step = cfg.eval_every_steps

    def greedy_fn(s):
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            p = policy(x)[0]
            return int(torch.argmax(p).item())

    def record_eval(step_label: int, loss_value: float):
        rets = evaluate(
            eval_env, greedy_fn, episodes=cfg.eval_episodes,
            seed_base=_eval_seed_base(seed, step_label),
        )
        logs.append({
            "algo": "REINFORCE",
            "seed": seed,
            "steps": int(step_label),
            "return_mean": float(np.mean(rets)),
            "return_iqr": _iqr(rets),
            "loss": float(loss_value),
        })

    for ep in range(cfg.episodes):
        s = env.reset(seed=seed + ep)
        rewards, logps = [], []
        for _ in range(cfg.steps_per_episode):
            a, logp = sample_action_and_logp(policy, s, env.n_actions)
            s_next, r, done = env.step(a)
            rewards.append(r)
            logps.append(logp)
            s = s_next
            total_steps += 1
            if done or total_steps >= cfg.total_step_budget:
                break

        G = compute_returns(rewards, gamma=cfg.gamma)
        mean_G = float(np.mean(G))
        baseline_mean = mean_G if baseline_mean is None else 0.9 * baseline_mean + 0.1 * mean_G
        loss = reinforce_update(policy, opt, logps, G, baseline_mean=baseline_mean)

        if total_steps >= next_eval_step:
            record_eval(next_eval_step, loss)
            next_eval_step += cfg.eval_every_steps
        if total_steps >= cfg.total_step_budget:
            break

    if logs and logs[-1]["steps"] != min(total_steps, cfg.total_step_budget):
        record_eval(min(total_steps, cfg.total_step_budget), loss)
    elif not logs:
        record_eval(min(total_steps, cfg.total_step_budget), loss if 'loss' in locals() else 0.0)

    eval_env.close()
    return logs


def run_actor_critic(
    seed: int,
    env: CartPoleWrapper,
    cfg: RunConfig,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-3,
):
    set_global_seed(seed)
    eval_env = _make_eval_env(env, seed=seed + 60_000)
    actor = PolicyNet(env.state_dim, env.n_actions).to(DEVICE)
    critic = Critic(env.state_dim).to(DEVICE)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=critic_lr)

    logs, total_steps = [], 0
    next_eval_step = cfg.eval_every_steps
    last_actor_loss = last_critic_loss = last_delta = 0.0

    def greedy_fn(s):
        with torch.no_grad():
            x = torch.as_tensor(s, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            p = actor(x)[0]
            return int(torch.argmax(p).item())

    def record_eval(step_label: int):
        rets = evaluate(
            eval_env, greedy_fn, episodes=cfg.eval_episodes,
            seed_base=_eval_seed_base(seed, step_label),
        )
        logs.append({
            "algo": "ActorCritic",
            "seed": seed,
            "steps": int(step_label),
            "return_mean": float(np.mean(rets)),
            "return_iqr": _iqr(rets),
            "actor_loss": float(last_actor_loss),
            "critic_loss": float(last_critic_loss),
            "delta": float(last_delta),
        })

    for ep in range(cfg.episodes):
        s = env.reset(seed=seed + ep)
        for _ in range(cfg.steps_per_episode):
            # Reuse the REINFORCE actor and sampler
            a, logp = sample_action_and_logp(actor, s, env.n_actions)
            s_next, r, done = env.step(a)
            
            # One TD (0) critic update + one actor update weighted by delta_t
            last_actor_loss, last_critic_loss, last_delta = actor_critic_update(
                actor, critic, opt_actor, opt_critic,
                s, a, r, s_next, done, logp, cfg.gamma,
            )
            s = s_next
            total_steps += 1

            if total_steps >= next_eval_step:
                record_eval(next_eval_step)
                next_eval_step += cfg.eval_every_steps
            if done or total_steps >= cfg.total_step_budget:
                break
        if total_steps >= cfg.total_step_budget:
            break

    if logs and logs[-1]["steps"] != min(total_steps, cfg.total_step_budget):
        record_eval(min(total_steps, cfg.total_step_budget))
    elif not logs:
        record_eval(min(total_steps, cfg.total_step_budget))

    eval_env.close()
    return logs


def run_dqn(seed: int, env: CartPoleWrapper, run_cfg: RunConfig, dqn_cfg: DQNConfig):
    set_global_seed(seed)
    eval_env = _make_eval_env(env, seed=seed + 70_000)
    agent = DQNAgent(env.state_dim, env.n_actions, dqn_cfg)
    logs, total_steps = [], 0
    warmup = max(dqn_cfg.batch_size, 1_000)
    name = "DoubleDQN" if dqn_cfg.double_dqn else "DQN"
    next_eval_step = run_cfg.eval_every_steps

    def greedy_fn(s):
        return agent.act(s, eval_mode=True)

    def record_eval(step_label: int):
        rets = evaluate(
            eval_env, greedy_fn, episodes=run_cfg.eval_episodes,
            seed_base=_eval_seed_base(seed, step_label),
        )
        logs.append({
            "algo": name,
            "seed": seed,
            "steps": int(step_label),
            "return_mean": float(np.mean(rets)),
            "return_iqr": _iqr(rets),
        })

    for ep in range(run_cfg.episodes):
        s = env.reset(seed=seed + ep)
        for _ in range(run_cfg.steps_per_episode):
            a = agent.act(s, eval_mode=False)
            s_next, r, done = env.step(a)
            agent.push(s, a, r, s_next, done)
            s = s_next
            total_steps += 1

            if total_steps > warmup:
                _ = agent.update()

            if total_steps >= next_eval_step:
                record_eval(next_eval_step)
                next_eval_step += run_cfg.eval_every_steps
            if done or total_steps >= run_cfg.total_step_budget:
                break
        if total_steps >= run_cfg.total_step_budget:
            break

    if logs and logs[-1]["steps"] != min(total_steps, run_cfg.total_step_budget):
        record_eval(min(total_steps, run_cfg.total_step_budget))
    elif not logs:
        record_eval(min(total_steps, run_cfg.total_step_budget))

    eval_env.close()
    return logs
