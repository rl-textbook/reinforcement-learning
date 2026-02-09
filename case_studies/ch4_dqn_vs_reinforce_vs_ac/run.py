"""DQN vs REINFORCE vs Actor-Critic on CartPole."""

from rlbook.environments import CartPoleWrapper
from rlbook.algorithms.deep.runners import RunConfig, run_reinforce, run_actor_critic, run_dqn
from rlbook.algorithms.deep.dqn import DQNConfig

env = CartPoleWrapper(seed=0)
cfg = RunConfig(
    gamma=0.99,
    episodes=10_000,
    steps_per_episode=1000,
    eval_every_steps=10_000,
    eval_episodes=5,
    total_step_budget=100_000,
)

import numpy as np

SEEDS = [101, 202, 303]

all_final = {}

print("Running REINFORCE...")
for seed in SEEDS:
    logs = run_reinforce(seed=seed, env=env, cfg=cfg, lr=1e-3)
    all_final.setdefault("REINFORCE", []).append(logs[-1]["return_mean"])

print("Running Actor-Critic...")
for seed in SEEDS:
    logs = run_actor_critic(seed=seed, env=env, cfg=cfg, actor_lr=3e-4, critic_lr=1e-3)
    all_final.setdefault("ActorCritic", []).append(logs[-1]["return_mean"])

print("Running DQN...")
dqn_cfg = DQNConfig(double_dqn=False)
for seed in SEEDS:
    logs = run_dqn(seed=seed, env=env, run_cfg=cfg, dqn_cfg=dqn_cfg)
    all_final.setdefault("DQN", []).append(logs[-1]["return_mean"])

print()
for algo, returns in all_final.items():
    print(f"{algo:15s} | mean: {np.mean(returns):6.1f}  std: {np.std(returns):5.1f}  seeds: {returns}")
