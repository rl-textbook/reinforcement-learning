# Algorithm Catalog

A complete index of every algorithm implemented in this repository, mapped to its textbook chapter, Python module, and case study.

---

## Chapter 2: Fundamentals of Reinforcement Learning

| Algorithm | Module | Description |
|-----------|--------|-------------|
| **Value Iteration** | [`rlbook.algorithms.dynamic_programming.value_iteration`](rlbook/algorithms/dynamic_programming/value_iteration.py) | Iterative Bellman optimality backup until convergence |
| **Extract Policy** | [`rlbook.algorithms.dynamic_programming.value_iteration`](rlbook/algorithms/dynamic_programming/value_iteration.py) | Greedy policy extraction from a value function |
| **Policy Evaluation** | [`rlbook.algorithms.dynamic_programming.policy_iteration`](rlbook/algorithms/dynamic_programming/policy_iteration.py) | Iterative policy evaluation (Bellman expectation backup) |
| **Policy Iteration** | [`rlbook.algorithms.dynamic_programming.policy_iteration`](rlbook/algorithms/dynamic_programming/policy_iteration.py) | Alternating evaluation and improvement until convergence |

**Environment:** [Grid World](rlbook/environments/grid_world.py) — Configurable grid with holes, stochastic transitions  
**Case Study:** [Value Iteration vs Policy Iteration](case_studies/ch2_value_iteration_vs_policy_iteration/) — Convergence comparison on deterministic and stochastic grids  
**Notebook:** [Chapter 2](notebooks/chapter2_mdp_foundations.ipynb)

---

## Chapter 3: Classical Reinforcement Learning Algorithms

| Algorithm | Module | Description |
|-----------|--------|-------------|
| **First-Visit MC Prediction** | [`rlbook.algorithms.tabular.monte_carlo`](rlbook/algorithms/tabular/monte_carlo.py) | Value estimation via episode-averaged returns |
| **TD(0) Prediction** | [`rlbook.algorithms.tabular.td_learning`](rlbook/algorithms/tabular/td_learning.py) | Online bootstrapping value estimation |
| **Q-Learning** | [`rlbook.algorithms.tabular.q_learning`](rlbook/algorithms/tabular/q_learning.py) | Off-policy TD control with max operator |
| **SARSA** | [`rlbook.algorithms.tabular.sarsa`](rlbook/algorithms/tabular/sarsa.py) | On-policy TD control |
| **ε-Greedy Policy** | [`rlbook.algorithms.tabular.helpers`](rlbook/algorithms/tabular/helpers.py) | Exploration strategy balancing exploitation and exploration |

**Environment:** [Windy Desert](rlbook/environments/windy_desert.py) — Stochastic grid with quicksand traps and wind drift  
**Case Studies:**
- [MC vs TD Convergence](case_studies/ch3_mc_vs_td_convergence/) — Learning curves, visitation patterns, RMSE comparison  
- [Q-Learning vs SARSA](case_studies/ch3_qlearning_vs_sarsa/) — Off-policy vs on-policy behavior near hazards  

**Notebook:** [Chapter 3](notebooks/chapter3_classical_algorithms.ipynb)

---

## Chapter 4: Deep Reinforcement Learning

| Algorithm | Module | Description |
|-----------|--------|-------------|
| **DQN** | [`rlbook.algorithms.deep.dqn`](rlbook/algorithms/deep/dqn.py) | Deep Q-Network with experience replay and target network |
| **Double DQN** | [`rlbook.algorithms.deep.dqn`](rlbook/algorithms/deep/dqn.py) | DQN with decoupled action selection and evaluation |
| **REINFORCE** | [`rlbook.algorithms.deep.reinforce`](rlbook/algorithms/deep/reinforce.py) | Monte Carlo policy gradient with baseline |
| **Actor-Critic** | [`rlbook.algorithms.deep.actor_critic`](rlbook/algorithms/deep/actor_critic.py) | One-step TD actor-critic with learned baseline |

**Supporting Components:**
- [`PolicyNet`](rlbook/algorithms/deep/reinforce.py) — Parameterized softmax policy network
- [`Critic`](rlbook/algorithms/deep/actor_critic.py) — State-value network
- [`QNet`](rlbook/algorithms/deep/dqn.py) — Action-value network
- [`ReplayBuffer`](rlbook/core/replay_buffer.py) — Circular experience replay buffer
- [`RunConfig` & Runners](rlbook/algorithms/deep/runners.py) — Experiment orchestration

**Environment:** [CartPole Wrapper](rlbook/environments/cartpole_wrapper.py) — Gymnasium CartPole-v1 with reproducible seeding  
**Case Study:** [DQN vs REINFORCE vs Actor-Critic](case_studies/ch4_dqn_vs_reinforce_vs_ac/) — Multi-seed learning curves, sample efficiency, stability  
**Notebook:** [Chapter 4](notebooks/chapter4_deep_rl.ipynb)

---

## Chapter 5: Preference-Based Alignment (RLHF)

| Algorithm | Module | Description |
|-----------|--------|-------------|
| **SFT (Supervised Fine-Tuning)** | [`rlbook.algorithms.alignment.sft`](rlbook/algorithms/alignment/sft.py) | Stage 1: Fine-tune on demonstrations |
| **Reward Model** | [`rlbook.algorithms.alignment.reward_model`](rlbook/algorithms/alignment/reward_model.py) | Stage 2: Learn scalar rewards from preferences |
| **DPO** | [`rlbook.algorithms.alignment.dpo`](rlbook/algorithms/alignment/dpo.py) | Direct Preference Optimization (no RM needed) |
| **PPO for LLMs** | [`rlbook.algorithms.alignment.ppo_alignment`](rlbook/algorithms/alignment/ppo_alignment.py) | Stage 3: PPO with KL penalty and learned RM |
| **GPT-4 Judge** | [`rlbook.algorithms.alignment.gpt4_judge`](rlbook/algorithms/alignment/gpt4_judge.py) | LLM-as-a-Judge for pairwise evaluation |

**Supporting Components:**
- [`Config`](rlbook/algorithms/alignment/config.py) — Hyperparameters for the full RLHF pipeline
- [`sequence_log_probs`](rlbook/algorithms/alignment/config.py) — Sequence-level log probability computation
- [Data processing utilities](rlbook/algorithms/alignment/config.py) — Prompt extraction, preference formatting

**Dataset:** [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)  
**Case Study:** [SFT → DPO → PPO Alignment](case_studies/ch5_sft_dpo_ppo_alignment/) — End-to-end RLHF pipeline with GPT-4 judge evaluation  
**Notebook:** [Chapter 5](notebooks/chapter5_alignment_rlhf.ipynb)

---

## Chapter 6: Reasoning Models & RLVR

*Coming soon* — GRPO, Reinforcement Learning with Verifiable Rewards

---

## Chapter 7: Agentic Architectures in RL

| Component | Module | Description |
|-----------|--------|-------------|
| **Customer Memory System** | [`rlbook.algorithms.agentic.email_triage`](rlbook/algorithms/agentic/email_triage.py) | Persistent cross-episode memory for customer history |
| **Progressive Environment** | [Chapter 7 Notebook](notebooks/chapter7_agentic_email_triage.ipynb) | 6-level environment with RL-learned tool use |
| **Baseline Agents** | [`rlbook.algorithms.agentic.email_triage`](rlbook/algorithms/agentic/email_triage.py) | Random and classification-based baselines |

**Case Study:** [Agentic Email Triage](case_studies/ch7_agentic_email_triage/) — Progressive capability integration: Random → Classification → RL Tool Use → Memory  
**Notebook:** [Chapter 7](notebooks/chapter7_agentic_email_triage.ipynb)

---

## Core Components

| Component | Module | Used By |
|-----------|--------|---------|
| **State** | [`rlbook.core.mdp`](rlbook/core/mdp.py) | Chapters 2, 3 |
| **Environment** | [`rlbook.core.mdp`](rlbook/core/mdp.py) | Chapters 2, 3 |
| **Agent** | [`rlbook.core.mdp`](rlbook/core/mdp.py) | Chapters 2, 3 |
| **ReplayBuffer** | [`rlbook.core.replay_buffer`](rlbook/core/replay_buffer.py) | Chapter 4 |
