# Notebooks

Interactive Jupyter notebooks for each chapter of **Reinforcement Learning: From Foundations to Frontiers**.

## Architecture

All notebooks import algorithms and utilities from the `rlbook` package rather than defining them inline. This ensures:

- **Single source of truth** — algorithms live in `rlbook/`, notebooks focus on execution and visualization
- **Consistency** — the `run.py` case studies and notebooks call the same code paths
- **Maintainability** — bug fixes in `rlbook/` automatically propagate to every notebook
- **Readability** — notebooks are shorter and more pedagogical

## Quick Start

### Local (recommended)

```bash
# 1. Clone and install
git clone https://github.com/rl-textbook/reinforcement-learning.git
cd reinforcement-learning
python3 -m venv venv && source venv/bin/activate
pip install -e ".[all]"

# 2. Register Jupyter kernel
pip install jupyter ipykernel
python -m ipykernel install --user --name=rlbook --display-name "Python (rlbook)"

# 3. Launch
jupyter notebook notebooks/
```

Select **Kernel → Python (rlbook)** and run cells with **Shift+Enter**.

### Google Colab (zero setup)

Each notebook's first code cell includes a commented install command. On Colab, uncomment it:

```python
# Chapters 2-3 (tabular, no GPU needed):
!pip install -q "git+https://github.com/rl-textbook/reinforcement-learning.git"

# Chapter 4 (deep RL — GPU recommended):
!pip install -q "git+https://github.com/rl-textbook/reinforcement-learning.git#egg=rlbook[deep]"

# Chapter 5 (alignment — GPU recommended):
!pip install -q "git+https://github.com/rl-textbook/reinforcement-learning.git#egg=rlbook[alignment]"

# Chapter 7 (agentic — GPU recommended):
!pip install -q "git+https://github.com/rl-textbook/reinforcement-learning.git#egg=rlbook[agentic]"
```

For Chapters 4, 5, and 7, switch to a GPU runtime: **Runtime → Change runtime type → T4 GPU**.

## Notebook Index

| Notebook | Chapter | Topics | Run Time | GPU? |
|----------|---------|--------|----------|------|
| [`chapter2_mdp_foundations.ipynb`](chapter2_mdp_foundations.ipynb) | 2 | Value Iteration, Policy Iteration, convergence analysis, agent evaluation | ~2 min | No |
| [`chapter3_classical_algorithms.ipynb`](chapter3_classical_algorithms.ipynb) | 3 | MC vs TD(0) prediction, Q-Learning vs SARSA, visitation heatmaps, animated value evolution | ~5 min | No |
| [`chapter4_deep_rl.ipynb`](chapter4_deep_rl.ipynb) | 4 | DQN, REINFORCE, Actor-Critic on CartPole, multi-seed learning curves, AUC comparison | ~10 min CPU / ~3 min GPU | Recommended |
| [`chapter5_alignment_rlhf.ipynb`](chapter5_alignment_rlhf.ipynb) | 5 | SFT → Reward Model → DPO → PPO on GPT-2, Anthropic HH dataset, qualitative evaluation | ~30 min GPU | Recommended |
| [`chapter7_agentic_email_triage.ipynb`](chapter7_agentic_email_triage.ipynb) | 7 | Progressive 6-level agentic RL, tool use, LLM planning, persistent memory | ~15 min | Recommended |

## What Each Notebook Contains

Every notebook follows the same structure:

1. **Setup cell** — `rlbook` imports + install instructions for Local/Colab
2. **Environment setup** — create and configure the environment
3. **Training** — run algorithms via `rlbook` library calls
4. **Visualization** — learning curves, policy maps, path overlays, bar charts
5. **Analysis** — quantitative comparison + qualitative evaluation
6. **Key takeaways** — summary of findings

## Library Imports Used

```python
# Chapter 2: Dynamic Programming
from rlbook.environments import create_grid_world
from rlbook.algorithms.dynamic_programming import value_iteration, policy_iteration, ...
from rlbook.viz import visualize_grid_world, visualize_agent_path

# Chapter 3: Tabular RL
from rlbook.environments import create_windy_desert_env, apply_reward_shaping
from rlbook.algorithms.tabular import q_learning, sarsa, first_visit_mc_prediction, ...
from rlbook.viz import visualize_grid_world_desert, plot_policy_evolution, ...

# Chapter 4: Deep RL
from rlbook.environments import CartPoleWrapper
from rlbook.algorithms.deep import RunConfig, run_reinforce, run_actor_critic, run_dqn, ...

# Chapter 5: Alignment
from rlbook.algorithms.alignment import Config, SFTTrainer, RewardModel, DPOTrainer, PPOTrainer, ...

# Chapter 7: Agentic
from rlbook.algorithms.agentic import SupportAgent, EmailTriageEnvProgressive, ...
```
