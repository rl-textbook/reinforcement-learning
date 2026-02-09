# Reinforcement Learning: From Foundations to Frontiers

> **Companion code repository for the textbook by Uday Kamath and Vedant**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive, hands-on codebase spanning five decades of reinforcement learning — from tabular methods and dynamic programming through deep RL, preference-based alignment (RLHF), reasoning models and RLVR, and agentic architectures.

---

## 📖 Book Overview

| Chapter | Title | Notebook | Key Algorithms |
|---------|-------|----------|----------------|
| 1 | Introduction to Reinforcement Learning | — | Historical context, RL taxonomy |
| 2 | Fundamentals of Reinforcement Learning | [Ch2 Notebook](notebooks/chapter2_mdp_foundations.ipynb) | Value Iteration, Policy Iteration |
| 3 | Classical Reinforcement Learning Algorithms | [Ch3 Notebook](notebooks/chapter3_classical_algorithms.ipynb) | MC Prediction, TD(0), Q-Learning, SARSA |
| 4 | Deep Reinforcement Learning | [Ch4 Notebook](notebooks/chapter4_deep_rl.ipynb) | DQN, Double DQN, REINFORCE, Actor-Critic |
| 5 | Preference-Based Alignment (RLHF) | [Ch5 Notebook](notebooks/chapter5_alignment_rlhf.ipynb) | SFT, Reward Model, DPO, PPO (vanilla + mitigated) |
| 6 | Reasoning Models & RLVR | [Ch6 Notebook](notebooks/chapter6_rlvr_grpo.ipynb) | GRPO, Verifiable Rewards, Pass@k |
| 7 | Agentic Architectures in RL | [Ch7 Notebook](notebooks/chapter7_agentic_email_triage.ipynb) | RL-learned tool use, Memory systems |

---

## 🚀 Quick Start (5 Minutes)

```bash
git clone https://github.com/rl-textbook/reinforcement-learning.git
cd reinforcement-learning
python3 -m venv venv && source venv/bin/activate
pip install -e ".[all]"
python -c "import rlbook; print(f'rlbook v{rlbook.__version__} ready')"
```

If that worked, jump to [Running the Notebooks](#-running-the-notebooks-locally) or [Using the Package](#-using-the-package-in-your-own-code).

---

## 🔧 Detailed Installation

### Prerequisites

| Requirement | Check with | Minimum |
|-------------|-----------|---------|
| Python | `python3 --version` | 3.9+ |
| pip | `pip --version` | 21.0+ |
| Git | `git --version` | any |
| GPU *(optional)* | `nvidia-smi` | CUDA 11.8+ (for Chapters 4–7) |

### Step 1: Clone the Repository

```bash
git clone https://github.com/rl-textbook/reinforcement-learning.git
cd reinforcement-learning
```

### Step 2: Create a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> After activation you should see `(venv)` at the start of your terminal prompt.

**Alternative — Conda:**
```bash
conda create -n rlbook python=3.11 -y
conda activate rlbook
```

### Step 3: Install Dependencies

Choose **one** of the following based on which chapters you plan to work through:

```bash
# Chapters 1–3 only (lightweight, ~50 MB, no GPU needed)
pip install -e .

# + Chapter 4: Deep RL (adds PyTorch + Gymnasium, ~2 GB)
pip install -e ".[deep]"

# + Chapters 5–6: RLHF alignment & RLVR (adds Transformers + Datasets)
pip install -e ".[alignment]"

# + Chapter 7: Agentic RL (adds Stable-Baselines3)
pip install -e ".[agentic]"

# Everything — all chapters + development tools
pip install -e ".[all]"
```

#### PyTorch with GPU Support

If you have an NVIDIA GPU, install PyTorch with CUDA **before** the package:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Apple Silicon (MPS support included by default)
pip install torch

# Then install rlbook
pip install -e ".[all]"
```

### Step 4: Verify Installation

```bash
python -c "import rlbook; print(f'rlbook v{rlbook.__version__} installed')"
```

For a more thorough check:

```bash
python -c "
from rlbook.environments import create_grid_world
from rlbook.algorithms.dynamic_programming import value_iteration, extract_policy

env = create_grid_world(grid_size=4)
V = value_iteration(env)
pi = extract_policy(env, V)
print('Value Iteration converged — installation verified!')
"
```

### Step 5: Deactivate When Done

```bash
deactivate          # venv
# conda deactivate  # if using conda
```

Next time, just re-activate:
```bash
cd reinforcement-learning
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate.bat  # Windows CMD
```

---

## 📓 Running the Notebooks Locally

Each notebook imports algorithms and utilities from the `rlbook` package, so you need to install the package first (`pip install -e .`). This gives you a clean separation: algorithms live in `rlbook/`, notebooks focus on configuration, execution, and visualization.

### Option A: Jupyter Notebook / JupyterLab

```bash
# 1. Activate the virtual environment
source venv/bin/activate

# 2. Install Jupyter (first time only)
pip install jupyter

# 3. Register the venv as a Jupyter kernel (first time only)
pip install ipykernel
python -m ipykernel install --user --name=rlbook --display-name "Python (rlbook)"

# 4. Launch Jupyter
jupyter notebook notebooks/
#   or for the modern interface:
# pip install jupyterlab
# jupyter lab notebooks/
```

When the browser opens, click on any `.ipynb` file. Make sure to select **Kernel → Change Kernel → Python (rlbook)** so that Jupyter uses the correct virtual environment.

Run cells one by one with **Shift + Enter**, or run the entire notebook with **Cell → Run All**.

### Option B: VS Code

1. Open the `reinforcement-learning/` folder in VS Code
2. Install the **Python** and **Jupyter** extensions (if not installed)
3. Open any `.ipynb` file from `notebooks/`
4. In the top-right kernel picker, select the Python interpreter from your `venv/` directory
5. Run cells with **Shift + Enter**

### Option C: Google Colab (Zero Local Setup)

Click any Colab badge below — no local installation needed:

| Notebook | Description | Colab |
|----------|-------------|-------|
| Chapter 2 | MDP framework, Value/Policy Iteration on Grid World | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-textbook/reinforcement-learning/blob/main/notebooks/chapter2_mdp_foundations.ipynb) |
| Chapter 3 | MC vs TD, Q-Learning vs SARSA on Windy Desert | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-textbook/reinforcement-learning/blob/main/notebooks/chapter3_classical_algorithms.ipynb) |
| Chapter 4 | DQN, REINFORCE, Actor-Critic on CartPole | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-textbook/reinforcement-learning/blob/main/notebooks/chapter4_deep_rl.ipynb) |
| Chapter 5 | RLHF pipeline: SFT → RM → DPO/PPO with GPT-4 Judge | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-textbook/reinforcement-learning/blob/main/notebooks/chapter5_alignment_rlhf.ipynb) |
| Chapter 6 | RLVR with GRPO on GSM8K math reasoning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-textbook/reinforcement-learning/blob/main/notebooks/chapter6_rlvr_grpo.ipynb) |
| Chapter 7 | Progressive agentic RL with tool use and memory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rl-textbook/reinforcement-learning/blob/main/notebooks/chapter7_agentic_email_triage.ipynb) |

> **Tip:** For Chapters 4, 5, 6, and 7, switch to a GPU runtime: **Runtime → Change runtime type → T4 GPU**.

**Colab Setup (first cell of every notebook):**

Each notebook's first code cell includes an install command. On Colab, uncomment the appropriate line:

```python
# Chapters 2-3 (tabular only):
!pip install -q git+https://github.com/rl-textbook/reinforcement-learning.git

# Chapter 4 (deep RL):
!pip install -q "git+https://github.com/rl-textbook/reinforcement-learning.git#egg=rlbook[deep]"

# Chapters 5–6 (alignment / RLVR):
!pip install -q "git+https://github.com/rl-textbook/reinforcement-learning.git#egg=rlbook[alignment]"

# Chapter 7 (agentic):
!pip install -q "git+https://github.com/rl-textbook/reinforcement-learning.git#egg=rlbook[agentic]"

# Everything:
!pip install -q "git+https://github.com/rl-textbook/reinforcement-learning.git#egg=rlbook[all]"
```

### Chapter-by-Chapter Notebook Guide

| Notebook | Run Time | GPU Needed? | Special Setup |
|----------|----------|-------------|---------------|
| Chapter 2 | ~2 min | No | None |
| Chapter 3 | ~5 min | No | None |
| Chapter 4 | ~10 min (CPU), ~3 min (GPU) | Recommended | `pip install -e ".[deep]"` |
| Chapter 5 | ~30 min (GPU) | Recommended | `pip install -e ".[alignment]"` + OpenAI API key for GPT-4 judge |
| Chapter 6 | ~30–60 min (GPU) | Recommended | `pip install -e ".[alignment]"` |
| Chapter 7 | ~15 min (GPU) | Recommended | `pip install -e ".[agentic]"` |

---

## 🏃 Running Case Studies

Each case study is a standalone experiment from the textbook with its own `run.py` script:

```bash
source venv/bin/activate

# Chapter 2: Value Iteration vs Policy Iteration
python case_studies/ch2_value_iteration_vs_policy_iteration/run.py

# Chapter 3: Monte Carlo vs TD(0) convergence
python case_studies/ch3_mc_vs_td_convergence/run.py

# Chapter 3: Q-Learning vs SARSA
python case_studies/ch3_qlearning_vs_sarsa/run.py

# Chapter 4: DQN vs REINFORCE vs Actor-Critic — 3 seeds (~15 min on CPU)
python case_studies/ch4_dqn_vs_reinforce_vs_ac/run.py

# Chapter 5: SFT → DPO → PPO alignment pipeline (~10-15 min GPU, ~30-45 min CPU)
python case_studies/ch5_sft_dpo_ppo_alignment/run.py

# Chapter 6: RLVR with GRPO on GSM8K (~30-60 min GPU)
python case_studies/ch6_rlvr_grpo/run.py

# Chapter 7: Agentic email triage — progressive 6-level RL (~5-10 min CPU)
python case_studies/ch7_agentic_email_triage/run.py
```

Each directory also contains a `README.md` with the problem description, methodology, and key findings. See [case_studies/README.md](case_studies/README.md) for the full index.

---

## 💻 Using the Package in Your Own Code

After installing (`pip install -e .`), import any algorithm in a Python script, REPL, or notebook:

### Chapter 2: Dynamic Programming

```python
from rlbook.environments import create_grid_world
from rlbook.algorithms.dynamic_programming import value_iteration, extract_policy, policy_iteration

# Create a 4x4 grid world with stochastic transitions and holes
env = create_grid_world(grid_size=5, transition_noise=0.2, holes=[(1,1), (2,3)])

# Solve with value iteration
V = value_iteration(env, theta=0.001)
policy = extract_policy(env, V)

# Or solve with policy iteration (produces identical results)
policy_pi, V_pi = policy_iteration(env, theta=0.001)
```

### Chapter 3: Classical Tabular Algorithms

```python
from rlbook.environments import create_windy_desert_env
from rlbook.algorithms.dynamic_programming import value_iteration, extract_policy
from rlbook.algorithms.tabular import (
    make_epsilon_greedy_policy,
    first_visit_mc_prediction, td0_prediction,
    q_learning, sarsa,
    extract_greedy_policy, evaluate_policy,
)

env = create_windy_desert_env()

# Prediction: MC vs TD(0)
V_opt = value_iteration(env, theta=1e-3, max_iterations=500)
greedy = extract_policy(env, V_opt)
policy = make_epsilon_greedy_policy(greedy, env.actions, epsilon=0.1)

V_mc = first_visit_mc_prediction(policy, env, num_episodes=500)
V_td = td0_prediction(policy, env, num_episodes=500)

# Control: Q-Learning vs SARSA
Q_ql, snapshots_ql = q_learning(env, episodes=4000, alpha=0.3, epsilon=0.1)
Q_sa, snapshots_sa = sarsa(env, episodes=4000, alpha=0.3, epsilon=0.1)

# Evaluate learned policies
pi_ql = extract_greedy_policy(Q_ql, env)
mean_return, mean_steps, success_rate = evaluate_policy(env, pi_ql)
print(f"Q-Learning: return={mean_return:.2f}, steps={mean_steps:.1f}, success={success_rate:.0f}%")
```

### Chapter 4: Deep RL

```python
from rlbook.environments import CartPoleWrapper
from rlbook.algorithms.deep.runners import RunConfig, run_reinforce, run_actor_critic, run_dqn
from rlbook.algorithms.deep.dqn import DQNConfig

env = CartPoleWrapper()
cfg = RunConfig(total_step_budget=100_000, eval_every_steps=5_000)

logs_reinforce  = run_reinforce(seed=42, env=env, cfg=cfg)
logs_ac         = run_actor_critic(seed=42, env=env, cfg=cfg)
logs_dqn        = run_dqn(seed=42, env=env, run_cfg=cfg, dqn_cfg=DQNConfig(double_dqn=True))

for logs in [logs_reinforce, logs_ac, logs_dqn]:
    last = logs[-1]
    print(f"{last['algo']:15s} | final return: {last['return_mean']:.1f}")
```


### Chapter 5: Alignment (RLHF)

```python
from rlbook.algorithms.alignment import Config, SFTTrainer, RewardModel, DPOTrainer, PPOTrainer
from rlbook.algorithms.alignment.config import sequence_log_probs
from rlbook.algorithms.alignment.gpt4_judge import gpt4_judge_pair, gpt4_judge_safety

# Full pipeline is demonstrated in the Chapter 5 notebook
cfg = Config()
```

### Chapter 6: RLVR (Reasoning with Verifiable Rewards)

```python
from rlbook.algorithms.rlvr import (
    Config, MathVerifier, CoTFormatter,
    grpo_step_one_prompt, sync_old_policy, pass_at_k,
)

cfg = Config()
verifier = MathVerifier()
formatter = CoTFormatter()

# Binary reward: 1.0 if numeric answer matches, else 0.0
reward, debug = verifier.reward("FINAL: 29", "29")  # → (1.0, "pred=29.0, gt=29.0")

# Full GRPO pipeline is demonstrated in the Chapter 6 notebook
```

---

## ✅ Running Tests

```bash
source venv/bin/activate
pip install pytest

# Run all tests
pytest tests/ -v

# Run a specific test
pytest tests/test_algorithms.py::test_vi_pi_agree -v
```

Tests verify algorithm correctness on small environments (runs in seconds, no GPU needed).

---

## 📁 Repository Structure

```
reinforcement-learning/
│
├── README.md                              ← You are here
├── ALGORITHMS.md                          ← Algorithm catalog with links to source & chapters
├── CONTRIBUTORS.md                        ← Author attribution
├── LICENSE                                ← MIT License (code)
├── pyproject.toml                         ← Package build config (pip install -e .)
├── requirements.txt                       ← Flat dependency list
├── .gitignore
│
├── rlbook/                                ← 📦 Importable Python package
│   ├── __init__.py                        ← Package version
│   ├── core/
│   │   ├── mdp.py                         ← State, Environment, Agent classes
│   │   ├── replay_buffer.py               ← Circular experience replay buffer
│   │   ├── agents.py                      ← PolicyAgent, RandomWalkAgent, run_episode
│   │   └── analysis.py                    ← VisitationCounter, ValueTrajectory, RMSE, MC/TD traces
│   ├── environments/
│   │   ├── grid_world.py                  ← Configurable NxN grid (Ch 2)
│   │   ├── windy_desert.py                ← Stochastic desert with quicksand (Ch 3)
│   │   └── cartpole_wrapper.py            ← Gymnasium CartPole-v1 wrapper (Ch 4)
│   ├── algorithms/
│   │   ├── dynamic_programming/
│   │   │   ├── value_iteration.py         ← Bellman optimality backup
│   │   │   ├── policy_iteration.py        ← Evaluation + improvement loop
│   │   │   └── traces.py                  ← Instrumented VI/PI with convergence recording
│   │   ├── tabular/
│   │   │   ├── monte_carlo.py             ← First-visit MC prediction
│   │   │   ├── td_learning.py             ← TD(0) prediction
│   │   │   ├── q_learning.py              ← Off-policy TD control
│   │   │   ├── sarsa.py                   ← On-policy TD control
│   │   │   └── helpers.py                 ← ε-greedy, policy extraction, evaluation
│   │   ├── deep/
│   │   │   ├── dqn.py                     ← DQN + Double DQN agent
│   │   │   ├── reinforce.py               ← REINFORCE with baseline
│   │   │   ├── actor_critic.py            ← One-step TD actor-critic
│   │   │   └── runners.py                 ← Training loops + evaluation harness
│   │   ├── alignment/
│   │   │   ├── config.py                  ← Config, data processing, sequence_log_probs
│   │   │   ├── sft.py                     ← Supervised fine-tuning trainer
│   │   │   ├── reward_model.py            ← Learned scalar reward model
│   │   │   ├── dpo.py                     ← Direct Preference Optimization
│   │   │   ├── ppo_alignment.py           ← PPO for LLM alignment (vanilla + mitigated)
│   │   │   └── gpt4_judge.py              ← LLM-as-a-Judge (pairwise + safety)
│   │   ├── rlvr/
│   │   │   ├── config.py                  ← Config, GSM8K parsing helpers
│   │   │   ├── verifier.py                ← MathVerifier (binary RLVR reward)
│   │   │   ├── formatter.py               ← CoTFormatter (FINAL: marker contract)
│   │   │   ├── grpo.py                    ← GRPO: sampling, logprobs, clipped update
│   │   │   └── evaluation.py              ← Pass@k, reasoning diagnostics
│   │   └── agentic/
│   │       ├── email_triage.py            ← Customer memory, baseline agents
│   │       ├── support_agent.py           ← LLM-based SupportAgent (GPT-2)
│   │       ├── scenarios.py               ← EMAIL_SCENARIOS dataset (20 scenarios)
│   │       └── environment.py             ← EmailTriageEnvProgressive, evaluate_rl_model
│   └── viz/                               ← 📊 Reusable visualization utilities
│       ├── grid_world.py                  ← Grid world value/policy plots (Ch 2)
│       └── desert.py                      ← Windy Desert maps, MC/TD plots (Ch 3)
│
├── notebooks/                             ← 📓 Interactive chapter notebooks (import from rlbook)
│   ├── README.md
│   ├── chapter2_mdp_foundations.ipynb
│   ├── chapter3_classical_algorithms.ipynb
│   ├── chapter4_deep_rl.ipynb
│   ├── chapter5_alignment_rlhf.ipynb
│   ├── chapter6_rlvr_grpo.ipynb
│   └── chapter7_agentic_email_triage.ipynb
│
├── case_studies/                          ← 🔬 Standalone reproducible experiments
│   ├── README.md
│   ├── ch2_value_iteration_vs_policy_iteration/
│   │   ├── README.md
│   │   ├── run.py
│   │   └── figures/
│   ├── ch3_mc_vs_td_convergence/
│   ├── ch3_qlearning_vs_sarsa/
│   ├── ch4_dqn_vs_reinforce_vs_ac/
│   ├── ch5_sft_dpo_ppo_alignment/
│   ├── ch6_rlvr_grpo/
│   └── ch7_agentic_email_triage/
│
├── figures/                               ← 📊 Publication-quality figures by chapter
│   ├── chapter1/ ... chapter7/
│
└── tests/                                 ← ✅ Algorithm correctness tests
    └── test_algorithms.py
```

---

## 📋 Dependencies by Chapter

| Chapter | Extra Packages | GPU? | Install Extras |
|---------|----------------|------|----------------|
| 1 | *(no code)* | No | — |
| 2 | `numpy`, `matplotlib` | No | `pip install -e .` |
| 3 | `numpy`, `matplotlib` | No | `pip install -e .` |
| 4 | + `torch`, `gymnasium` | Recommended | `pip install -e ".[deep]"` |
| 5 | + `transformers`, `datasets`, `openai`¹ | Recommended | `pip install -e ".[alignment]"` |
| 6 | + `transformers`, `datasets` | Recommended | `pip install -e ".[alignment]"` |
| 7 | + `stable-baselines3`, `transformers` | Recommended | `pip install -e ".[agentic]"` |

¹ The `openai` package is only needed for the GPT-4 judge evaluation. All training code (SFT, Reward Model, DPO, PPO) works without it.

### Chapter 5: OpenAI API Key Setup

The GPT-4 judge evaluation requires an API key. Set it before running the notebook:

```bash
export OPENAI_API_KEY='sk-...'        # macOS/Linux
set OPENAI_API_KEY=sk-...             # Windows CMD
$env:OPENAI_API_KEY = "sk-..."        # PowerShell
```

Or inside a notebook cell:
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

---

## 🔄 Notebooks vs Package — How They Relate

The notebooks and `rlbook` package share a **single source of truth** architecture:

| | Notebooks (`notebooks/`) | Package (`rlbook/`) | Case Studies (`case_studies/`) |
|---|---|---|---|
| **Purpose** | Interactive walkthroughs with plots | Importable algorithms for any project | Standalone CLI experiments |
| **Code** | Imports from `rlbook` | Canonical algorithm implementations | Imports from `rlbook` |
| **Visualizations** | Full plots, animations, analysis | Reusable viz utilities in `rlbook.viz` | Figures saved to disk |
| **Requires install?** | Yes — `pip install -e .` | Yes — `pip install -e .` | Yes — `pip install -e .` |


---

## 🗺️ Algorithm Catalog

See [**ALGORITHMS.md**](ALGORITHMS.md) for a complete, navigable index of every algorithm with links to source modules, chapter references, and case studies.

---

## 🧰 Troubleshooting

### `ModuleNotFoundError: No module named 'rlbook'`

Make sure you installed the package *and* your venv is active:
```bash
cd reinforcement-learning
source venv/bin/activate
pip install -e .
```

### `ModuleNotFoundError: No module named 'torch'`

Install the deep learning extras:
```bash
pip install -e ".[deep]"     # Chapter 4
pip install -e ".[all]"      # All chapters
```

### Jupyter doesn't see packages from the venv

Register the venv as a Jupyter kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name=rlbook --display-name "Python (rlbook)"
```
Then restart Jupyter and select **Kernel → Python (rlbook)**.

### `CUDA out of memory` (Chapters 4, 5, 6, 7)

Reduce batch sizes. For Chapter 5, change `cfg.batch_size` from 8 to 4 and `cfg.max_length` from 512 to 256 in the Config cell. For Chapter 6, reduce `cfg.group_size` from 8 to 4.

### Colab disconnects during long training

Enable Colab Pro for longer sessions, or reduce training steps in the Config cell. Chapter 5 and 6 defaults are demo-scale and should complete in ~30–60 min on a T4.

### Tests fail with import errors

Run from the repo root with the package installed:
```bash
cd reinforcement-learning
source venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

---

## 🤝 Contributing

This repository accompanies a published textbook. We welcome:

- **Bug fixes** — errors in algorithms or case studies
- **Visualizations** — improved plots, animations, interactive demos
- **Case studies** — new experiments using the `rlbook` package
- **Documentation** — clearer docstrings, additional examples
- **Translations** — notebook translations to other languages

**How to contribute:**

```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR-USERNAME/reinforcement-learning.git
cd reinforcement-learning

# 3. Create a branch
git checkout -b fix/value-iteration-typo

# 4. Make changes, add tests if applicable
# 5. Push and open a pull request
git push origin fix/value-iteration-typo
```

Please open an issue before submitting large changes.

---

## 📄 License

Code is released under the [MIT License](LICENSE). Textbook content (LaTeX, figures, prose) is © the authors.

## 📚 Citation

```bibtex
@book{kamath2026reinforcement,
  title     = {Reinforcement Learning: From Foundations to Frontiers},
  author    = {Kamath, Uday and Vajre, Vedant},
  year      = {2026},
  publisher = {CRC Press}
}
```

## 📬 Contact

- **Issues & Questions**: [GitHub Issues](https://github.com/rl-textbook/reinforcement-learning/issues)
