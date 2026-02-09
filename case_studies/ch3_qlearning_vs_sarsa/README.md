# Case Study: Q-Learning vs SARSA

**Chapter 3 — Classical Reinforcement Learning Algorithms**

## Problem
Compare off-policy (Q-Learning) vs on-policy (SARSA) TD control on the
Windy Desert environment with quicksand hazards.

## Key Findings
- Q-Learning finds the optimal (shortest) path but may traverse near hazards
- SARSA learns a safer policy that avoids quicksand under ε-greedy exploration
- Reward shaping accelerates convergence for both algorithms

## Run
```bash
python run.py
```
