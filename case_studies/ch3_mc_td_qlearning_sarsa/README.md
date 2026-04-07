# Case Study: Monte Carlo vs TD(0) Convergence

**Chapter 3 — Classical Reinforcement Learning Algorithms**

## Problem
Compare First-Visit MC Prediction and TD(0) Prediction on the Windy Desert
environment under an ε-greedy policy derived from the optimal solution.

## Key Findings
- TD(0) converges faster in early episodes due to bootstrapping
- MC produces unbiased but higher-variance estimates
- Visitation patterns differ: TD updates incrementally, MC waits for episode end

## Run
```bash
python run.py
```
