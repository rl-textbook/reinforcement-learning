# Case Study: Value Iteration vs Policy Iteration

**Chapter 2 — Fundamentals of Reinforcement Learning**

## Problem
Compare convergence behavior of Value Iteration (VI) and Policy Iteration (PI)
on grid worlds of varying complexity (deterministic, stochastic, with holes).

## Key Findings
- PI converges in fewer policy updates but each iteration is more expensive
- VI shows monotonic value improvement; PI shows rapid policy stabilization
- Both converge to identical optimal policies

## Run
```bash
python run.py
```
