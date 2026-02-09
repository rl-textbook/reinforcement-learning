# Case Study: SFT → DPO → PPO Alignment Pipeline

**Chapter 5 — Preference-Based Alignment (RLHF)**

## Problem
Implement the full RLHF pipeline on GPT-2 using the Anthropic HH-RLHF dataset:
SFT → Reward Model → DPO and PPO, evaluated with GPT-4 as judge.

## Key Findings
- SFT provides the foundation; both DPO and PPO improve over SFT baseline
- DPO is simpler (no RM needed) and competitive with PPO
- GPT-4 judge provides reliable automated evaluation

## Prerequisites
- GPU recommended (T4 or better)
- OpenAI API key for GPT-4 judge evaluation

## Run
See the [Chapter 5 notebook](../../notebooks/chapter5_alignment_rlhf.ipynb) for the complete pipeline.
