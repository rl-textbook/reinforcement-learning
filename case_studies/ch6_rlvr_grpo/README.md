# Chapter 6 Case Study — RLVR with GRPO on GSM8K

## Problem

Can reinforcement learning with verifiable rewards (RLVR) improve a language model's
mathematical reasoning ability beyond supervised fine-tuning?

## Method

We compare three models on the GSM8K math reasoning benchmark:

1. **Base** — Pretrained GPT-2 (124M) with no task-specific training.
2. **SFT** — Fine-tuned on GSM8K chain-of-thought demonstrations using teacher forcing.
3. **GRPO (RLVR)** — Group Relative Policy Optimization starting from SFT. Samples
   G completions per prompt, scores each with a binary math verifier, and updates
   using group-relative advantages with a clipped PPO-style objective.

### Pipeline

```
Base GPT-2 → SFT (CoT demonstrations) → GRPO (verifiable rewards)
```

### Evaluation

- **Pass@k** on held-out GSM8K test set (k = 1, 4, 8)
- **Test-time compute scaling**: measured Pass@N vs independence theory
- **Reasoning quality**: response length, reasoning steps, self-correction rate

## Key Algorithms

| Algorithm | Module | Description |
|-----------|--------|-------------|
| MathVerifier | `rlbook.algorithms.rlvr.verifier` | Binary reward: 1.0 if numeric answer matches |
| CoTFormatter | `rlbook.algorithms.rlvr.formatter` | Prompt contract with FINAL: marker |
| GRPO | `rlbook.algorithms.rlvr.grpo` | Group sampling → z-score advantage → clipped objective |
| Pass@k | `rlbook.algorithms.rlvr.evaluation` | Best-of-k evaluation on held-out test |

## Running

```bash
pip install -e ".[alignment]"
python case_studies/ch6_rlvr_grpo/run.py
```

Runtime: ~30–60 min on GPU, ~2–3 hr on CPU.

## Configuration

Edit `Config` in `rlbook/algorithms/rlvr/config.py` or override in `run.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grpo_steps` | 200 | GRPO training steps |
| `group_size` | 8 | Samples per prompt (G) |
| `clip_eps` | 0.2 | PPO clipping epsilon |
| `sync_old_every` | 10 | Old policy sync interval |
| `sft_steps` | 200 | SFT baseline training steps |
