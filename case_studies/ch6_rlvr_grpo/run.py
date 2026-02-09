#!/usr/bin/env python3
"""Chapter 6 Case Study — RLVR with GRPO on GSM8K.

Usage:
    python case_studies/ch6_rlvr_grpo/run.py

Requires: pip install -e ".[alignment]"
"""

import os, re, math, copy, random, time
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from rlbook.algorithms.rlvr import (
    Config,
    parse_gsm8k_answer,
    gsm8k_solution_to_cot_and_final,
    MathVerifier,
    CoTFormatter,
    grpo_step_one_prompt,
    sync_old_policy,
    pass_at_k,
    diagnostics_on_test,
    greedy_response,
)

# ── Reproducibility ────────────────────────────────────────────────
SEED = 42

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except TypeError:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = Config()
print(f"Device: {device} | Model: {cfg.model_name}")

# ── Data ───────────────────────────────────────────────────────────
gsm8k = load_dataset("gsm8k", "main")
train_raw_full = gsm8k["train"].select(range(min(cfg.train_size, len(gsm8k["train"]))))
test_raw = gsm8k["test"].select(range(min(cfg.test_size, len(gsm8k["test"]))))

val_n = min(cfg.val_size, len(train_raw_full))
train_n = max(0, len(train_raw_full) - val_n)
train_raw = train_raw_full.select(range(train_n))
val_raw = train_raw_full.select(range(train_n, train_n + val_n))

train_data = [
    {"question": ex["question"], "answer": parse_gsm8k_answer(ex["answer"]),
     "cot": gsm8k_solution_to_cot_and_final(ex["answer"])}
    for ex in train_raw
]
val_data = [
    {"question": ex["question"], "answer": parse_gsm8k_answer(ex["answer"]),
     "cot": gsm8k_solution_to_cot_and_final(ex["answer"])}
    for ex in val_raw
]
test_data = [
    {"question": ex["question"], "answer": parse_gsm8k_answer(ex["answer"]),
     "cot": gsm8k_solution_to_cot_and_final(ex["answer"])}
    for ex in test_raw
]
print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

# ── Components ─────────────────────────────────────────────────────
verifier = MathVerifier()
formatter = CoTFormatter()
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
base_model.resize_token_embeddings(len(tokenizer))

# ── SFT ────────────────────────────────────────────────────────────
def make_sft_example(ex):
    prompt = formatter.format(ex["question"])
    full_text = prompt + " " + ex["cot"]
    enc = tokenizer(full_text, truncation=True, max_length=cfg.max_length, return_tensors=None)
    prompt_enc = tokenizer(prompt, truncation=True, max_length=cfg.max_length, return_tensors=None)
    prompt_len = len(prompt_enc["input_ids"])
    labels = enc["input_ids"].copy()
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

sft_train = [make_sft_example(ex) for ex in train_data]

def collate_batch(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    def pad(seq, pad_id):
        return seq + [pad_id] * (max_len - len(seq))
    input_ids = torch.tensor([pad(x["input_ids"], tokenizer.pad_token_id) for x in batch])
    attn = torch.tensor([pad(x["attention_mask"], 0) for x in batch])
    labels = torch.tensor([pad(x["labels"], -100) for x in batch])
    return {"input_ids": input_ids.to(device), "attention_mask": attn.to(device), "labels": labels.to(device)}

train_loader = DataLoader(sft_train, batch_size=cfg.sft_batch_size, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED), collate_fn=collate_batch)

print("\n=== SFT Training ===")
sft_model = copy.deepcopy(base_model).to(device)
sft_opt = AdamW(sft_model.parameters(), lr=cfg.sft_lr)

it = iter(train_loader)
for step in range(cfg.sft_steps):
    try:
        batch = next(it)
    except StopIteration:
        it = iter(train_loader)
        batch = next(it)
    sft_model.train()
    loss = sft_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                     labels=batch["labels"]).loss
    sft_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
    sft_opt.step()
    if (step + 1) % 50 == 0:
        print(f"  SFT step {step+1}/{cfg.sft_steps} | loss={loss.item():.4f}")
print("SFT done.")

# ── GRPO ───────────────────────────────────────────────────────────
print("\n=== GRPO Training ===")
policy = copy.deepcopy(sft_model).to(device)
old_policy = copy.deepcopy(policy).to(device)
grpo_opt = AdamW(policy.parameters(), lr=cfg.grpo_lr)

train_indices = list(range(len(train_data)))
rng = random.Random(SEED)
rng.shuffle(train_indices)

eval_steps = []
curve_pass1, curve_pass4, curve_pass8 = [], [], []

def eval_checkpoint(step_idx):
    passks = {}
    for k in cfg.eval_ks_learning_curves:
        seed_everything(SEED + 10_000 + step_idx + k)
        passks[k] = pass_at_k(policy, tokenizer, verifier, formatter, test_data, k,
                               max_new_tokens=cfg.max_new_tokens_eval,
                               temperature=cfg.temperature, top_p=cfg.top_p)
    eval_steps.append(step_idx)
    curve_pass1.append(passks[1])
    curve_pass4.append(passks[4])
    curve_pass8.append(passks[8])
    diag = diagnostics_on_test(policy, tokenizer, formatter, test_data, cfg,
                                n_examples=min(64, len(test_data)))
    print(f"  [Eval @ step {step_idx}] Pass@1={passks[1]:.3f} Pass@4={passks[4]:.3f} "
          f"Pass@8={passks[8]:.3f} | len={diag['avg_len_tokens']:.1f} "
          f"steps={diag['avg_steps']:.1f} selfcorr={diag['self_correction_rate']:.2f}")

eval_checkpoint(0)

t0 = time.time()
for step in range(1, cfg.grpo_steps + 1):
    ex = train_data[train_indices[(step - 1) % len(train_indices)]]
    seed_everything(SEED + 50_000 + step)

    loss, metrics = grpo_step_one_prompt(
        policy, old_policy, tokenizer, verifier, formatter,
        question_text=ex["question"], answer_text=ex["answer"],
        group_size=cfg.group_size, clip_eps=cfg.clip_eps, adv_eps=cfg.adv_eps,
    )

    grpo_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    grpo_opt.step()

    if step % cfg.sync_old_every == 0:
        sync_old_policy(old_policy, policy)

    if step % 10 == 0:
        print(f"  GRPO step {step:4d} | loss={loss.item():.4f} | "
              f"mean_reward={metrics['mean_reward']:.3f} ± {metrics['reward_std']:.3f}")

    if step % cfg.eval_every == 0:
        eval_checkpoint(step)

t1 = time.time()
print(f"GRPO finished in {(t1-t0)/60:.1f} min")

# ── Final eval ─────────────────────────────────────────────────────
print("\n=== Final Results (held-out test set) ===")
for label, model in [("Base", base_model), ("SFT", sft_model), ("GRPO", policy)]:
    passks = {}
    for k in cfg.eval_ks_learning_curves:
        seed_everything(SEED + 90_000 + k)
        passks[k] = pass_at_k(model, tokenizer, verifier, formatter, test_data, k,
                               max_new_tokens=cfg.max_new_tokens_eval,
                               temperature=cfg.temperature, top_p=cfg.top_p)
    diag = diagnostics_on_test(model, tokenizer, formatter, test_data, cfg,
                                n_examples=min(64, len(test_data)))
    print(f"  {label:6s} | Pass@1={passks[1]:.3f} Pass@4={passks[4]:.3f} "
          f"Pass@8={passks[8]:.3f} | len={diag['avg_len_tokens']:.1f} "
          f"steps={diag['avg_steps']:.1f} selfcorr={diag['self_correction_rate']:.2f}")

# ── Qualitative ────────────────────────────────────────────────────
print("\n=== Qualitative Example ===")
ex = test_data[0]
prompt = formatter.format(ex["question"])
print(f"Problem: {ex['question'][:200]}")
print(f"Ground truth: {ex['answer']}")
for label, m in [("SFT", sft_model), ("GRPO", policy)]:
    resp = greedy_response(m, tokenizer, prompt, max_new_tokens=128)
    print(f"\n--- {label} ---\n{resp[:300]}")
