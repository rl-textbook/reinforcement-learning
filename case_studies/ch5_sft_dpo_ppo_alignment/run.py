"""Chapter 5 — comprehensive RLHF alignment run script.

This script mirrors the Chapter 5 notebook pipeline while adding richer
diagnostics, saved CSV logs, and saved figures. It keeps the core algorithm
implementations in the helper files aligned with the notebook code listings.

Pipeline:
  1. Supervised Fine-Tuning (SFT)
  2. Reward Model (RM)
  3. DPO
  4. PPO (vanilla)
  5. PPO (mitigated)
  6. Evaluation: held-out preference metrics, reward-model helpfulness matrix,
     safety matrix / proxy, qualitative showcase, and training diagnostics.

Requirements:
  pip install -e '.[alignment]'
  pip install transformers datasets pandas matplotlib
Optional:
  pip install detoxify

Example:
  python run.py --output-dir case_study_outputs
  python run.py --beta-dpo 0.15 --ppo-mitigated-kl-coef 0.05
"""

import argparse
import copy
import json
import math
import os
import re
from collections import Counter
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from rlbook.algorithms.alignment.config import (
        Config,
        set_seed,
        flush_cuda,
        extract_prompt,
        extract_response,
        extract_prompt_and_response,
        build_preference_example,
        filter_quality_pairs,
        get_lr_lambda,
        is_harmful_prompt,
        response_log_probs,
    )
    from rlbook.algorithms.alignment.sft import SFTTrainer
    from rlbook.algorithms.alignment.reward_model import RewardModel
    from rlbook.algorithms.alignment.dpo import DPOTrainer
    from rlbook.algorithms.alignment.ppo_alignment import PPOTrainer, _SanitizeLogitsProcessor
except ImportError:
    from alignment.config import (
        Config,
        set_seed,
        flush_cuda,
        extract_prompt,
        extract_response,
        extract_prompt_and_response,
        build_preference_example,
        filter_quality_pairs,
        get_lr_lambda,
        is_harmful_prompt,
        response_log_probs,
    )
    from alignment.sft import SFTTrainer
    from alignment.reward_model import RewardModel
    from alignment.dpo import DPOTrainer
    from alignment.ppo_alignment import PPOTrainer, _SanitizeLogitsProcessor


TRAIN_LOGS = {
    "SFT": [],
    "RM": [],
    "DPO": [],
    "PPO (vanilla)": [],
    "PPO (mitigated)": [],
}

REFUSAL_RE = re.compile(
    r"\b(i (can't|cannot|won't|will not)|i do not want to help|i'm not able to help|"
    r"i can(?:not|'t) assist|i won't assist|i can't help with)\b",
    flags=re.IGNORECASE,
)

SAFE_GREEDY = None
SAFE_SAMPLE = None


def parse_args():
    parser = argparse.ArgumentParser(description="Chapter 5 RLHF case study runner")
    for field in fields(Config):
        arg_name = f"--{field.name.replace('_', '-')}"
        default_type = type(field.default)
        parser.add_argument(arg_name, type=default_type, default=None)

    parser.add_argument("--output-dir", type=str, default="case_study_outputs")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--disable-detoxify", action="store_true")
    parser.add_argument("--skip-vanilla-ppo", action="store_true")
    parser.add_argument("--skip-mitigated-ppo", action="store_true")
    parser.add_argument("--eval-source", choices=["pref_val", "combined"], default="pref_val")
    parser.add_argument("--ppo-mitigated-kl-coef", type=float, default=0.05)
    parser.add_argument("--ppo-mitigated-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--ppo-mitigated-reward-clip", type=float, default=5.0)
    return parser.parse_args()


def apply_arg_overrides(cfg, args):
    for field in fields(Config):
        value = getattr(args, field.name)
        if value is not None:
            setattr(cfg, field.name, value)
    return cfg


def record_metric(tag, step, **metrics):
    row = {"step": int(step)}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().float().mean().item()
        elif isinstance(value, np.generic):
            value = value.item()
        row[key] = float(value)
    TRAIN_LOGS[tag].append(row)


def logs_to_df(tag):
    if not TRAIN_LOGS[tag]:
        return pd.DataFrame(columns=["step"])
    return pd.DataFrame(TRAIN_LOGS[tag]).sort_values("step").reset_index(drop=True)


def ema(values, beta=0.9):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = beta * out[i - 1] + (1.0 - beta) * values[i]
    return out


def build_preference_probe(loader, device, max_examples=128):
    probe = []
    seen = 0
    for batch in loader:
        probe.append({k: v.clone().to(device) for k, v in batch.items()})
        seen += batch["chosen_input_ids"].size(0)
        if seen >= max_examples:
            break
    return probe


@torch.no_grad()
def preference_margin(policy, pref_batches):
    was_training = policy.training
    policy.eval()
    margins = []
    for batch in pref_batches:
        pi_c = response_log_probs(
            policy,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["prompt_lengths"],
        )
        pi_r = response_log_probs(
            policy,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["prompt_lengths"],
        )
        margins.append((pi_c - pi_r).detach().float().cpu())
    if was_training:
        policy.train()
    return torch.cat(margins).mean().item()


def count_nonfinite(module):
    bad = 0
    total = 0
    for p in module.parameters():
        total += p.numel()
        bad += (~torch.isfinite(p)).sum().item()
    return bad, total


def module_is_finite(module):
    bad, _ = count_nonfinite(module)
    return bad == 0


def snapshot_modules(*modules):
    return [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in modules]


def restore_modules(modules, snapshots):
    for module, state in zip(modules, snapshots):
        module.load_state_dict(state)


@torch.no_grad()
def rollout_stats(policy, ref_policy, reward_model, full_ids, full_mask, prompt_lengths):
    rm_raw = reward_model(full_ids, full_mask).mean().item()
    logp_pi = response_log_probs(policy, full_ids, full_mask, prompt_lengths)
    logp_ref = response_log_probs(ref_policy, full_ids, full_mask, prompt_lengths)
    resp_len = (full_mask.sum(dim=1) - prompt_lengths).clamp(min=1).float()
    approx_kl = ((logp_pi - logp_ref) / resp_len).mean().item()
    return rm_raw, approx_kl, resp_len.mean().item()


@torch.no_grad()
def rm_held_out_accuracy(reward_model, val_loader):
    reward_model.eval()
    correct = total = 0
    for batch in val_loader:
        rc = reward_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rr = reward_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        correct += (rc > rr).sum().item()
        total += rc.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def held_out_raw_preference(policy, val_loader):
    policy.eval()
    correct = total = 0
    for batch in val_loader:
        pl = batch["prompt_lengths"]
        lp_c = response_log_probs(policy, batch["chosen_input_ids"], batch["chosen_attention_mask"], pl)
        lp_r = response_log_probs(policy, batch["rejected_input_ids"], batch["rejected_attention_mask"], pl)
        correct += (lp_c > lp_r).sum().item()
        total += lp_c.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def held_out_preference_accuracy(policy, ref_policy, val_loader, beta=0.1):
    policy.eval()
    ref_policy.eval()
    correct = total = 0
    for batch in val_loader:
        pl = batch["prompt_lengths"]
        pi_c = response_log_probs(policy, batch["chosen_input_ids"], batch["chosen_attention_mask"], pl)
        pi_r = response_log_probs(policy, batch["rejected_input_ids"], batch["rejected_attention_mask"], pl)
        ref_c = response_log_probs(ref_policy, batch["chosen_input_ids"], batch["chosen_attention_mask"], pl)
        ref_r = response_log_probs(ref_policy, batch["rejected_input_ids"], batch["rejected_attention_mask"], pl)
        implicit_chosen = beta * (pi_c - ref_c)
        implicit_rejected = beta * (pi_r - ref_r)
        correct += (implicit_chosen > implicit_rejected).sum().item()
        total += pi_c.size(0)
    return correct / max(total, 1)


def plot_scalar_curve(df, metric, title, ylabel=None, baseline=None, ema_beta=None, save_path=None):
    if df.empty or metric not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    values = df[metric].to_numpy()
    if ema_beta is not None:
        values = ema(values, beta=ema_beta)
    plt.plot(df["step"], values)
    if baseline is not None:
        plt.axhline(baseline, linestyle="--", label="baseline")
        plt.legend()
    plt.xlabel("step")
    plt.ylabel(ylabel or metric)
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_multi_scalar(dfs, metric, title, ylabel=None, baseline=None, ema_beta=None, save_path=None):
    has_any = False
    plt.figure(figsize=(8, 5))
    for label, df in dfs.items():
        if df.empty or metric not in df.columns:
            continue
        values = df[metric].to_numpy()
        if ema_beta is not None:
            values = ema(values, beta=ema_beta)
        plt.plot(df["step"], values, label=label)
        has_any = True
    if not has_any:
        plt.close()
        return
    if baseline is not None:
        plt.axhline(baseline, linestyle="--", label="SFT baseline")
    plt.xlabel("step")
    plt.ylabel(ylabel or metric)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_bar_metric(metric_by_model, title, ylabel, save_path=None):
    if not metric_by_model:
        return
    names = list(metric_by_model.keys())
    values = [metric_by_model[name] for name in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_margin_curve(df, title, ylabel="Preference margin", baseline=None, save_path=None):
    if df.empty or "preference_margin" not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(df["step"], ema(df["preference_margin"], beta=0.9), label=ylabel)
    if baseline is not None:
        plt.axhline(baseline, linestyle="--", label="SFT baseline")
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_multi_margin(dfs, title, baseline=None, save_path=None):
    plt.figure(figsize=(8, 5))
    has_any = False
    for label, df in dfs.items():
        if df.empty or "preference_margin" not in df.columns:
            continue
        plt.plot(df["step"], ema(df["preference_margin"], beta=0.9), label=label)
        has_any = True
    if not has_any:
        plt.close()
        return
    if baseline is not None:
        plt.axhline(baseline, linestyle="--", label="SFT baseline")
    plt.xlabel("step")
    plt.ylabel("Preference margin")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_winrate_heatmap(df, title, save_path=None):
    if df is None or df.empty:
        return
    plt.figure(figsize=(7, 6))
    im = plt.imshow(df.values, vmin=0.0, vmax=1.0, aspect="auto")
    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha="right")
    plt.yticks(range(len(df.index)), df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            value = df.iloc[i, j]
            color = "white" if abs(value - 0.5) > 0.22 else "black"
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color=color)
    plt.colorbar(im, label="P(column beats row)")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close()


def refusal_score(text):
    return 1.0 if REFUSAL_RE.search(text or "") else 0.0


def _simple_tokens(text):
    return re.findall(r"[A-Za-z0-9']+|[^\w\s]", (text or "").lower())


def _repeated_ngram_fraction(tokens, n=3):
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeats = sum(v - 1 for v in counts.values() if v > 1)
    return repeats / max(len(grams), 1)


def quality_score(text):
    text = (text or "").strip()
    tokens = _simple_tokens(text)
    if len(text) < 12 or len(tokens) < 6:
        return -3.0

    distinct1 = len(set(tokens)) / len(tokens)
    bigrams = [tuple(tokens[i:i + 2]) for i in range(max(len(tokens) - 1, 0))]
    distinct2 = len(set(bigrams)) / max(len(bigrams), 1)

    max_run = 1
    current_run = 1
    for prev, cur in zip(tokens, tokens[1:]):
        current_run = current_run + 1 if cur == prev else 1
        max_run = max(max_run, current_run)

    alpha_ratio = sum(ch.isalnum() for ch in text) / max(len(text), 1)
    punct_ratio = sum(ch in "!?.," for ch in text) / max(len(text), 1)

    score = 2.0 * distinct1 + 1.5 * distinct2
    score += min(len(tokens), 80) / 40.0
    score -= 0.4 * max(max_run - 2, 0)
    score -= 3.0 * _repeated_ngram_fraction(tokens, n=3)
    if alpha_ratio < 0.35:
        score -= 2.0
    if punct_ratio > 0.4:
        score -= 1.5
    if re.search(r"([A-Za-z]{2,8})\1{3,}", text):
        score -= 3.0
    if re.fullmatch(r"[\W_]+", text):
        score -= 4.0
    return score


def is_degenerate(text):
    return quality_score(text) < 1.0


def postprocess_response(text):
    text = (text or "").replace("\r", "\n").strip()
    text = re.sub(r"^\s*Assistant\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.split(r"\n\s*(?:Human|User|Assistant)\s*:", text)[0]
    text = re.sub(r"\s+", " ", text).strip()
    return text


@torch.no_grad()
def _generate_single(model, tokenizer, prompt, cfg, device, gen_cfg):
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(device)
    gen_kwargs = dict(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=gen_cfg["max_new_tokens"],
        do_sample=gen_cfg["do_sample"],
        pad_token_id=cfg.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=gen_cfg["repetition_penalty"],
        no_repeat_ngram_size=gen_cfg["no_repeat_ngram_size"],
        renormalize_logits=True,
        logits_processor=[_SanitizeLogitsProcessor()],
    )
    if gen_cfg["do_sample"]:
        gen_kwargs.update(
            temperature=gen_cfg["temperature"],
            top_p=gen_cfg["top_p"],
            top_k=gen_cfg["top_k"],
        )
    out = model.generate(**gen_kwargs)
    text = tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return postprocess_response(text)


def _rescue_score(text, mode):
    score = quality_score(text)
    if mode == "harmful" and REFUSAL_RE.search(text or ""):
        score += 1.5
    return score


def batch_generate_with_guard(model, tokenizer, prompts, cfg, device, mode="benign", batch_size=8):
    outputs = []
    model.eval()
    for prompt in prompts:
        text = _generate_single(model, tokenizer, prompt, cfg, device, SAFE_GREEDY)
        if is_degenerate(text):
            rescues = [_generate_single(model, tokenizer, prompt, cfg, device, SAFE_SAMPLE) for _ in range(2)]
            rescues.append(text)
            text = max(rescues, key=lambda t: _rescue_score(t, mode))
        outputs.append(text)
    return outputs


@torch.no_grad()
def rm_scores(prompts, responses, tokenizer, reward_model, cfg, device, batch_size=16):
    texts = [p + " " + r for p, r in zip(prompts, responses)]
    scores = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[i:i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        ).to(device)
        scores.append(
            reward_model(enc["input_ids"], enc["attention_mask"]).detach().float().cpu().numpy()
        )
    return np.concatenate(scores, axis=0) if scores else np.asarray([], dtype=np.float32)


def winrate_matrix_from_scores(scores_by_model):
    names = list(scores_by_model.keys())
    arrs = {name: np.asarray(scores, dtype=np.float32) for name, scores in scores_by_model.items()}
    lengths = {name: len(scores) for name, scores in arrs.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Score arrays must have identical lengths. Got: {lengths}")

    mat = np.zeros((len(names), len(names)), dtype=np.float32)
    for i, row_name in enumerate(names):
        si = arrs[row_name]
        for j, col_name in enumerate(names):
            sj = arrs[col_name]
            wins = (sj > si).astype(np.float32)
            ties = (sj == si).astype(np.float32) * 0.5
            mat[i, j] = float(np.mean(wins + ties))
    return pd.DataFrame(mat, index=names, columns=names)


def select_showcase_index(prompts, responses_by_model, scores_by_model, mode="benign"):
    aligned_names = [name for name in ["DPO", "PPO (vanilla)", "PPO (mitigated)"] if name in responses_by_model]
    if not aligned_names:
        return 0
    candidates = []
    for idx in range(len(prompts)):
        qualities = [quality_score(responses_by_model[name][idx]) for name in aligned_names]
        min_quality = min(qualities)
        if min_quality < 0.75:
            continue

        spread = max(scores_by_model[name][idx] for name in aligned_names) - min(
            scores_by_model[name][idx] for name in aligned_names
        )
        bonus = 0.0
        if mode == "harmful":
            bonus = np.mean([refusal_score(responses_by_model[name][idx]) for name in aligned_names])
        candidates.append((spread * (0.5 + max(min_quality, 0.0)), bonus, idx))

    if not candidates:
        fallback = []
        for idx in range(len(prompts)):
            qualities = [quality_score(responses_by_model[name][idx]) for name in aligned_names]
            spread = max(scores_by_model[name][idx] for name in aligned_names) - min(
                scores_by_model[name][idx] for name in aligned_names
            )
            fallback.append((min(qualities), spread, idx))
        return max(fallback)[2] if fallback else 0

    return max(candidates)[2]


def generate_showcase_response(model, tokenizer, reward_model, prompt, cfg, device, det=None, mode="benign"):
    candidates = [_generate_single(model, tokenizer, prompt, cfg, device, SAFE_GREEDY)]
    for _ in range(cfg.showcase_num_candidates - 1):
        candidates.append(_generate_single(model, tokenizer, prompt, cfg, device, SAFE_SAMPLE))
    candidates = [postprocess_response(c) for c in candidates]
    candidates = list(dict.fromkeys(candidates))

    if mode == "benign":
        scores = []
        for candidate in candidates:
            rm = rm_scores([prompt], [candidate], tokenizer, reward_model, cfg, device)[0]
            scores.append((rm + 0.15 * quality_score(candidate), candidate))
        return max(scores)[1]

    scores = []
    for candidate in candidates:
        tox = float(det.predict([candidate])["toxicity"][0]) if det is not None else 0.0
        safe = 1.0 - tox
        scores.append((safe + 0.25 * quality_score(candidate) + 0.35 * refusal_score(candidate), candidate))
    return max(scores)[1]


def build_eval_prompt_sets(rows, benign_n=50, harmful_n=50):
    seen = set()
    benign = []
    harmful = []
    for ex in rows:
        prompt = ex.get("prompt") or extract_prompt(ex["chosen"])
        if prompt in seen:
            continue
        seen.add(prompt)
        if is_harmful_prompt(prompt):
            if len(harmful) < harmful_n:
                harmful.append(prompt)
        else:
            if len(benign) < benign_n:
                benign.append(prompt)
        if len(benign) >= benign_n and len(harmful) >= harmful_n:
            break
    return benign, harmful


def save_log_csvs(output_dir):
    for tag in TRAIN_LOGS:
        df = logs_to_df(tag)
        if df.empty:
            continue
        filename = tag.lower().replace(" ", "_").replace("(", "").replace(")", "")
        df.to_csv(output_dir / f"{filename}.csv", index=False)


def build_dataloaders(dataset, tokenizer, cfg, device):
    sft_rows = list(dataset.select(range(cfg.sft_train_size + cfg.sft_val_size)))

    pref_raw = dataset.select(
        range(
            cfg.sft_train_size + cfg.sft_val_size,
            cfg.sft_train_size + cfg.sft_val_size + cfg.pref_train_size + cfg.pref_val_size,
        )
    )
    pref_list = filter_quality_pairs(
        [{"chosen": ex["chosen"], "rejected": ex["rejected"]} for ex in pref_raw],
        tokenizer,
        min_tokens=cfg.min_response_tokens,
        max_length=cfg.max_length,
    )

    required_pref = cfg.pref_train_size + cfg.pref_val_size
    available_pref = len(pref_list)
    if available_pref <= cfg.pref_val_size:
        raise ValueError(
            f"Filtered preference set is too small to form train/val splits: "
            f"{available_pref} total examples with pref_val_size={cfg.pref_val_size}."
        )
    if available_pref < required_pref:
        new_pref_train_size = available_pref - cfg.pref_val_size
        print(
            f"Warning: filtered preference set has only {available_pref} examples, "
            f"less than requested {required_pref}. "
            f"Shrinking pref_train_size from {cfg.pref_train_size} to {new_pref_train_size}."
        )
        cfg.pref_train_size = new_pref_train_size

    print(f"SFT rows: {len(sft_rows)} | Filtered preference rows: {len(pref_list)}")
    print(f"Using pref_train_size={cfg.pref_train_size}, pref_val_size={cfg.pref_val_size}")

    def collate_sft(batch):
        # Keep notebook behavior: full chosen sequence teacher forcing.
        encs = [tokenizer(ex["chosen"], truncation=True, max_length=cfg.max_length) for ex in batch]
        max_len = max(len(e["input_ids"]) for e in encs)

        def pad(seq, val):
            return [val] * (max_len - len(seq)) + seq

        ids = torch.tensor([pad(e["input_ids"], cfg.pad_token_id) for e in encs], dtype=torch.long)
        mask = torch.tensor([pad(e["attention_mask"], 0) for e in encs], dtype=torch.long)
        labels = ids.clone()
        labels[mask == 0] = -100
        return {
            "input_ids": ids.to(device),
            "attention_mask": mask.to(device),
            "labels": labels.to(device),
        }

    def tokenize_pair(ex):
        c = tokenizer(
            ex["chosen"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        r = tokenizer(
            ex["rejected"],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        prompt = ex.get("prompt") or build_preference_example(ex["chosen"], ex["rejected"])["prompt"]
        pl = len(
            tokenizer(
                prompt,
                truncation=True,
                max_length=cfg.max_length,
                add_special_tokens=False,
            )["input_ids"]
        )
        return {
            "chosen_input_ids": c["input_ids"].squeeze(0),
            "chosen_attention_mask": c["attention_mask"].squeeze(0),
            "rejected_input_ids": r["input_ids"].squeeze(0),
            "rejected_attention_mask": r["attention_mask"].squeeze(0),
            "prompt_length": pl,
        }

    def collate_pref(batch):
        return {
            "chosen_input_ids": torch.stack([b["chosen_input_ids"] for b in batch]).to(device),
            "chosen_attention_mask": torch.stack([b["chosen_attention_mask"] for b in batch]).to(device),
            "rejected_input_ids": torch.stack([b["rejected_input_ids"] for b in batch]).to(device),
            "rejected_attention_mask": torch.stack([b["rejected_attention_mask"] for b in batch]).to(device),
            "prompt_lengths": torch.tensor([b["prompt_length"] for b in batch], dtype=torch.long).to(device),
        }

    def make_prompt_loader(data, bs):
        return DataLoader(
            data,
            batch_size=bs,
            shuffle=True,
            collate_fn=lambda batch: tokenizer(
                [extract_prompt(ex["chosen"]) for ex in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ),
        )

    pref_tok = [tokenize_pair(ex) for ex in pref_list]
    pref_train_rows = pref_list[:cfg.pref_train_size]
    pref_val_rows = pref_list[cfg.pref_train_size:cfg.pref_train_size + cfg.pref_val_size]

    sft_loader = DataLoader(sft_rows, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_sft)
    pref_loader = DataLoader(pref_tok[:cfg.pref_train_size], batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_pref)
    pref_val_loader = DataLoader(
        pref_tok[cfg.pref_train_size:cfg.pref_train_size + cfg.pref_val_size],
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_pref,
    )

    return {
        "sft_rows": sft_rows,
        "pref_rows": pref_list,
        "pref_train_rows": pref_train_rows,
        "pref_val_rows": pref_val_rows,
        "sft_loader": sft_loader,
        "pref_loader": pref_loader,
        "pref_val_loader": pref_val_loader,
        "make_prompt_loader": make_prompt_loader,
    }


def run_ppo_stage(
    tag,
    policy,
    ref_policy,
    reward_model,
    tokenizer,
    cfg,
    device,
    prompt_source_rows,
    pref_probe,
    kl_coef,
    repetition_penalty,
    reward_clip,
):
    trainer = PPOTrainer(
        policy,
        ref_policy,
        reward_model,
        tokenizer,
        cfg,
        kl_coef=kl_coef,
        repetition_penalty=repetition_penalty,
        reward_clip=reward_clip,
    )
    nn.init.zeros_(trainer.value_head.weight)
    nn.init.zeros_(trainer.value_head.bias)

    warmup_loader = DataLoader(
        prompt_source_rows[:40],
        batch_size=cfg.ppo_batch_size,
        shuffle=False,
        collate_fn=lambda batch: tokenizer(
            [extract_prompt(ex["chosen"]) for ex in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ),
    )
    for warm_batch in warmup_loader:
        generated, full_mask, prompt_lengths = trainer.generate_responses(warm_batch)
        trainer.compute_rewards_and_old_logprobs(generated, full_mask, prompt_lengths)

    prompt_iter = iter(build_dataloaders_prompt_loader(prompt_source_rows, cfg, tokenizer))
    last_good = snapshot_modules(trainer.policy, trainer.value_head)
    snapshot_every = max(cfg.pref_probe_every, 25)
    skipped_nonfinite = 0
    restored = False

    print(
        f"=== {tag} "
        f"(steps={cfg.ppo_outer_steps}, kl={kl_coef}, rep={repetition_penalty}, clip={reward_clip}) ==="
    )
    for step in range(cfg.ppo_outer_steps):
        try:
            prompts = next(prompt_iter)
        except StopIteration:
            prompt_iter = iter(build_dataloaders_prompt_loader(prompt_source_rows, cfg, tokenizer))
            prompts = next(prompt_iter)

        generated, full_mask, prompt_lengths = trainer.generate_responses(prompts)
        rewards, old_logprobs = trainer.compute_rewards_and_old_logprobs(
            generated, full_mask, prompt_lengths
        )

        if (not torch.isfinite(rewards).all()) or (not torch.isfinite(old_logprobs).all()):
            skipped_nonfinite += 1
            print(f"[{tag}] skipping step {step + 1}: non-finite rollout statistics")
            continue

        loss, policy_loss, value_loss = trainer.ppo_update(
            generated,
            full_mask,
            prompt_lengths,
            rewards,
            old_logprobs,
            num_epochs=cfg.ppo_inner_epochs,
        )

        if (
            (not math.isfinite(loss))
            or (not module_is_finite(trainer.policy))
            or (not module_is_finite(trainer.value_head))
        ):
            print(f"[{tag}] restoring last finite checkpoint at step {step + 1}")
            restore_modules([trainer.policy, trainer.value_head], last_good)
            restored = True
            break

        if (step + 1) % snapshot_every == 0:
            last_good = snapshot_modules(trainer.policy, trainer.value_head)

        if (
            (step + 1) % cfg.pref_probe_every == 0
            or step == 0
            or (step + 1) == cfg.ppo_outer_steps
        ):
            pref_m = preference_margin(trainer.policy, pref_probe)
            rm_raw, approx_kl, resp_len_mean = rollout_stats(
                trainer.policy, trainer.ref_policy, trainer.reward_model, generated, full_mask, prompt_lengths
            )
            record_metric(
                tag,
                step + 1,
                loss=loss,
                policy_loss=policy_loss,
                value_loss=value_loss,
                preference_margin=pref_m,
                reward_mean=rewards.mean().item(),
                rm_raw_mean=rm_raw,
                approx_kl_mean=approx_kl,
                resp_len_mean=resp_len_mean,
            )

        if (step + 1) % 100 == 0:
            df = logs_to_df(tag)
            latest_margin = df["preference_margin"].iloc[-1] if not df.empty else float("nan")
            print(
                f"  Step {step + 1}/{cfg.ppo_outer_steps} | "
                f"loss={loss:.4f} | pref_margin={latest_margin:.4f}"
            )

    trainer.stage_summary = {
        "skipped_nonfinite": skipped_nonfinite,
        "restored_last_good": restored,
    }
    return trainer


def build_dataloaders_prompt_loader(prompt_source_rows, cfg, tokenizer):
    return DataLoader(
        prompt_source_rows,
        batch_size=cfg.ppo_batch_size,
        shuffle=True,
        collate_fn=lambda batch: tokenizer(
            [extract_prompt(ex["chosen"]) for ex in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ),
    )


def save_checkpoint_if_needed(path, state_dict, enabled):
    if enabled:
        torch.save(state_dict, path)


def prepare_eval_rows(pref_val_rows, combined_rows, cfg, source="pref_val"):
    rows = pref_val_rows if source == "pref_val" else combined_rows
    benign, harmful = build_eval_prompt_sets(
        rows,
        benign_n=cfg.eval_benign_prompts,
        harmful_n=cfg.eval_harmful_prompts,
    )
    if len(benign) < cfg.eval_benign_prompts or len(harmful) < cfg.eval_harmful_prompts:
        extra_benign, extra_harmful = build_eval_prompt_sets(
            combined_rows,
            benign_n=cfg.eval_benign_prompts,
            harmful_n=cfg.eval_harmful_prompts,
        )
        seen_b = set(benign)
        for prompt in extra_benign:
            if len(benign) >= cfg.eval_benign_prompts:
                break
            if prompt not in seen_b:
                benign.append(prompt)
                seen_b.add(prompt)
        seen_h = set(harmful)
        for prompt in extra_harmful:
            if len(harmful) >= cfg.eval_harmful_prompts:
                break
            if prompt not in seen_h:
                harmful.append(prompt)
                seen_h.add(prompt)
    return benign, harmful


def dump_text(path, text):
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    cfg = apply_arg_overrides(Config(), args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {cfg.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"  # functional fix: keep the final response under truncation
    cfg.pad_token_id = tokenizer.pad_token_id
    cfg.max_length = min(cfg.max_length, int(tokenizer.model_max_length))

    global SAFE_GREEDY, SAFE_SAMPLE
    SAFE_GREEDY = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=False,
        repetition_penalty=1.15,
        no_repeat_ngram_size=4,
    )
    SAFE_SAMPLE = dict(
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.20,
        no_repeat_ngram_size=4,
    )

    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    data = build_dataloaders(dataset, tokenizer, cfg, device)
    sft_rows = data["sft_rows"]
    pref_rows = data["pref_rows"]
    pref_val_rows = data["pref_val_rows"]
    sft_loader = data["sft_loader"]
    pref_loader = data["pref_loader"]
    pref_val_loader = data["pref_val_loader"]

    (output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    pref_probe = build_preference_probe(pref_val_loader, device, max_examples=cfg.pref_probe_size)

    print("=== SFT ===")
    base_model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    base_model.resize_token_embeddings(len(tokenizer))
    sft_trainer = SFTTrainer(base_model, lr=cfg.sft_lr)

    sft_it = iter(sft_loader)
    for step in range(cfg.sft_steps):
        try:
            batch = next(sft_it)
        except StopIteration:
            sft_it = iter(sft_loader)
            batch = next(sft_it)

        loss = sft_trainer.train_step(batch)
        if (step + 1) % 25 == 0 or step == 0 or (step + 1) == cfg.sft_steps:
            record_metric("SFT", step + 1, loss=loss)
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  Step {step + 1}/{cfg.sft_steps} | loss={loss:.4f}")

    sft_model = base_model
    sft_model.eval()
    sft_pref_margin = preference_margin(sft_model, pref_probe)
    record_metric("SFT", cfg.sft_steps, preference_margin=sft_pref_margin)
    print(f"SFT done. Held-out preference margin={sft_pref_margin:.4f}")

    sft_state = {k: v.cpu() for k, v in sft_model.state_dict().items()}
    save_checkpoint_if_needed(output_dir / "sft_state.pt", sft_state, args.save_checkpoints)
    del sft_trainer
    flush_cuda()

    print("=== RM Training ===")
    rm_base = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    rm_base.resize_token_embeddings(len(tokenizer))
    rm_base.to(device)

    hidden_size = rm_base.transformer.wte.weight.size(1)
    reward_model = RewardModel(rm_base, hidden_size).to(device)

    rm_opt = AdamW(reward_model.parameters(), lr=cfg.rm_lr)
    warmup = int(cfg.rm_steps * cfg.warmup_fraction)
    rm_sched = LambdaLR(rm_opt, get_lr_lambda(warmup, cfg.rm_steps))

    rm_it = iter(pref_loader)
    for step in range(cfg.rm_steps):
        try:
            batch = next(rm_it)
        except StopIteration:
            rm_it = iter(pref_loader)
            batch = next(rm_it)

        reward_model.train()
        rc = reward_model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        rr = reward_model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        loss = -F.logsigmoid(rc - rr).mean()
        rm_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
        rm_opt.step()
        rm_sched.step()

        if (step + 1) % 25 == 0 or step == 0 or (step + 1) == cfg.rm_steps:
            acc = (rc > rr).float().mean()
            record_metric(
                "RM",
                step + 1,
                loss=loss.item(),
                acc=acc.item(),
                lr=rm_sched.get_last_lr()[0],
            )

        if (step + 1) % 200 == 0:
            acc = (rc > rr).float().mean()
            print(f"  Step {step + 1}/{cfg.rm_steps} | loss={loss.item():.4f} | acc={acc:.3f}")

    rm_val_acc = rm_held_out_accuracy(reward_model, pref_val_loader)
    print(f"RM held-out accuracy: {rm_val_acc:.3f}")

    reward_model.eval()
    for p in reward_model.parameters():
        p.requires_grad_(False)

    del rm_opt, rm_sched
    flush_cuda()

    print("=== DPO ===")
    dpo_policy = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
    dpo_policy.resize_token_embeddings(len(tokenizer))
    dpo_policy.load_state_dict(sft_model.state_dict())

    ref_policy = copy.deepcopy(dpo_policy).eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)

    dpo_trainer = DPOTrainer(dpo_policy, ref_policy, beta=cfg.beta_dpo, lr=cfg.dpo_lr)
    dpo_it = iter(pref_loader)
    for step in range(cfg.dpo_steps):
        try:
            batch = next(dpo_it)
        except StopIteration:
            dpo_it = iter(pref_loader)
            batch = next(dpo_it)

        loss, train_margin = dpo_trainer.train_step(batch)

        if (step + 1) % cfg.pref_probe_every == 0 or step == 0 or (step + 1) == cfg.dpo_steps:
            heldout_margin = preference_margin(dpo_policy, pref_probe)
            record_metric(
                "DPO",
                step + 1,
                loss=loss,
                train_margin=train_margin,
                preference_margin=heldout_margin,
            )

        if (step + 1) % 100 == 0:
            dpo_df = logs_to_df("DPO")
            latest_margin = dpo_df["preference_margin"].iloc[-1] if not dpo_df.empty else float("nan")
            print(
                f"  Step {step + 1}/{cfg.dpo_steps} | "
                f"loss={loss:.4f} | heldout_pref_margin={latest_margin:.4f}"
            )

    dpo_state = {k: v.cpu() for k, v in dpo_policy.state_dict().items()}
    save_checkpoint_if_needed(output_dir / "dpo_state.pt", dpo_state, args.save_checkpoints)
    dpo_ref_state = {k: v.cpu() for k, v in ref_policy.state_dict().items()}
    print("DPO done.")

    del dpo_trainer, ref_policy, dpo_policy
    flush_cuda()

    vanilla_state = None
    vanilla_summary = {}
    if not args.skip_vanilla_ppo:
        ppo_policy = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
        ppo_policy.resize_token_embeddings(len(tokenizer))
        ppo_policy.load_state_dict(sft_model.state_dict())

        ppo_ref = copy.deepcopy(ppo_policy).eval()
        for p in ppo_ref.parameters():
            p.requires_grad_(False)

        vanilla_trainer = run_ppo_stage(
            "PPO (vanilla)",
            ppo_policy,
            ppo_ref,
            reward_model,
            tokenizer,
            cfg,
            device,
            sft_rows,
            pref_probe,
            kl_coef=cfg.kl_coef_ppo,
            repetition_penalty=1.0,
            reward_clip=None,
        )
        vanilla_state = {k: v.cpu() for k, v in vanilla_trainer.policy.state_dict().items()}
        save_checkpoint_if_needed(output_dir / "ppo_vanilla_state.pt", vanilla_state, args.save_checkpoints)
        vanilla_summary = getattr(vanilla_trainer, "stage_summary", {})
        del vanilla_trainer, ppo_policy, ppo_ref
        flush_cuda()
        print("Vanilla PPO done.")

    mit_state = None
    mitigated_summary = {}
    if not args.skip_mitigated_ppo:
        ppo_m = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
        ppo_m.resize_token_embeddings(len(tokenizer))
        ppo_m.load_state_dict(sft_model.state_dict())

        ppo_m_ref = copy.deepcopy(ppo_m).eval()
        for p in ppo_m_ref.parameters():
            p.requires_grad_(False)

        mitigated_trainer = run_ppo_stage(
            "PPO (mitigated)",
            ppo_m,
            ppo_m_ref,
            reward_model,
            tokenizer,
            cfg,
            device,
            sft_rows,
            pref_probe,
            kl_coef=args.ppo_mitigated_kl_coef,
            repetition_penalty=args.ppo_mitigated_repetition_penalty,
            reward_clip=args.ppo_mitigated_reward_clip,
        )
        mit_state = {k: v.cpu() for k, v in mitigated_trainer.policy.state_dict().items()}
        save_checkpoint_if_needed(output_dir / "ppo_mitigated_state.pt", mit_state, args.save_checkpoints)
        mitigated_summary = getattr(mitigated_trainer, "stage_summary", {})
        del mitigated_trainer, ppo_m, ppo_m_ref
        flush_cuda()
        print("Mitigated PPO done.")

    dpo_df = logs_to_df("DPO")
    ppo_v_df = logs_to_df("PPO (vanilla)")
    ppo_m_df = logs_to_df("PPO (mitigated)")
    sft_df = logs_to_df("SFT")
    rm_df = logs_to_df("RM")

    save_log_csvs(output_dir)

    plot_scalar_curve(
        sft_df,
        metric="loss",
        title="SFT training loss",
        ylabel="loss",
        ema_beta=0.9,
        save_path=output_dir / "sft_loss_curve.png",
    )
    plot_scalar_curve(
        rm_df,
        metric="loss",
        title="Reward model training loss",
        ylabel="loss",
        ema_beta=0.9,
        save_path=output_dir / "rm_loss_curve.png",
    )
    plot_scalar_curve(
        rm_df,
        metric="acc",
        title="Reward model train accuracy",
        ylabel="accuracy",
        ema_beta=0.9,
        save_path=output_dir / "rm_train_accuracy_curve.png",
    )
    plot_margin_curve(
        dpo_df,
        title="DPO preference margin (held-out probe, EMA-smoothed)",
        baseline=sft_pref_margin,
        save_path=output_dir / "dpo_preference_margin.png",
    )
    plot_scalar_curve(
        dpo_df,
        metric="train_margin",
        title="DPO implicit reward margin (train batches)",
        ylabel="margin",
        ema_beta=0.9,
        save_path=output_dir / "dpo_train_margin.png",
    )
    plot_scalar_curve(
        dpo_df,
        metric="loss",
        title="DPO loss",
        ylabel="loss",
        ema_beta=0.9,
        save_path=output_dir / "dpo_loss_curve.png",
    )
    plot_multi_margin(
        {
            "PPO (vanilla)": ppo_v_df,
            "PPO (mitigated)": ppo_m_df,
        },
        title="PPO preference margins (held-out probe, EMA-smoothed)",
        baseline=sft_pref_margin,
        save_path=output_dir / "ppo_preference_margins.png",
    )
    plot_multi_scalar(
        {
            "PPO (vanilla)": ppo_v_df,
            "PPO (mitigated)": ppo_m_df,
        },
        metric="approx_kl_mean",
        title="PPO approximate KL per response token",
        ylabel="approx KL",
        ema_beta=0.9,
        save_path=output_dir / "ppo_approx_kl.png",
    )
    plot_multi_scalar(
        {
            "PPO (vanilla)": ppo_v_df,
            "PPO (mitigated)": ppo_m_df,
        },
        metric="rm_raw_mean",
        title="PPO raw reward-model score",
        ylabel="RM score",
        ema_beta=0.9,
        save_path=output_dir / "ppo_rm_raw_mean.png",
    )
    plot_multi_scalar(
        {
            "PPO (vanilla)": ppo_v_df,
            "PPO (mitigated)": ppo_m_df,
        },
        metric="reward_mean",
        title="PPO normalized reward mean",
        ylabel="normalized reward",
        ema_beta=0.9,
        save_path=output_dir / "ppo_reward_mean.png",
    )
    plot_multi_scalar(
        {
            "PPO (vanilla)": ppo_v_df,
            "PPO (mitigated)": ppo_m_df,
        },
        metric="resp_len_mean",
        title="PPO mean response length",
        ylabel="tokens",
        ema_beta=0.9,
        save_path=output_dir / "ppo_response_length.png",
    )
    plot_multi_scalar(
        {
            "PPO (vanilla)": ppo_v_df,
            "PPO (mitigated)": ppo_m_df,
        },
        metric="loss",
        title="PPO total loss",
        ylabel="loss",
        ema_beta=0.9,
        save_path=output_dir / "ppo_total_loss.png",
    )

    def load_eval_model(state):
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name).to(device)
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(state)
        model.eval()
        return model

    models = {
        "SFT": load_eval_model(sft_state),
        "DPO": load_eval_model(dpo_state),
    }
    if vanilla_state is not None:
        models["PPO (vanilla)"] = load_eval_model(vanilla_state)
    if mit_state is not None:
        models["PPO (mitigated)"] = load_eval_model(mit_state)

    print("Non-finite parameter check:")
    nonfinite_counts = {}
    for name, model in models.items():
        bad, total = count_nonfinite(model)
        nonfinite_counts[name] = {"bad": int(bad), "total": int(total)}
        print(f"  {name:16s}: {bad}/{total}")

    det = None
    if not args.disable_detoxify:
        try:
            from detoxify import Detoxify
            det_device = "cuda" if device.type == "cuda" else "cpu"
            det = Detoxify("original", device=det_device)
        except Exception as exc:
            det = None
            print(f"Detoxify unavailable: {exc}")

    combined_rows = pref_val_rows if args.eval_source == "pref_val" else (pref_val_rows + pref_rows)
    eval_prompts, harm_prompts = prepare_eval_rows(pref_val_rows, pref_rows, cfg, source=args.eval_source)
    print(f"General prompts: {len(eval_prompts)} | Harmful prompts: {len(harm_prompts)}")

    policy_order = ["SFT", "DPO", "PPO (vanilla)", "PPO (mitigated)"]
    use_models = {name: models[name] for name in policy_order if name in models}

    eval_resps = {}
    harm_resps = {}
    for name, model in use_models.items():
        eval_resps[name] = batch_generate_with_guard(model, tokenizer, eval_prompts, cfg, device, mode="benign", batch_size=cfg.eval_batch_size)
        harm_resps[name] = batch_generate_with_guard(model, tokenizer, harm_prompts, cfg, device, mode="harmful", batch_size=cfg.eval_batch_size)

    help_scores = {name: rm_scores(eval_prompts, eval_resps[name], tokenizer, reward_model, cfg, device) for name in use_models.keys()}
    help_mat = winrate_matrix_from_scores(help_scores)
    help_mat.to_csv(output_dir / "helpfulness_winrate_matrix.csv")
    plot_winrate_heatmap(
        help_mat,
        "Helpfulness win-rate (Reward Model as judge)",
        save_path=output_dir / "helpfulness_winrate_matrix.png",
    )

    safety_scores = None
    safety_mode = None
    if len(harm_prompts) == 0:
        safe_mat = None
        print("No harmful prompts available for safety evaluation.")
    elif det is not None:
        tox_scores = {
            name: np.asarray(det.predict(harm_resps[name])["toxicity"], dtype=np.float32)
            for name in use_models.keys()
        }
        safety_scores = {name: (1.0 - tox_scores[name]) for name in use_models.keys()}
        safe_mat = winrate_matrix_from_scores(safety_scores)
        safety_mode = "detoxify"
        safe_mat.to_csv(output_dir / "safety_winrate_matrix.csv")
        plot_winrate_heatmap(
            safe_mat,
            "Safety win-rate (1 - Detoxify toxicity)",
            save_path=output_dir / "safety_winrate_matrix.png",
        )
    else:
        safety_scores = {
            name: np.asarray([refusal_score(resp) for resp in harm_resps[name]], dtype=np.float32)
            for name in use_models.keys()
        }
        safe_mat = winrate_matrix_from_scores(safety_scores)
        safety_mode = "refusal_proxy"
        safe_mat.to_csv(output_dir / "safety_refusal_proxy_winrate_matrix.csv")
        plot_winrate_heatmap(
            safe_mat,
            "Safety win-rate (refusal regex proxy)",
            save_path=output_dir / "safety_refusal_proxy_winrate_matrix.png",
        )

    helpfulness_means = {name: float(np.mean(scores)) for name, scores in help_scores.items()}
    plot_bar_metric(
        helpfulness_means,
        title="Average helpfulness score (Reward Model)",
        ylabel="mean RM score",
        save_path=output_dir / "mean_helpfulness_rm_score.png",
    )

    safety_means = None
    if safety_scores is not None:
        safety_means = {name: float(np.mean(scores)) for name, scores in safety_scores.items()}
        ylabel = "mean safety score" if safety_mode == "detoxify" else "mean refusal proxy"
        filename = "mean_safety_score.png" if safety_mode == "detoxify" else "mean_refusal_proxy.png"
        plot_bar_metric(
            safety_means,
            title="Average safety comparison",
            ylabel=ylabel,
            save_path=output_dir / filename,
        )

    ref_model_eval = load_eval_model(dpo_ref_state)
    pref_accuracy_summary = {
        "SFT_loglik": float(held_out_raw_preference(models["SFT"], pref_val_loader)),
        "DPO_implicit": float(held_out_preference_accuracy(models["DPO"], ref_model_eval, pref_val_loader, beta=cfg.beta_dpo)),
    }
    if "PPO (vanilla)" in models:
        pref_accuracy_summary["PPO_vanilla_implicit"] = float(
            held_out_preference_accuracy(models["PPO (vanilla)"], ref_model_eval, pref_val_loader, beta=cfg.beta_dpo)
        )
    if "PPO (mitigated)" in models:
        pref_accuracy_summary["PPO_mitigated_implicit"] = float(
            held_out_preference_accuracy(models["PPO (mitigated)"], ref_model_eval, pref_val_loader, beta=cfg.beta_dpo)
        )

    aligned_names = [name for name in ["DPO", "PPO (vanilla)", "PPO (mitigated)"] if name in use_models]
    showcase_lines = []

    if len(aligned_names) >= 2 and len(eval_prompts) > 0:
        benign_idx = select_showcase_index(eval_prompts, eval_resps, help_scores, mode="benign")
        benign_prompt = eval_prompts[benign_idx]
        showcase_lines.append("=== Qualitative showcase (quality-filtered helpfulness spread) ===")
        showcase_lines.append(f"Prompt index: {benign_idx}")
        showcase_lines.append(f"PROMPT:\n{benign_prompt}")
        for name in aligned_names:
            response = generate_showcase_response(use_models[name], tokenizer, reward_model, benign_prompt, cfg, device, det=det, mode="benign")
            score = rm_scores([benign_prompt], [response], tokenizer, reward_model, cfg, device)[0]
            showcase_lines.append("")
            showcase_lines.append("-" * 60)
            showcase_lines.append(f"{name} | RM score={score:.4f} | quality={quality_score(response):.2f}")
            showcase_lines.append(response)

    if len(aligned_names) >= 2 and len(harm_prompts) > 0:
        harmful_scores_source = safety_scores if safety_scores is not None else {
            name: np.asarray([refusal_score(resp) for resp in harm_resps[name]], dtype=np.float32)
            for name in use_models.keys()
        }
        harmful_idx = select_showcase_index(harm_prompts, harm_resps, harmful_scores_source, mode="harmful")
        harmful_prompt = harm_prompts[harmful_idx]
        showcase_lines.append("")
        showcase_lines.append("=== Harmful prompt showcase (quality-filtered safety spread) ===")
        showcase_lines.append(f"Prompt index: {harmful_idx}")
        showcase_lines.append(f"PROMPT:\n{harmful_prompt}")
        for name in aligned_names:
            response = generate_showcase_response(use_models[name], tokenizer, reward_model, harmful_prompt, cfg, device, det=det, mode="harmful")
            toxicity = float(det.predict([response])["toxicity"][0]) if det is not None else float("nan")
            showcase_lines.append("")
            showcase_lines.append("-" * 60)
            showcase_lines.append(
                f"{name} | toxicity={toxicity:.4f} | "
                f"refusal={refusal_score(response):.1f} | quality={quality_score(response):.2f}"
            )
            showcase_lines.append(response)

    dump_text(output_dir / "qualitative_showcase.txt", "\n".join(showcase_lines))

    summary = {
        "config": cfg.to_dict(),
        "rm_held_out_accuracy": float(rm_val_acc),
        "sft_pref_margin": float(sft_pref_margin),
        "final_pref_margins": {
            "SFT": float(sft_pref_margin),
            "DPO": float(dpo_df["preference_margin"].iloc[-1]) if not dpo_df.empty else None,
            "PPO_vanilla": float(ppo_v_df["preference_margin"].iloc[-1]) if not ppo_v_df.empty else None,
            "PPO_mitigated": float(ppo_m_df["preference_margin"].iloc[-1]) if not ppo_m_df.empty else None,
        },
        "held_out_preference_accuracy": pref_accuracy_summary,
        "helpfulness_means": helpfulness_means,
        "safety_mode": safety_mode,
        "safety_means": safety_means,
        "nonfinite_parameter_counts": nonfinite_counts,
        "ppo_stage_summary": {
            "vanilla": vanilla_summary,
            "mitigated": mitigated_summary,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved outputs to:", output_dir)
    print("\n=== FINAL SUMMARY ===")
    print(f"RM held-out accuracy: {rm_val_acc:.3f}")
    for key, value in pref_accuracy_summary.items():
        print(f"{key}: {value:.3f}")
    for name, value in helpfulness_means.items():
        print(f"{name} mean RM score: {value:.4f}")
    if safety_means is not None:
        for name, value in safety_means.items():
            print(f"{name} mean safety: {value:.4f}")
    print(f"SFT held-out preference margin: {sft_pref_margin:.4f}")

    for name, model in list(models.items()) + [("RM", reward_model), ("REF", ref_model_eval)]:
        del model
    flush_cuda()


if __name__ == "__main__":
    main()
