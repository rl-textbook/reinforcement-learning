"""SFT helpers for the Chapter 6 RLVR case study."""

import random
import time

import torch

from rlbook.algorithms.rlvr.config import build_sft_completion


def build_sft_batch(examples, tokenizer, formatter, cfg, device):
    prompts = [formatter.format(ex["question"]) for ex in examples]
    completions = [build_sft_completion(ex["answer"]) for ex in examples]
    texts = [p + c for p, c in zip(prompts, completions)]

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.sft_max_seq_len,
    ).to(device)

    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100  # ignore padding

    # mask prompt tokens (LEFT padding aware)
    for i, prompt_text in enumerate(prompts):
        p_ids = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_prompt_tokens,
        )["input_ids"][0]
        prompt_len = p_ids.numel()
        pad_len = int((enc["attention_mask"][i] == 0).sum().item())
        labels[i, pad_len: pad_len + prompt_len] = -100

    enc["labels"] = labels
    return enc


def sft_train(
    model,
    tokenizer,
    formatter,
    train_data,
    cfg,
    autocast_ctx,
    use_scaler: bool,
    scaler,
    device,
    steps=None,
    lr=None,
    batch_size=None,
    grad_accum=None,
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr or cfg.sft_lr)
    t0 = time.time()

    steps = cfg.sft_steps if steps is None else steps
    batch_size = cfg.sft_batch_size if batch_size is None else batch_size
    grad_accum = cfg.sft_grad_accum if grad_accum is None else grad_accum

    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        total = 0.0

        for _ in range(grad_accum):
            batch = [train_data[random.randrange(len(train_data))] for _ in range(batch_size)]
            enc = build_sft_batch(batch, tokenizer=tokenizer, formatter=formatter, cfg=cfg, device=device)

            with autocast_ctx():
                out = model(**enc)
                loss = out.loss / grad_accum

            if not torch.isfinite(loss).all():
                raise RuntimeError(f"Non-finite SFT loss at step {step}: {loss}")

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total += float(loss.item())

        if use_scaler:
            scaler.unscale_(opt)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if use_scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        if step % 25 == 0 or step == 1:
            dt = (time.time() - t0) / 60.0
            print(f"SFT step {step:4d}/{steps} | loss {total:.4f} | {dt:.1f} min")


__all__ = ["build_sft_batch", "sft_train"]
