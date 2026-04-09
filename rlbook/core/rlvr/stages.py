"""High-level stage runners for the Chapter 6 RLVR case study."""

import random
import time

import torch

from rlbook.algorithms.rlvr.evaluation import evaluate_passk_and_diagnostics
from rlbook.algorithms.rlvr.grpo import grpo_update_step


def run_grpo_training_loop(
    model,
    tokenizer,
    verifier,
    formatter,
    train_ds,
    test_ds,
    cfg,
    seed,
    autocast_ctx,
    use_scaler: bool,
    scaler,
    device,
    batch_prompts_eval: int = 8,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.grpo_lr)

    eval_steps = []
    grpo_pass1, grpo_pass4, grpo_pass8 = [], [], []
    diag_len, diag_steps, diag_math, diag_selfcorr = [], [], [], []
    train_mean_reward, train_loss = [], []

    # Eval at step 0 (SFT-initialized policy)
    p0, d0, _ = evaluate_passk_and_diagnostics(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        formatter=formatter,
        dataset=test_ds,
        n_eval=min(cfg.eval_curve_n, len(test_ds)),
        ks=cfg.eval_ks_curve,
        seed=cfg.eval_seed,
        max_new_tokens=cfg.max_new_tokens_eval,
        temperature=cfg.temperature_eval,
        top_p=cfg.top_p_eval,
        batch_prompts=batch_prompts_eval,
        max_prompt_tokens=cfg.max_prompt_tokens,
        device=device,
    )

    print(
        f"[Eval @ step 0] Pass@1={p0[1]:.3f} Pass@4={p0[4]:.3f} Pass@8={p0[8]:.3f} | "
        f"len={d0['len_tokens']:.1f} steps={d0['steps']:.1f} selfcorr={d0['selfcorr_rate']:.2f}"
    )

    eval_steps.append(0)
    grpo_pass1.append(p0[1]); grpo_pass4.append(p0[4]); grpo_pass8.append(p0[8])
    diag_len.append(d0["len_tokens"]); diag_steps.append(d0["steps"])
    diag_math.append(d0["math_markers"]); diag_selfcorr.append(d0["selfcorr_rate"])

    t0 = time.time()

    for step in range(1, cfg.grpo_steps + 1):
        batch = [train_ds[random.randrange(len(train_ds))] for _ in range(cfg.prompts_per_step)]
        metrics = grpo_update_step(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            formatter=formatter,
            batch_examples=batch,
            optimizer=optimizer,
            cfg=cfg,
            seed=seed + 10_000 + step,
            autocast_ctx=autocast_ctx,
            use_scaler=use_scaler,
            scaler=scaler,
            device=device,
        )

        train_mean_reward.append(metrics["mean_reward"])
        train_loss.append(metrics["loss"])

        if step % 10 == 0 or step == 1:
            dt = (time.time() - t0) / 60.0
            print(
                f"GRPO step {step:4d} | loss {metrics['loss']:.6f} | "
                f"mean_reward {metrics['mean_reward']:.4f} +- {metrics['reward_std']:.4f} | "
                f"adv_nz {metrics['adv_nz']:.2f} | exact {metrics['exact']} | parse_fail {metrics['parse_fail']} | "
                f"tokens {metrics['tokens']} | {dt:.1f} min"
            )

        if step % cfg.eval_every == 0:
            p, d, _ = evaluate_passk_and_diagnostics(
                model=model,
                tokenizer=tokenizer,
                verifier=verifier,
                formatter=formatter,
                dataset=test_ds,
                n_eval=min(cfg.eval_curve_n, len(test_ds)),
                ks=cfg.eval_ks_curve,
                seed=cfg.eval_seed,
                max_new_tokens=cfg.max_new_tokens_eval,
                temperature=cfg.temperature_eval,
                top_p=cfg.top_p_eval,
                batch_prompts=batch_prompts_eval,
                max_prompt_tokens=cfg.max_prompt_tokens,
                device=device,
            )
            print(
                f"[Eval @ step {step}] Pass@1={p[1]:.3f} Pass@4={p[4]:.3f} Pass@8={p[8]:.3f} | "
                f"len={d['len_tokens']:.1f} steps={d['steps']:.1f} selfcorr={d['selfcorr_rate']:.2f}"
            )

            eval_steps.append(step)
            grpo_pass1.append(p[1]); grpo_pass4.append(p[4]); grpo_pass8.append(p[8])
            diag_len.append(d["len_tokens"]); diag_steps.append(d["steps"])
            diag_math.append(d["math_markers"]); diag_selfcorr.append(d["selfcorr_rate"])

    return {
        "eval_steps": eval_steps,
        "grpo_pass1": grpo_pass1,
        "grpo_pass4": grpo_pass4,
        "grpo_pass8": grpo_pass8,
        "diag_len": diag_len,
        "diag_steps": diag_steps,
        "diag_math": diag_math,
        "diag_selfcorr": diag_selfcorr,
        "train_mean_reward": train_mean_reward,
        "train_loss": train_loss,
    }


__all__ = ["run_grpo_training_loop"]
