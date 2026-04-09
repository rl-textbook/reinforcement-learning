"""Model and tokenizer loading helpers for the Chapter 6 RLVR case study."""

from contextlib import nullcontext

import torch
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LEFT padding matters for RoPE decoder-only models when batching prompts for generation
    tokenizer.padding_side = "left"

    use_cuda = torch.cuda.is_available()

    bf16_supported = False
    if use_cuda and hasattr(torch.cuda, "is_bf16_supported"):
        bf16_supported = torch.cuda.is_bf16_supported()

    amp_dtype = torch.bfloat16 if bf16_supported else (torch.float16 if use_cuda else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        torch_dtype=amp_dtype,
        trust_remote_code=True,
    ).to(device)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    # Sanity: tokenizer/model vocab alignment (should be equal; resize only if needed)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    use_amp = use_cuda and (amp_dtype in (torch.float16, torch.bfloat16))
    use_scaler = use_cuda and (amp_dtype == torch.float16)
    scaler = GradScaler(enabled=use_scaler)

    def autocast_ctx():
        return autocast(dtype=amp_dtype) if use_amp else nullcontext()

    return tokenizer, model, amp_dtype, use_scaler, scaler, autocast_ctx


__all__ = ["load_model_and_tokenizer"]
