"""Train NanoChat-X.

    python -m src.train                      # defaults (char tokenizer)
    python -m src.train --max_iters 2000 --n_layer 6 --tokenizer word
    python -m src.train --resume             # continue from out/ckpt.pt

Features folded in from the "NanoScope" plan: train/val split, AdamW with
weight-decay groups, warmup + cosine LR schedule, gradient clipping, gradient
accumulation, mixed precision (auto-enabled on CUDA), periodic train/val eval,
best-checkpoint saving with resume, and a CSV loss log for plotting.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import fields

import torch

from src.config import TrainConfig, GPTConfig
from src.model import NanoGPT
from src.tokenizer import build_tokenizer, save_tokenizer, tokenizer_from_dict
from src.data import read_text, build_corpus
from utils.helpers import cosine_lr, log_training_step, CsvLogger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> tuple[TrainConfig, bool, dict]:
    p = argparse.ArgumentParser(description="Train NanoChat-X")
    p.add_argument("--resume", action="store_true", help="resume from out/ckpt.pt")
    # Every scalar TrainConfig/GPTConfig field is an override flag.
    defaults = TrainConfig()
    for name, val in defaults.to_dict().items():
        if name == "model":
            for mname, mval in val.items():
                p.add_argument(f"--{mname}", default=None)
            continue
        p.add_argument(f"--{name}", default=None)
    ns = p.parse_args()

    cfg = TrainConfig()
    overrides = {
        k: v for k, v in vars(ns).items() if k != "resume" and v is not None
    }
    cfg.apply_overrides(overrides)
    # Keep the cosine schedule aligned with the run length unless set explicitly.
    if "lr_decay_iters" not in overrides:
        cfg.lr_decay_iters = cfg.max_iters
    return cfg, ns.resume, overrides


@torch.no_grad()
def estimate_loss(model, corpus, cfg: TrainConfig, device: str) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = corpus.get_batch(split, cfg.model.block_size, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    cfg, resume, overrides = parse_args()
    torch.manual_seed(cfg.seed)
    device = cfg.resolved_device()
    out_dir = os.path.join(BASE_DIR, cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    tok_path = os.path.join(out_dir, "tokenizer.json")

    # --- data + tokenizer ---------------------------------------------------
    text = read_text(cfg.data_path, BASE_DIR)

    start_iter = 0
    best_val = float("inf")
    if resume and os.path.exists(ckpt_path):
        print(f"resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        cfg = TrainConfig.from_dict(ckpt["config"])
        cfg.lr_decay_iters = cfg.lr_decay_iters or cfg.max_iters
        tokenizer = tokenizer_from_dict(ckpt["tokenizer"])
        start_iter = ckpt["iter"] + 1
        best_val = ckpt.get("best_val", float("inf"))

        # Re-apply CLI overrides on top of the checkpoint's config, EXCEPT model
        # architecture (those are fixed — the saved weights must still fit).
        model_fields = {f.name for f in fields(GPTConfig)}
        arch = [k for k in overrides if k in model_fields]
        if arch:
            print(f"[resume] ignoring architecture flags {arch} (fixed by checkpoint)")
        train_overrides = {k: v for k, v in overrides.items() if k not in model_fields}
        if train_overrides:
            cfg.apply_overrides(train_overrides)
            print(f"[resume] applied overrides {list(train_overrides)}")
        # Extend the LR schedule to the new run length unless set explicitly.
        if "max_iters" in train_overrides and "lr_decay_iters" not in train_overrides:
            cfg.lr_decay_iters = cfg.max_iters
    else:
        tokenizer = build_tokenizer(cfg.tokenizer, text)
        cfg.model.vocab_size = tokenizer.vocab_size

    corpus = build_corpus(text, tokenizer, cfg.val_fraction)
    print(
        f"device={device}  tokenizer={cfg.tokenizer}  vocab={cfg.model.vocab_size}  "
        f"train_tokens={len(corpus.train)}  val_tokens={len(corpus.val)}"
    )

    # --- model + optimiser --------------------------------------------------
    model = NanoGPT(cfg.model).to(device)
    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2)
    )
    if resume and os.path.exists(ckpt_path):
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    if cfg.compile:
        model = torch.compile(model)
    print(f"parameters: {model.num_params()/1e6:.2f}M")

    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if use_amp
        else torch.autocast("cpu", enabled=False)
    )

    logger = CsvLogger(os.path.join(out_dir, "loss.csv"))
    save_tokenizer(tokenizer, tok_path)

    # --- training loop ------------------------------------------------------
    model.train()
    for it in range(start_iter, cfg.max_iters):
        lr = cosine_lr(
            it,
            learning_rate=cfg.learning_rate,
            warmup_iters=cfg.warmup_iters,
            lr_decay_iters=cfg.lr_decay_iters,
            min_lr=cfg.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        # Gradient accumulation: sum grads over several micro-batches.
        optimizer.zero_grad(set_to_none=True)
        last_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = corpus.get_batch("train", cfg.model.block_size, cfg.batch_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            last_loss += loss.item()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if it % cfg.log_interval == 0:
            log_training_step(it, last_loss, lr)

        if it > 0 and it % cfg.eval_interval == 0:
            losses = estimate_loss(model, corpus, cfg, device)
            print(f"  eval  train {losses['train']:.4f}  val {losses['val']:.4f}")
            logger.log(it, losses["train"], losses["val"], lr)
            if losses["val"] < best_val:
                best_val = losses["val"]
                _save(ckpt_path, model, optimizer, cfg, tokenizer, it, best_val)
                print(f"  saved best checkpoint (val {best_val:.4f})")

    # Final eval + save so a short run still produces a usable checkpoint.
    losses = estimate_loss(model, corpus, cfg, device)
    logger.log(cfg.max_iters, losses["train"], losses["val"], cfg.min_lr)
    if losses["val"] <= best_val or not os.path.exists(ckpt_path):
        best_val = min(best_val, losses["val"])
        _save(ckpt_path, model, optimizer, cfg, tokenizer, cfg.max_iters - 1, best_val)
    print(f"done. best val loss {best_val:.4f}. checkpoint: {ckpt_path}")


def _save(path, model, optimizer, cfg, tokenizer, it, best_val) -> None:
    raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
    torch.save(
        {
            "model": raw.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "tokenizer": tokenizer.to_dict(),
            "iter": it,
            "best_val": best_val,
        },
        path,
    )


if __name__ == "__main__":
    main()
