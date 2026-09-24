"""Small, dependency-free training utilities: LR schedule and logging."""

from __future__ import annotations

import math


def cosine_lr(it: int, *, learning_rate: float, warmup_iters: int,
              lr_decay_iters: int, min_lr: float) -> float:
    """Linear warmup for ``warmup_iters`` then cosine decay down to ``min_lr``."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it >= lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)


def log_training_step(step: int, loss: float, lr: float | None = None) -> None:
    if lr is None:
        print(f"[step {step:>6}] loss {loss:.4f}")
    else:
        print(f"[step {step:>6}] loss {loss:.4f}  lr {lr:.2e}")


class CsvLogger:
    """Append (step, split, loss, lr) rows to a CSV for later plotting."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "w", encoding="utf-8") as f:
            f.write("step,train_loss,val_loss,lr\n")

    def log(self, step: int, train_loss: float, val_loss: float, lr: float) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(f"{step},{train_loss:.6f},{val_loss:.6f},{lr:.8f}\n")
