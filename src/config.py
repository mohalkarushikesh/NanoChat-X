"""Central configuration for NanoChat-X.

Everything a run needs lives in two small dataclasses so there is exactly one
place to look for a hyperparameter. `TrainConfig` embeds a `GPTConfig`; the CLI
in ``train.py`` overrides any field by name (e.g. ``--n_layer 6 --lr 6e-4``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, fields
from typing import Any


@dataclass
class GPTConfig:
    """Architecture of the causal transformer (a small GPT)."""

    vocab_size: int = 256      # filled in from the tokenizer before building
    block_size: int = 128      # max context length (also caps positional embeddings)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = True          # use bias in Linear/LayerNorm layers

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )


@dataclass
class TrainConfig:
    """Everything about a training run (model config included)."""

    model: GPTConfig = field(default_factory=GPTConfig)

    # data / tokenizer
    data_path: str = "data/data.txt"
    tokenizer: str = "char"        # "char" or "word"
    val_fraction: float = 0.1

    # optimisation
    batch_size: int = 32
    grad_accum_steps: int = 1      # effective batch = batch_size * grad_accum_steps
    max_iters: int = 5000
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # learning-rate schedule (warmup then cosine decay)
    warmup_iters: int = 100
    lr_decay_iters: int = 5000     # usually == max_iters
    min_lr: float = 3e-5

    # evaluation / checkpointing
    eval_interval: int = 250
    eval_iters: int = 50           # batches averaged per loss estimate
    log_interval: int = 50
    out_dir: str = "out"

    # runtime
    device: str = "auto"           # "auto" -> cuda if available else cpu
    seed: int = 1337
    compile: bool = False          # torch.compile (off by default for portability)

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    # --- (de)serialisation helpers used by the checkpoint --------------------
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        d = dict(d)
        model = d.pop("model", {})
        cfg = cls(**d)
        cfg.model = GPTConfig(**model)
        return cfg

    def apply_overrides(self, overrides: dict[str, Any]) -> None:
        """Set fields by name; keys matching a GPTConfig field go to ``model``."""
        model_fields = {f.name for f in fields(GPTConfig)}
        train_fields = {f.name for f in fields(TrainConfig)}
        for key, value in overrides.items():
            if key in model_fields:
                setattr(self.model, key, _coerce(getattr(self.model, key), value))
            elif key in train_fields and key != "model":
                setattr(self, key, _coerce(getattr(self, key), value))
            else:
                raise KeyError(f"Unknown config field: {key!r}")
        self.model.__post_init__()


def _coerce(current: Any, value: Any) -> Any:
    """Cast a CLI string to the type of the existing default."""
    if isinstance(value, str):
        if isinstance(current, bool):
            return value.lower() in ("1", "true", "yes", "y")
        if isinstance(current, int) and not isinstance(current, bool):
            return int(value)
        if isinstance(current, float):
            return float(value)
    return value
