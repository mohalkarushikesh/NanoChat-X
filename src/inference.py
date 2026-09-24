"""Shared checkpoint loading for sample.py and chat.py."""

from __future__ import annotations

import os

import torch

from src.config import TrainConfig
from src.model import NanoGPT
from src.tokenizer import tokenizer_from_dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_model(ckpt_path: str | None = None, device: str | None = None):
    """Return (model, tokenizer, device) from a training checkpoint."""
    if ckpt_path is None:
        ckpt_path = os.path.join(BASE_DIR, "out", "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"no checkpoint at {ckpt_path}. Train first: python -m src.train"
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = TrainConfig.from_dict(ckpt["config"])
    tokenizer = tokenizer_from_dict(ckpt["tokenizer"])

    model = NanoGPT(cfg.model).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer, device
