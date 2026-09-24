"""Load the corpus, tokenize it once, and serve random contiguous batches.

The whole file is treated as one long token stream (a plain language model),
split into a train and a validation slice so we can watch for over-fitting.
"""

from __future__ import annotations

import os

import torch


def read_text(data_path: str, base_dir: str) -> str:
    path = data_path if os.path.isabs(data_path) else os.path.join(base_dir, data_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class Corpus:
    """Holds tokenized train/val tensors and hands out batches."""

    def __init__(self, ids: torch.Tensor, val_fraction: float):
        n_val = int(len(ids) * val_fraction)
        if n_val == 0 or n_val >= len(ids):
            # Degenerate (tiny) corpus: reuse the whole thing for both splits.
            self.train = ids
            self.val = ids
        else:
            self.train = ids[:-n_val]
            self.val = ids[-n_val:]

    def get_batch(self, split: str, block_size: int, batch_size: int, device: str):
        data = self.train if split == "train" else self.val
        high = len(data) - block_size
        if high <= 0:
            raise ValueError(
                f"corpus split '{split}' has {len(data)} tokens but block_size is "
                f"{block_size}; use a smaller block_size or more data"
            )
        ix = torch.randint(high, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        if device.startswith("cuda"):
            # Pinned async transfer keeps the GPU fed.
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


def build_corpus(text: str, tokenizer, val_fraction: float) -> Corpus:
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return Corpus(ids, val_fraction)
