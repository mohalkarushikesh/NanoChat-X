"""Tokenizers written from scratch — no downloads, no external vocab files.

Two flavours:

* :class:`CharTokenizer` — one id per character. Tiny vocab, perfectly
  reversible (punctuation and the ``->`` separator survive), and the fastest
  path to coherent samples on a small corpus. This is the default.
* :class:`WordTokenizer` — one id per whitespace token, with ``<unk>`` for
  anything unseen. Bigger vocab, shorter sequences.

Both learn their vocabulary from the training text and can ``save``/``load`` it,
so ``chat.py`` and ``sample.py`` decode with the *exact* vocab the model was
trained on. (The previous version rebuilt the vocab from ``data.txt`` at import
time, which silently broke a checkpoint whenever the data changed.)
"""

from __future__ import annotations

import json
from typing import List


class CharTokenizer:
    kind = "char"

    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: ch for ch, i in stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def train(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        return cls(stoi)

    def encode(self, s: str) -> List[int]:
        # Unknown chars are skipped (char vocab covers all training chars).
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if i in self.itos)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "stoi": self.stoi}


class WordTokenizer:
    kind = "word"
    UNK = "<unk>"

    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: w for w, i in stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def train(cls, text: str) -> "WordTokenizer":
        words = sorted(set(text.split()))
        stoi = {cls.UNK: 0}
        for w in words:
            stoi[w] = len(stoi)
        return cls(stoi)

    def encode(self, s: str) -> List[int]:
        unk = self.stoi[self.UNK]
        return [self.stoi.get(w, unk) for w in s.split()]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i in self.itos)

    def to_dict(self) -> dict:
        return {"kind": self.kind, "stoi": self.stoi}


_REGISTRY = {"char": CharTokenizer, "word": WordTokenizer}


def build_tokenizer(kind: str, text: str):
    if kind not in _REGISTRY:
        raise ValueError(f"unknown tokenizer {kind!r}; choose from {list(_REGISTRY)}")
    return _REGISTRY[kind].train(text)


def tokenizer_from_dict(d: dict):
    cls = _REGISTRY[d["kind"]]
    # JSON keys are strings; stoi values (ids) load back as ints already.
    return cls(dict(d["stoi"]))


def save_tokenizer(tok, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tok.to_dict(), f, ensure_ascii=False)


def load_tokenizer(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return tokenizer_from_dict(json.load(f))
