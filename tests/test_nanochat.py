"""Fast CPU tests for NanoChat-X. Run with:  python -m pytest -q  (from repo root)."""

import torch

from src.config import GPTConfig, TrainConfig
from src.model import NanoGPT
from src.tokenizer import (
    build_tokenizer,
    tokenizer_from_dict,
    CharTokenizer,
    WordTokenizer,
)
from src.data import build_corpus


def _tiny_config(vocab_size=16, block_size=8):
    return GPTConfig(
        vocab_size=vocab_size, block_size=block_size,
        n_layer=2, n_head=2, n_embd=16, dropout=0.0,
    )


def test_forward_shapes_and_loss():
    cfg = _tiny_config()
    model = NanoGPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (4, cfg.block_size))
    logits, loss = model(x, y)
    assert logits.shape == (4, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and loss.item() > 0


def test_causal_mask_no_future_leak():
    """Changing a *later* token must not change earlier positions' logits.

    This is the property the old bidirectional encoder violated.
    """
    cfg = _tiny_config()
    model = NanoGPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        base, _ = model(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 1) % cfg.vocab_size  # perturb only the last token
        changed, _ = model(x2)
    # Every position except the last must be identical.
    assert torch.allclose(base[0, :-1], changed[0, :-1], atol=1e-6)
    assert not torch.allclose(base[0, -1], changed[0, -1])


def test_generate_crops_context():
    """Generating past block_size must not raise (context is cropped)."""
    cfg = _tiny_config(block_size=8)
    model = NanoGPT(cfg).eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=20, top_k=4)
    assert out.shape[1] == 21  # 1 seed + 20 generated


def test_char_tokenizer_roundtrip():
    text = "Hello -> Hi\nBye -> Goodbye"
    tok = build_tokenizer("char", text)
    assert tok.decode(tok.encode(text)) == text
    # survives a save/load round-trip
    tok2 = tokenizer_from_dict(tok.to_dict())
    assert tok2.decode(tok2.encode(text)) == text


def test_word_tokenizer_unk():
    tok = WordTokenizer.train("hello there friend")
    ids = tok.encode("hello stranger")  # 'stranger' unseen -> <unk>
    assert tok.itos[ids[1]] == WordTokenizer.UNK


def test_loss_decreases_on_overfit():
    """A few steps on a tiny repeating corpus should reduce the loss."""
    torch.manual_seed(0)
    text = "the quick brown fox jumps over the lazy dog " * 50
    tok = build_tokenizer("char", text)
    cfg = _tiny_config(vocab_size=tok.vocab_size, block_size=16)
    model = NanoGPT(cfg)
    corpus = build_corpus(text, tok, val_fraction=0.1)
    opt = model.configure_optimizers(0.1, 1e-2, (0.9, 0.95))

    x, y = corpus.get_batch("train", cfg.block_size, 8, "cpu")
    _, first = model(x, y)
    for _ in range(50):
        x, y = corpus.get_batch("train", cfg.block_size, 8, "cpu")
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < first.item()


def test_config_overrides_and_roundtrip():
    cfg = TrainConfig()
    cfg.apply_overrides({"n_layer": "6", "learning_rate": "0.001"})  # CLI strings coerce
    assert cfg.model.n_layer == 6 and abs(cfg.learning_rate - 0.001) < 1e-9
    restored = TrainConfig.from_dict(cfg.to_dict())
    assert restored.model.n_layer == 6
