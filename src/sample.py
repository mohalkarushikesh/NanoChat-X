"""One-shot text generation from a trained checkpoint.

    python -m src.sample --prompt "The thing is " --max_new_tokens 200
    python -m src.sample --prompt "Hello" --temperature 0.8 --top_k 40 --num_samples 3
"""

from __future__ import annotations

import argparse

import torch

from src.inference import load_model


def main() -> None:
    p = argparse.ArgumentParser(description="Sample from NanoChat-X")
    p.add_argument("--ckpt", default=None, help="path to checkpoint (default out/ckpt.pt)")
    p.add_argument("--prompt", default="\n", help="starting text")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    model, tokenizer, device = load_model(args.ckpt)

    ids = tokenizer.encode(args.prompt)
    if not ids:  # empty/unknown prompt -> start from a single fallback token
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(args.num_samples):
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        text = tokenizer.decode(out[0].tolist())
        print(f"--- sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
