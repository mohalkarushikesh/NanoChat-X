"""Interactive prompt-completion loop over a trained NanoChat-X checkpoint.

NanoChat-X is a plain language model, so "chat" here means: it continues from
your line. With the Cornell dialog data (formatted ``line -> reply``) prompting
with ``your text ->`` nudges it toward reply-shaped completions.

    python -m src.chat
"""

from __future__ import annotations

import argparse

import torch

from src.inference import load_model


def main() -> None:
    p = argparse.ArgumentParser(description="Chat with NanoChat-X")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    args = p.parse_args()

    model, tokenizer, device = load_model(args.ckpt)
    print("NanoChat-X ready. Type 'exit' or Ctrl-C to quit.\n")

    while True:
        try:
            user = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user.strip().lower() in {"exit", "quit"}:
            break

        prompt = f"{user} -> "
        ids = tokenizer.encode(prompt) or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        # Show only the freshly generated continuation.
        completion = tokenizer.decode(out[0].tolist()[len(ids):])
        # Trim at the next turn boundary if the model produced one.
        reply = completion.split("->")[0].split("\n")[0].strip()
        print("NanoChat:", reply if reply else completion.strip())


if __name__ == "__main__":
    main()
