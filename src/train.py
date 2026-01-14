import torch
import torch.nn.functional as F
from torch import optim
import os

from src.model import NanoTransformer
from src.tokenizer import encode, decode, stoi, itos
from utils.helpers import log_training_step

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.txt")

text = open(DATA_PATH, "r", encoding="utf-8").read()

# Use word-level vocabulary size from tokenizer
vocab_size = len(stoi)

model = NanoTransformer(vocab_size=vocab_size)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

data = torch.tensor(encode(text), dtype=torch.long)
block_size = 64

# Optional: limit dataset size for faster experiments
# Uncomment the next line to use only the first 10000 tokens
# data = data[:10000]

for step in range(50000):  # increase steps for better training
    ix = torch.randint(len(data) - block_size, (4,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        log_training_step(step, loss.item())

torch.save(model.state_dict(), os.path.join(BASE_DIR, "nanochat_model.pt"))
