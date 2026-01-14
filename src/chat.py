import torch
import os

from src.model import NanoTransformer
from src.tokenizer import encode, decode, stoi
from utils.helpers import generate_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "nanochat_model.pt")

# Use word-level vocabulary size from tokenizer
vocab_size = len(stoi)

model = NanoTransformer(vocab_size=vocab_size)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

if __name__ == "__main__":
    while True:
        user = input("You: ")
        response = generate_text(model, encode, decode, user, temperature=0.7, top_k=5)
        print("Nanochat:", response)
