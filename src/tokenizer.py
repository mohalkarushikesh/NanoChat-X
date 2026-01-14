import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "data.txt")

# Read dataset
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

# Basic cleaning: lowercase and strip punctuation except sentence markers
def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-zA-Z0-9\s]", "", s)  # remove punctuation
    return s

text = clean_text(text)

# Split into words
words = text.split()

# Add special tokens
special_tokens = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
vocab = sorted(set(words))
vocab = special_tokens + vocab

# Build vocab dictionaries
stoi = {word: i for i, word in enumerate(vocab)}
itos = {i: word for word, i in stoi.items()}

def encode(s):
    """Convert a string into a list of word IDs."""
    s = clean_text(s)
    tokens = s.split()
    return [stoi.get(w, stoi["<UNK>"]) for w in tokens]

def decode(l):
    """Convert a list of word IDs back into a string."""
    words = [itos[i] for i in l if i in itos]
    return " ".join(words)
