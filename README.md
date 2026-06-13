# WORK IN PROGRESS

# NanoChat‑X 🧠💬 

A **minimal transformer chatbot**, inspired by Andrej Karpathy’s teaching style.  
NanoChat‑X is designed to help you understand every moving part of a GPT‑like model by building the lightest possible version, then gradually extending it.

---

## 📂 Project Structure

```
NanoChat-X/
 ├── data/
 │    ├── cornell_movie_dialogs/   # raw dataset files
 │    └── data.txt                 # preprocessed Q->A pairs
 ├── src/
 │    ├── model.py                 # NanoTransformer definition
 │    ├── tokenizer.py             # encode/decode functions
 │    ├── train.py                 # training loop
 │    ├── chat.py                  # interactive chat loop
 │    └── __init__.py
 ├── utils/
 │    └── helpers.py               # training logs, generation helpers
 ├── requirements.txt              # dependencies
 └── Readme.md                     # project documentation
```

---

## 🚀 Getting Started

### 1. Setup
Install Python (≥3.9 recommended) and PyTorch:

```bash
pip install torch
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Start with a tiny toy dataset in `data/data.txt`:

```
Hello -> Hi
How are you? -> I am fine
What is your name? -> I am Nanochat
Bye -> Goodbye
```

Later, you can preprocess the Cornell Movie Dialogs corpus into Q→A pairs and place it in `data/data.txt`.

### 3. Train
Run:

```bash
python -m src.train
```

This trains the `NanoTransformer` on your dataset and saves a checkpoint (`nanochat_model.pt`).

### 4. Chat
Run:

```bash
python -m src.chat
```

You’ll enter an interactive loop:

```
You: Hello
Nanochat: Hi
```

---

## 🧠 How It Works

- **Tokenizer**: Converts text into tokens (character‑level or word‑level).  
- **NanoTransformer**: A tiny Transformer encoder with embeddings, positional encodings, and a linear head.  
- **Training Loop**: Learns next‑token prediction using cross‑entropy loss.  
- **Chat Loop**: Generates responses autoregressively, sampling one token at a time.

---

## 🌱 Extensions

Once you’ve run the minimal version, you can extend NanoChat‑X by:
- Switching to **word‑level or BPE tokenization**.  
- Using the **Cornell Movie Dialogs dataset** for richer conversations.  
- Adding **causal masking** for proper autoregressive generation.  
- Increasing model size (more layers, heads, embeddings).  
- Adding `<SOS>` and `<EOS>` tokens for cleaner start/stop behavior.  
- Experimenting with **temperature** and **top‑k sampling** for more natural outputs.

---

## 🎯 Goal

NanoChat‑X is not meant to be a production chatbot.  
It’s a **learning project**: a hands‑on way to understand how GPT‑like models are built from scratch.

