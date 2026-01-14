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
python src/train.py
```

This trains the `NanoTransformer` on your dataset and saves a checkpoint (`nanochat_model.pt`).

### 4. Chat
Run:

```bash
python src/chat.py
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

---

## Training 

```
[Step 0] Loss: 11.3068
[Step 500] Loss: 7.0753
[Step 1000] Loss: 6.3511
[Step 1500] Loss: 6.7833
[Step 2000] Loss: 6.1445
[Step 2500] Loss: 6.5186
[Step 3000] Loss: 6.5293
[Step 3500] Loss: 6.3374
[Step 4000] Loss: 6.0313
[Step 4500] Loss: 6.2061
[Step 5000] Loss: 6.0675
[Step 5500] Loss: 4.2447
[Step 6000] Loss: 3.7475
[Step 6500] Loss: 2.8710
[Step 7000] Loss: 3.6409
[Step 7500] Loss: 2.0355
[Step 8000] Loss: 1.8431
[Step 8500] Loss: 1.6789
[Step 9000] Loss: 1.8349
[Step 9500] Loss: 1.1243
[Step 10000] Loss: 0.9294
[Step 10500] Loss: 1.0204
[Step 11000] Loss: 1.8780
[Step 11500] Loss: 1.5437
[Step 12000] Loss: 1.6548
[Step 12500] Loss: 1.1308
[Step 13000] Loss: 2.2530
[Step 13500] Loss: 0.9382
[Step 14000] Loss: 2.2658
[Step 14500] Loss: 1.3078
[Step 15000] Loss: 1.0895
[Step 15500] Loss: 0.7778
[Step 16000] Loss: 1.2873
[Step 16500] Loss: 1.2989
[Step 17000] Loss: 1.2744
[Step 17500] Loss: 1.3275
[Step 18000] Loss: 0.7972
[Step 18500] Loss: 1.0857
[Step 19000] Loss: 1.0296
[Step 19500] Loss: 1.1542
[Step 20000] Loss: 1.0182
[Step 20500] Loss: 0.8373
[Step 21000] Loss: 1.2232
[Step 21500] Loss: 0.8689
[Step 22000] Loss: 0.6415
[Step 22500] Loss: 0.8765
[Step 23000] Loss: 1.1951
[Step 23500] Loss: 1.2557
[Step 24000] Loss: 1.1097
[Step 24500] Loss: 0.5330
[Step 25000] Loss: 0.8944
[Step 25500] Loss: 0.9987
[Step 26000] Loss: 0.3881
[Step 26500] Loss: 1.0769
[Step 27000] Loss: 0.5387
[Step 27500] Loss: 0.7730
[Step 28000] Loss: 0.9052
[Step 28500] Loss: 0.9448
[Step 29000] Loss: 0.8574
[Step 29500] Loss: 0.7537
[Step 30000] Loss: 1.0162
[Step 30500] Loss: 1.2502
[Step 31000] Loss: 0.6818
[Step 31500] Loss: 0.7383
[Step 32000] Loss: 1.3212
[Step 32500] Loss: 1.3619
[Step 33000] Loss: 1.1126
[Step 33500] Loss: 1.2416
[Step 34000] Loss: 0.7426
[Step 34500] Loss: 0.9977
[Step 35000] Loss: 1.0706
[Step 35500] Loss: 0.8345
[Step 36000] Loss: 0.6686
[Step 36500] Loss: 0.9600
[Step 37000] Loss: 0.5366
[Step 37500] Loss: 1.5218
[Step 38000] Loss: 0.6323
[Step 38500] Loss: 1.4760
[Step 39000] Loss: 0.6996
[Step 39500] Loss: 0.8806
[Step 40000] Loss: 1.2258
[Step 40500] Loss: 0.6645
[Step 41000] Loss: 1.5808
[Step 41500] Loss: 1.0285
[Step 42000] Loss: 0.7025
[Step 42500] Loss: 0.7624
[Step 43000] Loss: 0.4374
[Step 43500] Loss: 0.3764
[Step 44000] Loss: 0.8847
[Step 44500] Loss: 1.3276
[Step 45000] Loss: 0.6864
[Step 45500] Loss: 0.4950
[Step 46000] Loss: 1.1148
[Step 46500] Loss: 1.3434
[Step 47000] Loss: 0.7819
[Step 47500] Loss: 0.6013
[Step 48000] Loss: 0.4945
[Step 48500] Loss: 0.9086
[Step 49000] Loss: 0.9562
[Step 49500] Loss: 0.6639

```
