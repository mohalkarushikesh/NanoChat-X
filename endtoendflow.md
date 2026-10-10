## What this is

NanoChat-X is a **minimal, from-scratch causal transformer** (decoder-only GPT) built in PyTorch for educational purposes. It trains on plain text files with no pretrained weights or external dependencies, making every component visible and learnable. The model includes a built-in web UI for interactive text generation and supports multiple tokenization strategies.

### Stack
- **Language(s):** Python (66%), JavaScript (8%), HTML (10%), CSS (15%)
- **Framework / runtime:** PyTorch + FastAPI + Uvicorn
- **Notable libraries:** PyTorch (model & training), FastAPI (web server), Pydantic (API validation), NumPy (tensor ops)

---

## How it's organized

```
NanoChat-X/
├── src/
│   ├── model.py          Core transformer architecture (NanoGPT class)
│   ├── tokenizer.py      CharTokenizer & WordTokenizer implementations
│   ├── config.py         GPTConfig & TrainConfig dataclasses
│   ├── data.py           Corpus class for train/val batching
│   ├── train.py          Training loop with AdamW, warmup, cosine LR
│   ├── sample.py         One-shot text generation from checkpoint
│   ├── chat.py           Interactive prompt-reply loop
│   ├── server.py         FastAPI web server + JSON API
│   ├── inference.py      Shared checkpoint loading
│   └── preprocess_cornell.py  Optional data preprocessing
├── web/
│   ├── index.html        USWDS-styled SPA with controls
│   ├── app.js            Frontend logic (API calls, slider binding)
│   └── styles.css        USWDS design tokens + custom styling
├── tests/
│   └── test_nanochat.py  Pytest suite (shapes, causal mask, loss decrease)
├── data/
│   ├── data.txt          Default training corpus
│   └── cornell_movie_dialogs/  Optional conversation data
└── utils/
    └── helpers.py        LR schedule, logging, CSV logger
```

## How it fits together

**The end-to-end data flow:**

1. **Input → Tokenization** (`tokenizer.py`): Raw text is converted to token IDs via `CharTokenizer.encode()` (character-level, fully reversible) or `WordTokenizer.encode()` (word-level with `<unk>` fallback).

2. **Token IDs → Embeddings** (`model.py`, `NanoGPT.forward`): Each token ID is looked up in `wte` (token embedding), positions are looked up in `wpe` (positional embedding), and the two are summed and passed through dropout.

3. **Embeddings → Transformer blocks** (`model.py`): The combined embedding flows through N transformer blocks. Each block applies:
   - `LayerNorm` → `CausalSelfAttention` (multi-head, causal mask blocks future attention) → residual add
   - `LayerNorm` → `MLP` (4× feed-forward) → residual add

4. **Transformer output → Logits** (`model.py`): A final `LayerNorm` is applied, then logits are projected via the weight-tied `lm_head` (sharing weights with `wte`).

5. **Training split** (`train.py`):
   - **Loss**: Cross-entropy between predicted logits and ground-truth next token, optimized with AdamW (weight decay on 2D+ params only)
   - **Schedule**: Warmup for `warmup_iters`, then cosine decay to `min_lr`
   - **Stability**: Gradient clipping, gradient accumulation, mixed precision (auto on CUDA)
   - **Checkpointing**: Best validation checkpoint saved; resume-capable

6. **Generation split** (`model.py`, `generate()`):
   - Start with prompt tokens; autoregressively sample one token per step
   - Apply temperature scaling, top-k filtering, softmax, and multinomial sampling
   - Append new token, crop context to `block_size`, loop

7. **Inference paths**:
   - **CLI sample** (`sample.py`): Load checkpoint, encode prompt, generate, decode output
   - **CLI chat** (`chat.py`): Interactive loop; formats prompt as `"<user> -> "` to nudge Cornell-trained models toward replies
   - **Web server** (`server.py` + `web/`): FastAPI loads checkpoint at startup; frontend posts JSON to `/api/generate`, gets `{prompt, completion, generated}` back

---

## How to run it

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Train (creates out/ckpt.pt + out/tokenizer.json + out/loss.csv)
python -m src.train                                      # char tokenizer, defaults
python -m src.train --max_iters 3000 --n_layer 6 --n_embd 256
python -m src.train --tokenizer word --block_size 64
python -m src.train --resume                             # continue from checkpoint

# 2. Generate from CLI
python -m src.sample --prompt "The thing is " --max_new_tokens 200 --top_k 40
python -m src.sample --num_samples 3 --temperature 0.5

# 3. Chat interactively
python -m src.chat                                       # prompts with " -> " suffix

# 4. Launch web UI
python -m src.server                                     # http://127.0.0.1:8000
python -m src.server --port 8080 --ckpt out/ckpt.pt

# 5. Run tests
python -m pytest -q
```

---

## End-to-End Flow Diagram

```
USER INPUT
    ↓
[Tokenization Phase]
  text → CharTokenizer.encode() / WordTokenizer.encode()
         → token IDs (1D array)
    ↓
[Training Phase]
  Token IDs → (Embedding + Positional) → Dropout
             → N × Transformer Blocks
                - LayerNorm → CausalSelfAttention → +residual
                - LayerNorm → MLP → +residual
             → Final LayerNorm → LM Head (tied weights)
             → Logits [B, T, vocab_size]
             ↓
          Cross-entropy loss vs. ground-truth next token
             ↓
          AdamW optimizer (weight-decay groups)
             ↓
          Warmup + Cosine LR schedule
             ↓
          Gradient clipping + accumulation
             ↓
          Checkpoint save (if best validation loss)
    ↓
[Inference Phase]
  1. Load checkpoint (model + tokenizer + config)
  2. Encode prompt → token IDs
  3. Initialize sequence with prompt tokens
  4. For each new token:
     - Feed context (last block_size tokens) to model
     - Get logits for last position
     - Apply temperature scaling
     - Apply top-k filtering (if top_k > 0)
     - Sample via multinomial(softmax(logits))
     - Append token, repeat until max_new_tokens reached
  5. Decode token IDs → text
    ↓
[Output Delivery]
  CLI:  print text directly (sample.py, chat.py)
  Web:  POST /api/generate → JSON {prompt, completion, generated} → render in HTML
```

---

## Key Functions & Methods

### Core Model (`src/model.py`)

| Symbol | Purpose |
|--------|---------|
| `NanoGPT.__init__()` | Build embedding layers, blocks, LM head; apply weight tying & GPT-2 scaled init |
| `NanoGPT.forward(idx, targets)` | Embed tokens + positions, run blocks, compute logits & cross-entropy loss |
| `NanoGPT.generate(idx, max_new_tokens, temperature, top_k)` | Autoregressively sample tokens with context cropping |
| `NanoGPT.configure_optimizers()` | Return AdamW with weight-decay groups (2D+ only) |
| `CausalSelfAttention.forward()` | Multi-head scaled dot-product attention with lower-triangular causal mask |
| `Block.forward()` | Pre-LayerNorm residuals: `x + attn(ln(x))` and `x + mlp(ln(x))` |
| `MLP.forward()` | Linear(4×) → GELU → Linear; return with dropout |

### Tokenization (`src/tokenizer.py`)

| Symbol | Purpose |
|--------|---------|
| `CharTokenizer.train(text)` | Extract unique characters, build vocab |
| `CharTokenizer.encode(s)` | Map each character to its token ID |
| `CharTokenizer.decode(ids)` | Reverse: token IDs → original characters |
| `WordTokenizer.train(text)` | Split on whitespace, build vocab with `<unk>` fallback |
| `build_tokenizer(kind, text)` | Factory to select & train tokenizer by name |
| `save_tokenizer()` / `load_tokenizer()` | JSON persistence (so inference uses exact training vocab) |

### Training (`src/train.py`)

| Symbol | Purpose |
|--------|---------|
| `parse_args()` | CLI parsing; every GPTConfig & TrainConfig field becomes a flag |
| `main()` | Load/build corpus, model, optimizer; training loop with checkpointing |
| `estimate_loss()` | Evaluate on train/val splits over `eval_iters` batches |
| `_save()` | Checkpoint: model state, optimizer state, config, tokenizer, iteration, best_val |

### Data (`src/data.py`)

| Symbol | Purpose |
|--------|---------|
| `Corpus.__init__()` | Split tokenized IDs into train/val by fraction |
| `Corpus.get_batch()` | Sample random contiguous chunks; pin memory on CUDA |

### Inference & Serving

| File | Key Functions |
|------|---|
| `inference.py` | `load_model(ckpt_path)` — deserialize checkpoint, build model, return (model, tokenizer, device) |
| `sample.py` | `main()` — one-shot generation from CLI flags |
| `chat.py` | `main()` — interactive loop with `" -> "` suffix formatting |
| `server.py` | `lifespan()` — startup model loading; `health()` — model info; `generate()` — API endpoint |

### Web UI

| File | Role |
|------|------|
| `index.html` | USWDS structure; sliders (temperature, top-k, max_tokens); output display |
| `app.js` | Fetch `/api/health` on load; bind slider outputs; POST to `/api/generate`; render & copy text |
| `styles.css` | USWDS design tokens, Montserrat font fallback, accessible focus states |

### Testing (`tests/test_nanochat.py`)

| Test | Purpose |
|------|---------|
| `test_forward_shapes_and_loss()` | Logits shape matches (B, T, vocab), loss is scalar > 0 |
| `test_causal_mask_no_future_leak()` | Perturbing position t must not change positions < t |
| `test_generate_crops_context()` | Generation past block_size does not raise |
| `test_char_tokenizer_roundtrip()` | Encode then decode recovers original text |
| `test_word_tokenizer_unk()` | Unknown words map to `<unk>` token |
| `test_loss_decreases_on_overfit()` | A few optimizer steps reduce loss on tiny corpus |
| `test_config_overrides_and_roundtrip()` | Config CLI coercion and dict serialization |

---

## Try asking

1. **"How does causal self-attention work in this model and why is it crucial for autoregressive generation?"**
   — See `CausalSelfAttention.forward()` in `src/model.py` lines 41–78; the lower-triangular mask prevents attending to future tokens.

2. **"What happens during checkpoint save and how does the resume feature preserve the training state?"**
   — See `_save()` in `src/train.py` lines 189–201 and resume logic in `main()` lines 83–104; saves model weights, optimizer state, config, tokenizer, and iteration counter.

3. **"How do the character and word tokenizers differ, and why does saving/loading the tokenizer matter for inference?"**
   — See `src/tokenizer.py` lines 23–79; CharTokenizer is reversible (every training char survives), WordTokenizer has a vocab size but needs `<unk>`. Saving ensures inference uses the *exact* training vocab, not a rebuilt one from `data.txt`.