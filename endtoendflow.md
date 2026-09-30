This repo is a fully local, from-scratch GPT-style language model implemented in PyTorch. There is no database, no external ML API, and no remote model provider. The workflow is:

local text corpus -> tokenizer -> PyTorch model -> training loop -> checkpoint -> inference/generation -> web API/UI

Below is the end-to-end execution flow, traced file by file, class by class, function by function, and call by call.

1) Project entry points and setup

Primary files:
- requirements.txt
- src/train.py
- src/sample.py
- src/chat.py
- src/server.py
- src/inference.py
- src/__init__.py

What happens:
- requirements.txt declares the runtime stack: torch, numpy, pytest, fastapi, uvicorn, pydantic.
- Python entry points are:
  - python -m src.train
  - python -m src.sample
  - python -m src.chat
  - python -m src.server
- src/__init__.py is empty, so it does nothing.

Important: There is no app bootstrap file like main.py. The actual runtime starts at the __main__ guard inside each script.

2) Configuration layer

File:
- src/config.py

Key definitions:
- GPTConfig
- TrainConfig

GPTConfig:
- Holds architecture settings:
  - vocab_size
  - block_size
  - n_layer
  - n_head
  - n_embd
  - dropout
  - bias
- __post_init__ validates n_embd % n_head == 0.

TrainConfig:
- Holds all training-run settings:
  - data_path
  - tokenizer
  - val_fraction
  - batch_size
  - grad_accum_steps
  - max_iters
  - learning_rate
  - weight_decay
  - beta1 / beta2
  - grad_clip
  - warmup_iters
  - lr_decay_iters
  - min_lr
  - eval_interval
  - eval_iters
  - log_interval
  - out_dir
  - device
  - seed
  - compile

Utility methods:
- resolved_device() chooses CUDA if available, else CPU.
- to_dict() turns config into a dict for checkpointing.
- from_dict() reconstructs TrainConfig from saved checkpoint.
- apply_overrides() accepts CLI override dicts and maps them into model or training settings. This is how command-line flags like --n_layer and --lr map into live config.

This config is the central source of truth for the entire run.

3) Raw data and corpus building

Files:
- src/data.py
- data/data.txt
- data/cornell_movie_dialogs/movie_lines.txt
- data/cornell_movie_dialogs/movie_conversations.txt
- src/preprocess_cornell.py

Important functions:
- read_text(data_path, base_dir)
- Corpus.__init__(ids, val_fraction)
- Corpus.get_batch(split, block_size, batch_size, device)
- build_corpus(text, tokenizer, val_fraction)

Flow:
- read_text() resolves the requested file path, opens it, and returns the full contents as a UTF-8 string.
- build_corpus():
  - tokenizes the full text
  - converts token IDs to a torch long tensor
  - passes that tensor to Corpus
- Corpus.__init__:
  - splits the token stream into train and validation segments using val_fraction
  - if the data is too small, it reuses the same ids for both splits
- Corpus.get_batch():
  - chooses train or val data depending on split
  - computes a valid random starting index range
  - creates x and y windows of length block_size
  - x = tokens[t : t+block_size]
  - y = tokens[t+1 : t+1+block_size]
  - This is standard next-token prediction training: model sees x; target is the next token in y.
  - if on CUDA, it uses pinned memory and non_blocking transfer for speed
- This is the actual “dataset feeding” stage for training.

Optional dataset preparation:
- src/preprocess_cornell.py:
  - reads Cornell Movie Dialogues files
  - maps line IDs to text
  - associates consecutive lines in conversations
  - writes pairs like “input line -> target line”
  - saves them into data/data.txt
- This lets the model train on dialogue-like corpora instead of arbitrary text.

4) Tokenization layer

File:
- src/tokenizer.py

Classes:
- CharTokenizer
- WordTokenizer

Shared logic:
- each tokenizer stores:
  - stoi: token string -> id
  - itos: id -> token string
- vocab_size property returns len(stoi)

CharTokenizer:
- CharTokenizer.train(text):
  - chars = sorted(set(text))
  - creates a vocabulary where each unique character gets an index
- encode(s):
  - maps each character in the string to its token id, skipping unknown characters
- decode(ids):
  - joins characters back into text
- to_dict() saves {"kind": "char", "stoi": ...}

WordTokenizer:
- WordTokenizer.train(text):
  - builds vocabulary from unique whitespace-delimited words
  - reserves <unk> as 0
- encode(s):
  - tokenizes by spaces
  - unknown words map to <unk>
- decode(ids):
  - joins words with spaces

Factory functions:
- build_tokenizer(kind, text) chooses a tokenizer implementation
- tokenizer_from_dict(d) reconstructs tokenizer from saved JSON
- save_tokenizer(tok, path)
- load_tokenizer(path)

This matters because:
- Model training is built around a specific tokenizer vocabulary.
- Inference must load the exact same vocabulary the model was trained with, otherwise the model sees token ids it does not understand.
- The saved tokenizer is checkpointed in out/tokenizer.json.

5) Model architecture: handwritten transformer

File:
- src/model.py

Key classes:
- LayerNorm
- CausalSelfAttention
- MLP
- Block
- NanoGPT
- NanoTransformer alias

Model design:
This repo specifically avoids using nn.TransformerEncoder because the old version was bidirectional and leaked future tokens. The current implementation is a decoder-only causal transformer.

LayerNorm:
- Custom implementation using F.layer_norm
- Allows optional bias
- This is used in transformer residual blocks

CausalSelfAttention:
- __init__:
  - sets n_head and n_embd
  - head_dim = n_embd / n_head
  - creates c_attn projection: input -> 3 * n_embd for q, k, v
  - creates c_proj projection back to n_embd
  - sets dropout
  - builds a lower-triangular causal mask:
    - torch.tril(torch.ones(block_size, block_size))
    - registered as a buffer
- forward(x):
  - B, T, C = x.shape
  - q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
  - reshape to (B, n_head, T, head_dim)
  - compute attention scores:
    - att = (q @ k.transpose(-2, -1)) / sqrt(head_dim)
    - apply causal mask:
      - masked_fill(mask == 0, -inf)
    - softmax
    - dropout
  - attention output = att @ v
  - merge heads back to (B, T, C)
  - project through c_proj
  - residual output is passed upward

This is exactly the causal self-attention step:
- each token cannot attend to future tokens
- only itself and previous positions

MLP:
- Linear(n_embd -> 4*n_embd)
- GELU
- Linear(4*n_embd -> n_embd)
- dropout

Block:
- Pre-LN residual block:
  - x = x + attn(ln_1(x))
  - x = x + mlp(ln_2(x))

NanoGPT:
- __init__:
  - stores config
  - creates transformer ModuleDict with:
    - wte: token embedding
    - wpe: positional embedding
    - drop: dropout
    - h: stack of transformer blocks
    - ln_f: final layer norm
  - creates lm_head = nn.Linear(n_embd, vocab_size, bias=False)
  - ties weights so wte.weight = lm_head.weight
  - applies weight initialization
  - special scaling for c_proj weights according to GPT-2 practice
- _init_weights():
  - normal init for Linear and Embedding weights
- num_params():
  - totals parameters
- forward(idx, targets=None):
  - B, T = idx.shape
  - ensure T <= block_size
  - build positional ids torch.arange(T)
  - tok_emb = wte(idx)
  - pos_emb = wpe(pos)
  - x = tok_emb + pos_emb
  - x = dropout(x)
  - pass through all blocks
  - apply final LayerNorm
  - logits = lm_head(x)
  - if targets is given:
    - compute cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-1)
  - return logits, loss
- configure_optimizers(weight_decay, learning_rate, betas):
  - builds AdamW optimizer
  - separates weight decay for matrices vs bias-like params
  - returns optimizer
- generate(idx, max_new_tokens, temperature=1.0, top_k=None, eos_id=None):
  - sets model to eval
  - loop for max_new_tokens:
    - idx_cond = idx[:, -block_size:]
    - logits, _ = self(idx_cond)
    - take logits for last token only
    - divide by temperature
    - optional top-k filtering
    - softmax to probabilities
    - sample one token via torch.multinomial
    - append to sequence
    - if eos_id was reached and all sampled tokens are eos, stop
  - returns expanded token sequence
- This method is the actual generation loop used by sample.py, chat.py, and server.py.

This is the core: token embedding + positional embedding + causal attention + MLP + final LM head.

6) Training pipeline

File:
- src/train.py

Main function flow:
- parse_args():
  - creates argparse parser
  - defaults = TrainConfig()
  - for each config field, adds CLI flag like --n_layer, --batch_size, --learning_rate, etc.
  - parse args
  - apply overrides to default config
  - if lr_decay_iters not specified, set it to max_iters
  - returns cfg, resume flag, overrides

- main():
  - cfg, resume, overrides = parse_args()
  - torch.manual_seed(cfg.seed)
  - device = cfg.resolved_device()
  - out_dir = repo/out
  - os.makedirs(out_dir, exist_ok=True)
  - ckpt_path = out/out/ckpt.pt
  - tok_path = out/tokenizer.json

Then:
- data + tokenizer setup
  - text = read_text(cfg.data_path, BASE_DIR)
  - if resume and checkpoint exists:
    - loads checkpoint from ckpt.pt
    - reconstructs TrainConfig from checkpoint
    - loads tokenizer from checkpoint
    - restores start_iter and best_val
    - re-applies only non-architecture CLI overrides
    - if model architecture overrides are passed during resume, they are ignored because model size must match checkpoint
  - else:
    - tokenizer = build_tokenizer(cfg.tokenizer, text)
    - cfg.model.vocab_size = tokenizer.vocab_size
  - corpus = build_corpus(text, tokenizer, cfg.val_fraction)

Then:
- model + optimizer configuration
  - model = NanoGPT(cfg.model).to(device)
  - optimizer = model.configure_optimizers(...)
  - if resuming, load saved model and optimizer state
  - if cfg.compile: model = torch.compile(model)
  - print parameter count

Then:
- AMP setup
  - use_amp = device.startswith("cuda")
  - scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
  - autocast_ctx = autocast if CUDA else no-op CPU autocast

Then:
- logger = CsvLogger(out/loss.csv)
- save_tokenizer(tokenizer, tok_path)

Training loop:
- model.train()
- for it in range(start_iter, cfg.max_iters):
  - lr = cosine_lr(...)
  - set optimizer learning rate per parameter group
  - optimizer.zero_grad(set_to_none=True)
  - last_loss = 0.0
  - for _ in range(cfg.grad_accum_steps):
    - x, y = corpus.get_batch("train", block_size, batch_size, device)
    - with autocast:
      - _, loss = model(x, y)
      - loss = loss / grad_accum_steps
    - scaler.scale(loss).backward()
    - last_loss += loss.item()
  - if grad_clip > 0:
    - scaler.unscale_(optimizer)
    - torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
  - scaler.step(optimizer)
  - scaler.update()
  - if it % log_interval == 0:
    - log_training_step(it, last_loss, lr)
  - if it > 0 and it % eval_interval == 0:
    - losses = estimate_loss(model, corpus, cfg, device)
    - logger.log(...)
    - if val loss improves, save best checkpoint

What estimate_loss does:
- model.eval()
- for each split in ("train","val"):
  - evaluate eval_iters batches
  - get loss from model(x, y)
  - average them
- resume model.train()

Final save:
- after loop, estimate final train/val losses
- if final val loss is better than best_val or checkpoint absent, save to ckpt.pt
- print completion message

Checkpoint schema:
- model state dict
- optimizer state dict
- config
- tokenizer
- iter
- best_val

This is the persisted “trained model artifact”.

7) Training utility functions

File:
- utils/helpers.py

Definitions:
- cosine_lr(it,...):
  - linear warmup for warmup_iters
  - then cosine decay to min_lr
- log_training_step(step, loss, lr=None):
  - prints training log line
- CsvLogger:
  - writes step,train_loss,val_loss,lr into loss.csv

This is used only during training and monitoring, not by the model itself.

8) Inference and generation from checkpoints

Files:
- src/inference.py
- src/sample.py
- src/chat.py

src/inference.py:
- load_model(ckpt_path=None, device=None)
  - resolves default checkpoint: out/ckpt.pt
  - raises FileNotFoundError if missing
  - loads checkpoint with torch.load(map_location=device)
  - rebuilds config from checkpoint
  - rebuilds tokenizer from saved dict
  - creates new NanoGPT(cfg.model)
  - loads model weights
  - calls model.eval()
  - returns (model, tokenizer, device)

This is the common loader used by generation paths.

src/sample.py:
- CLI parser:
  - --ckpt
  - --prompt
  - --max_new_tokens
  - --temperature
  - --top_k
  - --num_samples
  - --seed
- torch.manual_seed(args.seed)
- load_model(...)
- tokenizer.encode(args.prompt)
- if prompt encodes to empty unknown array, fallback to [0]
- create tensor [ids] on device
- loop over sample count
  - call model.generate(...)
  - decode full output back to text
  - print sample

This is one-shot text completion.

src/chat.py:
- CLI parser:
  - --ckpt
  - --max_new_tokens
  - --temperature
  - --top_k
- load_model(...)
- enters interactive loop:
  - prompt user input
  - if exit or quit: break
  - prompt = f"{user} -> "
  - ids = tokenizer.encode(prompt) or [0]
  - idx = torch.tensor([ids], device=device)
  - out = model.generate(...)
  - completion = tokenizer.decode(out[0].tolist()[len(ids):])
  - reply = completion.split("->")[0].split("\n")[0].strip()
  - print only the reply portion
- This is a conversation-style interface where prompts are shaped like “You -> ” to encourage reply generation.

9) Web API and frontend

Files:
- src/server.py
- web/index.html
- web/app.js
- web/styles.css

Backend:
- FastAPI app
- startup via lifespan()
- app.state.model = None
- app.state.tokenizer = None
- app.state.device = "cpu"
- app.state.error = None
- try load_model(_CKPT_PATH)
- if checkpoint exists:
  - store model, tokenizer, device, config
  - print loaded info
- else:
  - keep server running but mark error

Routes:
- GET /:
  - returns web/index.html
- GET /api/health:
  - returns status, model_loaded, device, tokenizer kind, vocab_size, model metadata
  - specifically includes:
    - n_layer
    - n_head
    - n_embd
    - block_size
    - parameter count
- POST /api/generate:
  - accepts JSON:
    - prompt
    - max_new_tokens
    - temperature
    - top_k
  - validates request shape with Pydantic
  - if no model loaded:
    - raise HTTPException 503
  - encode the prompt
  - if empty, fallback to [0]
  - call model.generate(...)
  - decode prompt + generated continuation
  - return JSON:
    - prompt
    - completion
    - generated

Frontend:
- web/index.html is the UI shell
- app.js:
  - loadHealth() calls /api/health and updates the status card
  - form submit calls /api/generate
  - displays output and copy button
  - slider values are bound to live UI outputs
- It is a static frontend served from FastAPI, not a Node app.

So the end-user flow is:
- browser opens /
- JS fetches /api/health
- if model is present, UI is enabled
- submit prompt to /api/generate
- backend tokenizes prompt, calls model.generate
- backend decodes to text and returns generated result
- browser renders it

10) Exact execution sequence: end to end

Here is the real chain, in order:

- User runs: python -m src.train
- src/train.py executes main()
- parse_args() builds config from defaults and CLI overrides
- TrainConfig is created from src/config.py
- read_text() loads data/data.txt
- build_tokenizer() creates char or word tokenizer from text
- build_corpus() tokenizes and splits corpus into train/val
- NanoGPT(cfg.model) is instantiated from src/model.py
- model.configure_optimizers() returns AdamW
- training loop begins
- on each step:
  - corpus.get_batch() returns x, y
  - model.forward() computes token embedding + positional embedding + blocks + final logits
  - attention uses causal mask in CausalSelfAttention.forward
  - cross-entropy computes training loss
  - backprop + optimizer step
  - evaluate and save best checkpoint
- checkpoint gets written to out/ckpt.pt
- tokenizer gets saved to out/tokenizer.json
- loss log gets written to out/loss.csv

Then for generation:
- User runs: python -m src.sample or python -m src.chat or server starts
- src/inference.py load_model() loads checkpoint and tokenizer
- prompt is tokenized
- model.generate() loops token by token
- each token is sampled from softmax probabilities
- the sequence is extended and truncated to block_size
- decoded output is printed or returned via API
- in web mode:
  - FastAPI server receives POST /api/generate
  - it encodes prompt and calls model.generate
  - returns generated text as JSON
  - frontend displays it

11) Data flow, component connections

Core data flow:
- raw text file -> tokenizer -> integer token ids -> torch tensors
- tensor batch -> token embedding + positional embedding -> causal transformer blocks -> logits
- logits -> next-token distribution -> sampling -> appended token -> next iteration
- final output is reconstructed text by tokenizer.decode()

This repo also has:
- config serialization
- model checkpoint serialization
- tokenizer serialization
- CSV logging for losses

12) External services and database status

There are no external database calls, no ORM, no SQL, no Redis, no PostgreSQL, no MySQL, no remote LLM service, and no API key or auth flow.

The only “service” is:
- local FastAPI server
- local PyTorch model
- local file-backed checkpoint and dataset

So this is a fully offline, local-only GPT implementation.

13) Final output of the repo

The repository is designed to produce:
- trained model weights in out/ckpt.pt
- tokenizer in out/tokenizer.json
- loss plot data in out/loss.csv
- generated text from prompts via:
  - CLI sample generation
  - interactive chat
  - FastAPI web console
- static web UI served locally

In one sentence:
This repo is a compact, end-to-end educational GPT implementation where text is tokenized, fed through a causal transformer, trained with cross-entropy on the next token, checkpointed locally, and then reused for generation via CLI or a local web API.

If you want, I can turn this into a deeper call graph with exact function invocations and the “who calls whom” chain for every module, or I can produce a simpler architecture diagram with arrows between files.
