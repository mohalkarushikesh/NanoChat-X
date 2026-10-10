# NanoChat-X: End-to-End Flow & Technologies

## Architecture Overview

```mermaid
graph TB
    subgraph DataLayer["📄 Data Layer"]
        A["Raw Text Data<br/>(data/data.txt)"]
        B["Preprocessing<br/>(preprocess_cornell.py)"]
    end
    
    subgraph Tokenization["🔤 Tokenization"]
        C["Tokenizer<br/>(CharTokenizer/<br/>WordTokenizer)"]
        D["Token IDs"]
    end
    
    subgraph Training["🧠 Model Training Pipeline"]
        E["Model Config<br/>(GPTConfig)"]
        F["NanoGPT Model"]
        G["Training Loop<br/>(train.py)"]
        H["Checkpoint<br/>(ckpt.pt)"]
    end
    
    subgraph Components["🔗 Model Components"]
        I["Token + Positional<br/>Embeddings"]
        J["Transformer Blocks<br/>N × repeat"]
        K["Causal Self-Attention"]
        L["MLP Feed-Forward"]
        M["LM Head<br/>weights tied"]
    end
    
    subgraph Optimization["⚡ Loss & Optimization"]
        N["Cross-Entropy Loss"]
        O["AdamW Optimizer<br/>+ Weight Decay"]
        P["Cosine LR Schedule<br/>+ Warmup"]
    end
    
    subgraph Eval["✅ Evaluation & Checkpointing"]
        Q["Train/Val Loss Tracking"]
        R["Best Checkpoint Saving"]
    end
    
    subgraph Inference["🚀 Inference & Generation"]
        S["Inference Module<br/>(inference.py)"]
        T["Sampling<br/>(temperature + top-k)"]
        U["Autoregressive Generation<br/>(generate.py)"]
    end
    
    subgraph UI["💬 User Interfaces"]
        V["CLI Chat<br/>(chat.py)"]
        W["Sample Generator<br/>(sample.py)"]
        X["Web UI<br/>(FastAPI + Vue)"]
    end
    
    subgraph API["🔌 API Layer"]
        Y["FastAPI Server<br/>(server.py)"]
        Z["REST Endpoints<br/>/api/health<br/>/api/generate"]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    D --> G
    F --> G
    G --> H
    G --> N
    N --> O
    O --> P
    P --> G
    G --> Q
    Q --> R
    R --> H
    H --> S
    S --> T
    S --> U
    U --> V
    U --> W
    S --> X
    X --> Y
    Y --> Z
    
    classDef dataStyle fill:#e1f5ff,stroke:#01579b,stroke-width:2px
    classDef tokenStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef trainStyle fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef compStyle fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef optStyle fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    classDef evalStyle fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px
    classDef inferStyle fill:#bbdefb,stroke:#0d47a1,stroke-width:2px
    classDef uiStyle fill:#f1f8e9,stroke:#558b2f,stroke-width:2px
    classDef apiStyle fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class DataLayer dataStyle
    class Tokenization tokenStyle
    class Training trainStyle
    class Components compStyle
    class Optimization optStyle
    class Eval evalStyle
    class Inference inferStyle
    class UI uiStyle
    class API apiStyle
```

## Technology Stack

### **Core ML/AI**
- **PyTorch** – Neural network framework, auto-differentiation, CUDA support
- **NumPy** – Numerical operations, array handling

### **Model Architecture**
- **Causal Transformer** (decoder-only, GPT-style)
  - Multi-head self-attention with causal masking
  - Pre-LayerNorm residual blocks
  - 4× MLP expansion
  - Weight-tied embeddings and LM head

### **Tokenization**
- **CharTokenizer** (default) – Character-level, fully reversible
- **WordTokenizer** – Word-level with `<unk>` fallback

### **Training & Optimization**
- **AdamW optimizer** with weight decay groups
- **Gradient clipping** and accumulation
- **Automatic Mixed Precision (AMP)** on CUDA
- **Warmup + Cosine LR decay** schedule
- **Distributed training-ready** checkpoint system

### **Data Handling**
- **Corpus** class – contiguous batch sampling from train/val split
- **CSV logging** for loss tracking and analysis

### **Web Stack (Optional)**
- **FastAPI** – Python async web framework
- **Uvicorn** – ASGI server
- **Pydantic** – Request/response validation
- **USWDS** – US Web Design System (CSS)
- **Vanilla JavaScript** – Client-side interaction (no build step)
- **Montserrat font** – System font fallback via USWDS

### **Testing & Development**
- **pytest** – Unit testing framework
- **Causal mask property tests** – No future-leak validation

## Project Structure

```
NanoChat-X/
├── src/                        # Core implementation
│   ├── model.py               # NanoGPT transformer architecture
│   ├── config.py              # GPTConfig, TrainConfig dataclasses
│   ├── tokenizer.py           # CharTokenizer, WordTokenizer
│   ├── data.py                # Corpus, batch sampling
│   ├── train.py               # Training loop, checkpoint save/resume
│   ├── inference.py           # Model loading for inference
│   ├── sample.py              # One-shot text generation
│   ├── chat.py                # Interactive prompt-completion loop
│   ├── server.py              # FastAPI web server
│   └── preprocess_cornell.py  # Dataset preprocessing
├── tests/
│   └── test_nanochat.py       # Shape, tokenizer, causal mask tests
├── web/                        # Single-page web console
│   ├── index.html             # Accessible HTML5 structure
│   ├── app.js                 # Fetch API, form handling, live updates
│   └── styles.css             # USWDS tokens, responsive grid
├── utils/
│   └── helpers.py             # Cosine LR, logging, CSV utilities
├── data/
│   ├── data.txt               # Training corpus
│   └── cornell_movie_dialogs/ # Optional: raw conversation data
├── requirements.txt           # torch, numpy, pytest, fastapi, uvicorn, pydantic
├── README.md
└── ARCHITECTURE.md
```

## Key Workflows

### 1. **Training**
```
Text Data → Tokenization → Token IDs → Batches → Forward Pass → Loss → Backprop → Weight Update → Checkpoint
```

### 2. **Generation (Inference)**
```
Checkpoint Load → Model + Tokenizer Init → Prompt Encode → Generate Loop (temp + top-k) → Decode → Output
```

### 3. **Web UI**
```
User Input → FastAPI Endpoint → Model.generate() → JSON Response → DOM Update → Display
```

## End-to-End Data Flow

```mermaid
sequenceDiagram
    participant User
    participant WebUI as Web UI<br/>(index.html)
    participant Server as FastAPI<br/>(server.py)
    participant Model as NanoGPT<br/>(model.py)
    participant Tokenizer as Tokenizer<br/>(tokenizer.py)

    User->>WebUI: Enter prompt + params
    WebUI->>Server: POST /api/generate
    Server->>Tokenizer: encode(prompt)
    Tokenizer-->>Server: token_ids[]
    Server->>Model: generate(token_ids, temp, top_k)
    Model->>Model: Causal attention forward pass
    Model->>Model: Sample next token (temp + top_k)
    Model-->>Server: generated_token_ids[]
    Server->>Tokenizer: decode(token_ids)
    Tokenizer-->>Server: text
    Server-->>WebUI: {prompt, completion, generated}
    WebUI->>WebUI: Update DOM
    WebUI-->>User: Display output
```

## Commands

```bash
# Training
python -m src.train --max_iters 3000 --n_layer 6
python -m src.train --tokenizer word --block_size 64
python -m src.train --resume  # continue from checkpoint

# Inference
python -m src.sample --prompt "The thing is " --max_new_tokens 200
python -m src.chat  # Interactive loop

# Web Server
python -m src.server --port 8000
python -m src.server --port 8080 --ckpt out/ckpt.pt

# Testing
python -m pytest -q

# Data Preprocessing (optional)
python -m src.preprocess_cornell  # Prepare Cornell Movie Dialogs
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train a model (generates out/ckpt.pt)
python -m src.train --max_iters 3000 --n_layer 6 --n_embd 256

# 3. Generate text
python -m src.sample --prompt "The thing is " --top_k 40

# 4. Or run the web UI
python -m src.server
# Open http://127.0.0.1:8000
```