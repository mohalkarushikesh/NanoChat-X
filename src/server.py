"""FastAPI server for NanoChat-X.

Serves a static single-page web console (no Node build) and a small JSON API the
page calls to generate text from a trained checkpoint.

    python -m src.server                 # http://127.0.0.1:8000
    python -m src.server --port 8080 --ckpt out/ckpt.pt

The model is loaded once at startup (if a checkpoint exists) and reused for every
request. If there is no checkpoint yet, the server still runs and the UI shows a
"train a model first" message.
"""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.inference import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "web")

# Set from the CLI before uvicorn starts; read inside the lifespan handler.
_CKPT_PATH: Optional[str] = None


class GenerateRequest(BaseModel):
    prompt: str = Field(default="", max_length=2000)
    max_new_tokens: int = Field(default=200, ge=1, le=1000)
    temperature: float = Field(default=0.8, gt=0.0, le=2.0)
    top_k: int = Field(default=40, ge=0, le=1000)  # 0 disables top-k


class GenerateResponse(BaseModel):
    prompt: str
    completion: str
    generated: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the checkpoint once at startup; degrade gracefully if absent."""
    app.state.model = None
    app.state.tokenizer = None
    app.state.device = "cpu"
    app.state.error = None
    try:
        model, tokenizer, device = load_model(_CKPT_PATH)
        app.state.model = model
        app.state.tokenizer = tokenizer
        app.state.device = device
        app.state.config = model.config
        print(f"[server] model loaded on {device}, vocab={tokenizer.vocab_size}")
    except FileNotFoundError as exc:
        app.state.error = str(exc)
        print(f"[server] no checkpoint: {exc}")
    yield


app = FastAPI(
    title="NanoChat-X",
    version="1.0.0",
    description="Web console for a from-scratch causal GPT. Generates text from a "
    "locally trained checkpoint.",
    lifespan=lifespan,
)

if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/api/health")
def health() -> dict:
    loaded = app.state.model is not None
    info: dict = {"status": "ok" if loaded else "no_model", "model_loaded": loaded}
    if loaded:
        cfg = app.state.config
        info["device"] = app.state.device
        info["tokenizer"] = getattr(app.state.tokenizer, "kind", "unknown")
        info["vocab_size"] = app.state.tokenizer.vocab_size
        info["model"] = {
            "n_layer": cfg.n_layer,
            "n_head": cfg.n_head,
            "n_embd": cfg.n_embd,
            "block_size": cfg.block_size,
            "parameters": app.state.model.num_params(),
        }
    else:
        info["detail"] = app.state.error
    return info


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model available. Run `python -m src.train` first.",
        )
    model = app.state.model
    tokenizer = app.state.tokenizer
    device = app.state.device

    ids = tokenizer.encode(req.prompt)
    if not ids:
        ids = [0]  # empty/unknown prompt -> start from a single fallback token
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    top_k = req.top_k if req.top_k > 0 else None
    out = model.generate(
        idx,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=top_k,
    )
    full = tokenizer.decode(out[0].tolist())
    generated = tokenizer.decode(out[0].tolist()[len(ids):])
    return GenerateResponse(prompt=req.prompt, completion=full, generated=generated)


def main() -> None:
    global _CKPT_PATH
    p = argparse.ArgumentParser(description="Serve the NanoChat-X web console")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ckpt", default=None, help="checkpoint path (default out/ckpt.pt)")
    args = p.parse_args()
    _CKPT_PATH = args.ckpt

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
