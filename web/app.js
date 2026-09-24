"use strict";

// NanoChat-X web console — talks to the FastAPI JSON API in src/server.py.

const $ = (sel) => document.querySelector(sel);

// ---------------------------------------------------------------------------
// Range sliders -> live <output> values
// ---------------------------------------------------------------------------
function bindRange(id, fmt) {
  const input = document.getElementById(id);
  const out = document.getElementById(id + "-out");
  if (!input || !out) return;
  const update = () => (out.textContent = fmt(input.value));
  input.addEventListener("input", update);
  update();
}
bindRange("max_new_tokens", (v) => v);
bindRange("temperature", (v) => Number(v).toFixed(2));
bindRange("top_k", (v) => v);

// ---------------------------------------------------------------------------
// Model status + facts (from /api/health)
// ---------------------------------------------------------------------------
async function loadHealth() {
  const status = $("#model-status");
  const text = status.querySelector(".model-status__text");
  const facts = $("#model-facts");
  const btn = $("#generate-btn");
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.model_loaded) {
      const m = data.model;
      status.classList.add("is-ok");
      text.textContent =
        `Model ready · ${(m.parameters / 1e6).toFixed(2)}M params · ${data.device.toUpperCase()}`;
      facts.innerHTML = `
        <li><strong>Tokenizer:</strong> ${data.tokenizer} (vocab ${data.vocab_size})</li>
        <li><strong>Layers:</strong> ${m.n_layer} &nbsp;·&nbsp; <strong>Heads:</strong> ${m.n_head} &nbsp;·&nbsp; <strong>Embedding:</strong> ${m.n_embd}</li>
        <li><strong>Context window:</strong> ${m.block_size} tokens</li>
        <li><strong>Parameters:</strong> ${m.parameters.toLocaleString()}</li>
        <li><strong>Device:</strong> ${data.device}</li>`;
    } else {
      status.classList.add("is-error");
      text.textContent = "No trained model found — run training first";
      btn.disabled = true;
      showAlert(
        "error",
        "No checkpoint found. Train a model with <code>python -m src.train</code>, then restart the server."
      );
    }
  } catch (err) {
    status.classList.add("is-error");
    text.textContent = "Cannot reach the server";
  }
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------
function showAlert(kind, html) {
  $("#alert-slot").innerHTML =
    `<div class="usa-alert usa-alert--${kind}" role="alert">${html}</div>`;
}
function clearAlert() {
  $("#alert-slot").innerHTML = "";
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------
const form = $("#generate-form");
const output = $("#output");
const copyBtn = $("#copy-btn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearAlert();

  const btn = $("#generate-btn");
  const payload = {
    prompt: $("#prompt").value,
    max_new_tokens: Number($("#max_new_tokens").value),
    temperature: Number($("#temperature").value),
    top_k: Number($("#top_k").value),
  };

  btn.disabled = true;
  const originalLabel = btn.textContent;
  btn.textContent = "Generating…";
  output.classList.add("is-loading");
  output.innerHTML = `<span class="spinner" aria-hidden="true"></span>Generating text…`;
  copyBtn.hidden = true;

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const detail = await res.json().catch(() => ({}));
      throw new Error(detail.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();
    renderOutput(data.prompt, data.generated);
    copyBtn.hidden = false;
  } catch (err) {
    output.textContent = "";
    showAlert("error", escapeHtml(err.message));
  } finally {
    output.classList.remove("is-loading");
    btn.disabled = false;
    btn.textContent = originalLabel;
  }
});

function renderOutput(prompt, generated) {
  output.innerHTML =
    `<span class="prompt-text">${escapeHtml(prompt)}</span>` +
    `<span class="gen-text">${escapeHtml(generated)}</span>`;
}

// Clear
$("#clear-btn").addEventListener("click", () => {
  $("#prompt").value = "";
  output.innerHTML = `<p class="output__placeholder">Generated text will appear here.</p>`;
  copyBtn.hidden = true;
  clearAlert();
  $("#prompt").focus();
});

// Copy
copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(output.textContent);
    copyBtn.textContent = "Copied!";
    setTimeout(() => (copyBtn.textContent = "Copy"), 1500);
  } catch (_) {
    /* clipboard unavailable */
  }
});

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

loadHealth();
