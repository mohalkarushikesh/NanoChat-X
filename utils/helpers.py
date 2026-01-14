import torch
import torch.nn.functional as F

def sample_next_token(logits, temperature=1.0, top_k=None):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k is not None:
        values, indices = torch.topk(probs, top_k)
        mask = torch.zeros_like(probs)
        mask[indices] = values
        probs = mask / mask.sum()

    return torch.multinomial(probs, num_samples=1).item()

def generate_text(model, encode, decode, prompt, max_new_tokens=50, temperature=1.0, top_k=None, stoi=None):
    device = next(model.parameters()).device
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        logits = model(idx)
        next_token = sample_next_token(logits[0, -1], temperature=temperature, top_k=top_k)
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        idx = torch.cat((idx, next_token_tensor), dim=1)

        # Optional: stop if EOS is generated
        if stoi and "<EOS>" in stoi and next_token == stoi["<EOS>"]:
            break

    return decode(idx[0].tolist())

def log_training_step(step, loss):
    print(f"[Step {step}] Loss: {loss:.4f}")
