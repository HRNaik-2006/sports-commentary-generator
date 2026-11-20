"""Tiny character language model with a from-scratch LoRA adapter."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .core import build_training_text, seed_examples


class LoRALinear(nn.Module):
    """Frozen linear projection plus trainable low-rank update (BA * alpha/r)."""

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale


class TinyCharLM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 48, rank: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.adapter = LoRALinear(embedding_dim, vocab_size, rank=rank)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.embedding(ids))


@dataclass
class TrainedModel:
    model: TinyCharLM
    stoi: dict[str, int]
    itos: list[str]
    training_loss: float


def train_lora(steps: int = 160, learning_rate: float = 0.025, rank: int = 4,
               examples: list[tuple[str, str]] | None = None, seed: int = 7) -> TrainedModel:
    """Train only LoRA parameters over a tiny character next-token objective."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    random.seed(seed)
    torch.manual_seed(seed)
    pairs = examples if examples is not None else seed_examples()
    text = build_training_text(pairs)
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    model = TinyCharLM(len(chars), rank=rank)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in (model.adapter.lora_A, model.adapter.lora_B):
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam([model.adapter.lora_A, model.adapter.lora_B], lr=learning_rate)
    seq_len = min(96, max(16, len(ids) // 8))
    loss_value = 0.0
    model.train()
    for _ in range(steps):
        starts = torch.randint(0, max(1, len(ids) - seq_len - 1), (16,))
        x = torch.stack([ids[s:s + seq_len] for s in starts])
        y = torch.stack([ids[s + 1:s + seq_len + 1] for s in starts])
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return TrainedModel(model.eval(), stoi, chars, loss_value)


def generate(model: TrainedModel, prompt: str, max_new_tokens: int = 180,
             temperature: float = 0.8, seed: int | None = None) -> str:
    """Sample locally from the trained model, with safe handling of unknown chars."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if seed is not None:
        torch.manual_seed(seed)
    stoi = model.stoi
    valid = [stoi[ch] for ch in prompt if ch in stoi]
    if not valid:
        valid = [0]
    ids = torch.tensor([valid], dtype=torch.long)
    generated = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model.model(ids[:, -256:])[:, -1, :]
            logits = logits / max(0.05, float(temperature))
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
            generated.append(model.itos[next_id])
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
            if generated[-1] == "\n" and len(generated) > 24:
                break
    result = "".join(generated).strip()
    return result or "The action is building, and every moment matters!"


def generate_commentary(prompt: str, model: TrainedModel | None = None,
                        temperature: float = 0.75, max_new_tokens: int = 180) -> str:
    """Friendly one-call interface; trains a small adapter when no model is supplied."""
    trained = model or train_lora()
    return generate(trained, prompt, max_new_tokens=max_new_tokens, temperature=temperature)

