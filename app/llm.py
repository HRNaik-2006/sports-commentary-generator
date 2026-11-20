"""Small character model and handwritten low-rank adapter for match calls."""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .core import build_training_text, seed_examples


class CommentaryProjection(nn.Module):
    """A frozen score projection with a trainable, rank-limited correction.

    The two adapter matrices form a down-then-up path. Starting the up matrix
    at zero keeps the freshly initialized model's predictions unchanged until
    training begins.
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = nn.Linear(in_features, out_features)
        self.base.weight.requires_grad_(False)
        self.base.bias.requires_grad_(False)
        self.adapter_down = nn.Parameter(torch.empty(rank, in_features))
        self.adapter_up = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.adapter_down, a=5**0.5)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_rank_update = F.linear(F.linear(x, self.adapter_down), self.adapter_up)
        return self.base(x) + low_rank_update * self.scale

    @property
    def lora_A(self) -> nn.Parameter:
        """Legacy accessor retained for the original drill's public API."""
        return self.adapter_down

    @property
    def lora_B(self) -> nn.Parameter:
        """Legacy accessor retained for the original drill's public API."""
        return self.adapter_up


class PlayByPlayLanguageModel(nn.Module):
    """Character-level next-token model used for the local training exercise."""
    def __init__(self, vocab_size: int, embedding_dim: int = 48, rank: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.adapter = CommentaryProjection(embedding_dim, vocab_size, rank=rank)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.embedding(ids))


@dataclass
class CommentaryModelSnapshot:
    model: PlayByPlayLanguageModel
    stoi: dict[str, int]
    itos: list[str]
    training_loss: float


def train_commentary_adapter(steps: int = 160, learning_rate: float = 0.025, rank: int = 4,
                             examples: list[tuple[str, str]] | None = None,
                             seed: int = 7) -> CommentaryModelSnapshot:
    """Fit the low-rank correction on event-to-call examples.

    The compact corpus is repeated as a next-character stream, making adapter
    behavior easy to inspect without downloading a base checkpoint.
    """
    if steps < 1:
        raise ValueError("steps must be at least 1")
    random.seed(seed)
    torch.manual_seed(seed)
    pairs = examples if examples is not None else seed_examples()
    text = build_training_text(pairs)
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    ids = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
    model = PlayByPlayLanguageModel(len(chars), rank=rank)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in (model.adapter.adapter_down, model.adapter.adapter_up):
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [model.adapter.adapter_down, model.adapter.adapter_up], lr=learning_rate
    )
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
    return CommentaryModelSnapshot(model.eval(), stoi, chars, loss_value)


def generate_play_call(model: CommentaryModelSnapshot, prompt: str,
                       max_new_tokens: int = 180, temperature: float = 0.8,
                       seed: int | None = None) -> str:
    """Sample a play call from the event prompt and the trained character model."""
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


def train_lora(*args, **kwargs) -> CommentaryModelSnapshot:
    """Compatibility name for callers from the first project iteration."""
    return train_commentary_adapter(*args, **kwargs)


def generate(model: CommentaryModelSnapshot, prompt: str, max_new_tokens: int = 180,
             temperature: float = 0.8, seed: int | None = None) -> str:
    """Compatibility name for the original sampling helper."""
    return generate_play_call(model, prompt, max_new_tokens, temperature, seed)


LoRALinear = CommentaryProjection
TinyCharLM = PlayByPlayLanguageModel
TrainedModel = CommentaryModelSnapshot


def generate_commentary(prompt: str, model: CommentaryModelSnapshot | None = None,
                        temperature: float = 0.75, max_new_tokens: int = 180) -> str:
    """Convenience entry point that trains an adapter if none was supplied."""
    trained = model or train_commentary_adapter()
    return generate_play_call(trained, prompt, max_new_tokens, temperature)

