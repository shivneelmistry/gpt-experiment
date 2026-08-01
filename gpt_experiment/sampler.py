"""Sampling strategies."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from gpt_experiment.model import LanguageModel
from gpt_experiment.tokenizer import CharTokenizer

__all__ = ["Sampler", "SamplingConfig"]

_NEG_INF = float("-inf")


@dataclass(frozen=True)
class SamplingConfig:
    """
    max_new_tokens  how many tokens to produce
    temperature     divides the scores. Below 1 sharpens, above 1 flattens
    top_k           keep only the k highest-scoring tokens
    top_p           keep the smallest set whose probabilities reach p
    seed            fixes the dice rolls
    """

    max_new_tokens: int = 200
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Below 1 widens the gaps between scores, above 1 closes them."""
    return logits / temperature


def apply_top_k(logits: Tensor, k: int) -> Tensor:
    """Keep the k highest scores, discard the rest."""
    k = min(k, logits.shape[-1])
    cutoff = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < cutoff, _NEG_INF)


def apply_top_p(logits: Tensor, p: float) -> Tensor:
    """Keep the smallest set of tokens whose probabilities reach p.

    Unlike top-k the size adapts: confident steps keep one or two candidates,
    uncertain ones keep many.
    """
    ordered, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = F.softmax(ordered, dim=-1).cumsum(dim=-1)

    # shift so the token that crosses p is itself kept
    drop = cumulative - F.softmax(ordered, dim=-1) >= p
    drop[..., 0] = False  # never filter everything away

    return logits.masked_fill(drop.scatter(-1, indices, drop), _NEG_INF)


class Sampler:
    """Generates text from a trained model."""

    def __init__(self, model: LanguageModel, tokenizer: CharTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def generate(self, prompt: str, config: SamplingConfig) -> str:
        """Continue prompt by config.max_new_tokens characters."""
        if config.seed is not None:
            torch.manual_seed(config.seed)

        self.model.eval()
        device = next(self.model.parameters()).device
        block_size = self.model.config.block_size

        ids = self.tokenizer.encode(prompt) or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        for _ in range(config.max_new_tokens):
            # positions past block_size have no embedding
            logits, _ = self.model(idx[:, -block_size:])
            next_id = self._sample(logits[:, -1, :], config)
            idx = torch.cat([idx, next_id], dim=1)

        return self.tokenizer.decode(idx[0].tolist())

    @staticmethod
    def _sample(logits: Tensor, config: SamplingConfig) -> Tensor:
        """(batch, vocab) scores -> (batch, 1) sampled ids."""
        logits = apply_temperature(logits, config.temperature)
        if config.top_k is not None:
            logits = apply_top_k(logits, config.top_k)
        if config.top_p is not None:
            logits = apply_top_p(logits, config.top_p)
        return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
