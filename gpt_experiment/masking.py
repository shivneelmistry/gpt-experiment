"""Masked-language-model corruption."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["MaskingConfig", "apply_mlm_masking", "masked_accuracy", "unigram_accuracy"]

IGNORE_INDEX = -100  # cross_entropy skips these positions


@dataclass(frozen=True)
class MaskingConfig:
    """
    mask_prob    share of tokens selected as prediction targets
    replace_prob of those, the share swapped for [MASK]
    random_prob  of those, the share swapped for a random token
    """

    mask_prob: float = 0.15
    replace_prob: float = 0.8
    random_prob: float = 0.1

    def __post_init__(self) -> None:
        if not 0 < self.mask_prob <= 1:
            raise ValueError(f"mask_prob must be in (0, 1], got {self.mask_prob}")
        if self.replace_prob + self.random_prob > 1:
            raise ValueError(
                f"replace_prob + random_prob must not exceed 1, got "
                f"{self.replace_prob + self.random_prob}"
            )

    @property
    def keep_prob(self) -> float:
        """Selected tokens left untouched, but still predicted."""
        return 1.0 - self.replace_prob - self.random_prob


def apply_mlm_masking(
    tokens: Tensor,
    vocab_size: int,
    mask_token_id: int,
    config: MaskingConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Corrupt a batch, return (inputs, labels), both (batch, time).

    Labels are IGNORE_INDEX outside selected positions, so only those are scored.
    The 10% random and 10% untouched exist because [MASK] never appears at
    inference time.
    """
    config = config or MaskingConfig()

    selected = torch.rand(tokens.shape, device=tokens.device) < config.mask_prob
    labels = tokens.masked_fill(~selected, IGNORE_INDEX)

    inputs = tokens.clone()
    roll = torch.rand(tokens.shape, device=tokens.device)

    replace = selected & (roll < config.replace_prob)
    inputs[replace] = mask_token_id

    randomize = (
        selected
        & (roll >= config.replace_prob)
        & (roll < config.replace_prob + config.random_prob)
    )
    inputs[randomize] = torch.randint(
        vocab_size, (int(randomize.sum()),), device=tokens.device
    )

    # remainder stays as-is, still predicted
    return inputs, labels


def masked_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Share of masked positions reconstructed correctly.

    Over all positions it would be dominated by tokens visible in the input.
    """
    selected = labels != IGNORE_INDEX
    if not bool(selected.any()):
        return 0.0
    correct = logits.argmax(dim=-1)[selected] == labels[selected]
    return float(correct.float().mean())


def unigram_accuracy(tokens: Tensor, vocab_size: int) -> float:
    """Accuracy of always guessing the most frequent token. The floor to clear."""
    counts = torch.bincount(tokens.flatten(), minlength=vocab_size)
    return float(counts.max() / counts.sum())
