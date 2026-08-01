"""Batching."""

from __future__ import annotations

from typing import Literal

import torch

from gpt_experiment.tokenizer import CharTokenizer

__all__ = ["Dataset", "Split"]

Split = Literal["train", "val"]


class Dataset:
    """Encoded corpus, split into train and val, sampled as random windows."""

    def __init__(
        self,
        text: str,
        tokenizer: CharTokenizer,
        train_frac: float = 0.9,
    ) -> None:
        if not 0.0 < train_frac < 1.0:
            raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        cut = int(len(data) * train_frac)
        self._splits: dict[str, torch.Tensor] = {
            "train": data[:cut],
            "val": data[cut:],
        }
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return sum(len(t) for t in self._splits.values())

    def size(self, split: Split) -> int:
        return len(self._splits[split])

    def get_batch(
        self,
        split: Split,
        batch_size: int,
        block_size: int,
        device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random windows and their next-character targets.

        Both (batch_size, block_size). y is x shifted one place, so every position
        is labelled by the character that follows it.
        """
        data = self._splits[split]
        if len(data) < block_size + 1:
            raise ValueError(
                f"{split} split has {len(data)} tokens, needs at least "
                f"{block_size + 1} for block_size={block_size}"
            )

        # high is exclusive, leaving room for y to reach one further than x
        starts = torch.randint(len(data) - block_size, (batch_size,))

        # stack copies; slicing alone would alias the corpus
        x = torch.stack([data[i : i + block_size] for i in starts])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])

        if device == "cpu":
            return x, y
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
