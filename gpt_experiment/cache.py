"""Key/value caching for generation."""

from __future__ import annotations

import torch
from torch import Tensor

from gpt_experiment.config import ModelConfig

__all__ = ["KVCache"]


class KVCache:
    """Stores past keys and values so generation stops recomputing them.

    Keys and values for existing tokens never change, yet an uncached decode
    recomputes all of them every step. Caching makes the per-token cost O(1) and
    the total O(n) instead of O(n^2).

    Preallocated rather than grown by concatenation, so no token pays for a
    reallocation and the latency curve stays flat.
    """

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (
            config.n_layer,
            batch_size,
            config.kv_heads,
            config.block_size,
            config.head_size,
        )
        self.keys = torch.zeros(shape, device=device, dtype=dtype)
        self.values = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0
        self._config = config

    @property
    def capacity(self) -> int:
        return self._config.block_size

    def bytes_allocated(self) -> int:
        """Memory reserved up front, occupied or not."""
        return self.keys.element_size() * self.keys.numel() * 2

    def bytes_used(self) -> int:
        """Memory actually holding tokens. Grows linearly with sequence length."""
        if self.capacity == 0:
            return 0
        return self.bytes_allocated() * self.length // self.capacity

    def append(self, layer: int, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Store this layer's new keys/values, return everything stored so far.

        k and v are (batch, n_head, new_tokens, head_size).
        """
        new_tokens = k.shape[2]
        if self.length + new_tokens > self.capacity:
            raise ValueError(
                f"cache holds {self.capacity} tokens, cannot add {new_tokens} "
                f"to the {self.length} already stored"
            )

        start, end = self.length, self.length + new_tokens
        self.keys[layer, :, :, start:end] = k
        self.values[layer, :, :, start:end] = v
        return self.keys[layer, :, :, :end], self.values[layer, :, :, :end]

    def advance(self, new_tokens: int) -> None:
        """Move the write position forward. Once per forward pass, not per layer."""
        self.length += new_tokens

    def rewind_to(self, length: int) -> None:
        """Move the write position back, keeping stored data.

        Lets a benchmark replay the same decode step without rebuilding the cache.
        """
        if not 0 <= length <= self.capacity:
            raise ValueError(f"length must be in 0..{self.capacity}, got {length}")
        self.length = length

    def reset(self) -> None:
        self.length = 0
