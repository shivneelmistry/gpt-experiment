"""Hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ModelConfig", "TrainConfig"]


@dataclass(frozen=True)
class ModelConfig:
    """
    vocab_size  number of distinct tokens the model can read and emit (65)
    n_embd      numbers representing each token
    n_head      lookups per layer, learned traits
    n_kv_head   key/value heads, shared across query heads. None means one each
    n_layer     rounds of look-around-then-think
    block_size  how many tokens back it can see
    dropout     fraction of activations zeroed during training
    """

    vocab_size: int
    n_embd: int = 64
    n_head: int = 4
    n_kv_head: int | None = None
    n_layer: int = 4
    block_size: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head:
            raise ValueError(
                f"n_embd ({self.n_embd}) must divide evenly by "
                f"n_head ({self.n_head})"
            )
        if self.n_head % self.kv_heads:
            raise ValueError(
                f"n_head ({self.n_head}) must divide evenly by "
                f"n_kv_head ({self.kv_heads})"
            )
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @property
    def kv_heads(self) -> int:
        """Key/value heads in use. Equal to n_head unless GQA is configured."""
        return self.n_kv_head or self.n_head

    @property
    def queries_per_kv(self) -> int:
        """Query heads sharing each key/value head. 1 means plain multi-head."""
        return self.n_head // self.kv_heads


@dataclass(frozen=True)
class TrainConfig:
    """
    batch_size     sequences trained on at once
    learning_rate  nudge size
    max_iters      training steps
    eval_interval  steps between loss evaluations
    eval_iters     batches averaged per evaluation
    weight_decay   pull on weights toward zero
    warmup_iters   steps spent ramping the learning rate up from zero
    grad_clip      cap on total gradient size
    seed           makes runs reproducible
    device         "mps" for Apple GPU, "cpu" as fallback
    """

    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 3000
    eval_interval: int = 500
    eval_iters: int = 200
    weight_decay: float = 0.1
    warmup_iters: int = 100
    grad_clip: float = 1.0
    seed: int = 1337
    device: str = "mps"

    def __post_init__(self) -> None:
        if self.warmup_iters >= self.max_iters:
            raise ValueError(
                f"warmup_iters ({self.warmup_iters}) must be less than "
                f"max_iters ({self.max_iters})"
            )
