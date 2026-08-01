"""Tests for config dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from gpt_experiment.config import ModelConfig, TrainConfig


def test_is_immutable() -> None:
    cfg = ModelConfig(vocab_size=65)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.n_embd = 128  # type: ignore[misc]


def test_head_size() -> None:
    assert ModelConfig(vocab_size=65, n_embd=64, n_head=4).head_size == 16


def test_uneven_head_split_rejected() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        ModelConfig(vocab_size=65, n_embd=64, n_head=5)


def test_bad_vocab_size_rejected() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        ModelConfig(vocab_size=0)


def test_warmup_longer_than_training_rejected() -> None:
    with pytest.raises(ValueError, match="less than"):
        TrainConfig(warmup_iters=5000, max_iters=3000)


def test_defaults_are_valid() -> None:
    assert ModelConfig(vocab_size=65).head_size == 16
    assert TrainConfig().device == "mps"
