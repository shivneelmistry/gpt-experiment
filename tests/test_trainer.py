"""Tests for the training loop."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpt_experiment.config import ModelConfig, TrainConfig
from gpt_experiment.data import Dataset
from gpt_experiment.model import BigramLM
from gpt_experiment.tokenizer import CharTokenizer
from gpt_experiment.trainer import Trainer

TEXT = "".join(chr(ord("a") + (i * 7) % 26) for i in range(20_000))


def make_trainer(**overrides: object) -> Trainer:
    settings: dict[str, object] = {
        "batch_size": 8,
        "max_iters": 60,
        "eval_interval": 30,
        "eval_iters": 5,
        "warmup_iters": 10,
        "device": "cpu",
    }
    settings.update(overrides)
    train_config = TrainConfig(**settings)  # type: ignore[arg-type]

    tokenizer = CharTokenizer(TEXT)
    dataset = Dataset(TEXT, tokenizer)
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=16)

    # Before the model, so weight init is seeded too.
    torch.manual_seed(train_config.seed)
    return Trainer(BigramLM(model_config), dataset, model_config, train_config)


def test_loss_decreases() -> None:
    trainer = make_trainer(max_iters=300, eval_interval=299, learning_rate=0.1)
    history = trainer.train()
    assert history[-1]["train"] < history[0]["train"]


def test_warmup_ramps_from_near_zero() -> None:
    trainer = make_trainer()
    peak = trainer.train_config.learning_rate
    assert trainer.learning_rate_at(0) == pytest.approx(peak / 10)
    assert trainer.learning_rate_at(9) == pytest.approx(peak)


def test_cosine_decays_to_floor() -> None:
    trainer = make_trainer()
    final = trainer.learning_rate_at(trainer.train_config.max_iters - 1)
    assert final == pytest.approx(trainer.train_config.learning_rate * 0.1, rel=0.05)


def test_learning_rate_never_exceeds_peak() -> None:
    trainer = make_trainer()
    peak = trainer.train_config.learning_rate
    assert all(
        trainer.learning_rate_at(s) <= peak + 1e-12
        for s in range(trainer.train_config.max_iters)
    )


def test_estimate_loss_reports_both_splits() -> None:
    losses = make_trainer().estimate_loss()
    assert set(losses) == {"train", "val"}
    assert all(v > 0 for v in losses.values())


def test_eval_restores_train_mode() -> None:
    trainer = make_trainer()
    trainer.model.train()
    trainer.estimate_loss()
    assert trainer.model.training


def test_same_seed_reproduces_run() -> None:
    a = make_trainer().train()
    b = make_trainer().train()
    assert a[-1]["train"] == pytest.approx(b[-1]["train"])


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    trainer = make_trainer()
    trainer.train()
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    fresh = make_trainer()
    fresh.load_checkpoint(path)

    for a, b in zip(
        trainer.model.parameters(), fresh.model.parameters(), strict=True
    ):
        assert torch.equal(a, b)


def test_checkpoint_rejects_mismatched_vocab(tmp_path: Path) -> None:
    trainer = make_trainer()
    path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(path)

    other_text = "xyz" * 5000
    tokenizer = CharTokenizer(other_text)
    dataset = Dataset(other_text, tokenizer)
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size, block_size=16)
    mismatched = Trainer(
        BigramLM(model_config), dataset, model_config, TrainConfig(device="cpu")
    )

    with pytest.raises(ValueError, match="vocabulary does not match"):
        mismatched.load_checkpoint(path)
