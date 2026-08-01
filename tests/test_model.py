"""Tests for models."""

from __future__ import annotations

import math

import torch

from gpt_experiment.config import ModelConfig
from gpt_experiment.model import BigramLM

VOCAB = 65


def make_model() -> BigramLM:
    torch.manual_seed(0)
    return BigramLM(ModelConfig(vocab_size=VOCAB))


def test_untrained_loss_matches_uniform_guess() -> None:
    """An untrained model should be equally unsure across the vocabulary.

    Loss must land near ln(vocab_size). If it does not, something is wrong before
    training has even started.
    """
    model = make_model()
    idx = torch.randint(VOCAB, (8, 16))
    targets = torch.randint(VOCAB, (8, 16))

    _, loss = model(idx, targets)

    assert loss is not None
    assert abs(loss.item() - math.log(VOCAB)) < 0.05


def test_logit_shape() -> None:
    model = make_model()
    logits, loss = model(torch.randint(VOCAB, (4, 8)))
    assert logits.shape == (4, 8, VOCAB)
    assert loss is None


def test_generate_extends_sequence() -> None:
    model = make_model()
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=20)
    assert out.shape == (1, 21)
    assert int(out.max()) < VOCAB


def test_generation_is_seeded() -> None:
    model = make_model()
    start = torch.zeros((1, 1), dtype=torch.long)

    torch.manual_seed(42)
    a = model.generate(start, max_new_tokens=20)
    torch.manual_seed(42)
    b = model.generate(start, max_new_tokens=20)

    assert torch.equal(a, b)


def test_gradients_reach_the_weights() -> None:
    model = make_model()
    _, loss = model(torch.randint(VOCAB, (4, 8)), torch.randint(VOCAB, (4, 8)))

    assert loss is not None
    loss.backward()

    grad = model.token_embedding.weight.grad
    assert grad is not None
    assert not torch.allclose(grad, torch.zeros_like(grad))


def test_parameter_count() -> None:
    model = make_model()
    assert sum(p.numel() for p in model.parameters()) == VOCAB * VOCAB
