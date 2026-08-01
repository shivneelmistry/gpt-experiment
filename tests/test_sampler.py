"""Tests for sampling."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from gpt_experiment.config import ModelConfig
from gpt_experiment.model import TransformerLM
from gpt_experiment.sampler import (
    Sampler,
    SamplingConfig,
    apply_temperature,
    apply_top_k,
    apply_top_p,
)
from gpt_experiment.tokenizer import CharTokenizer

TEXT = "".join(chr(ord("a") + i % 26) for i in range(2000))


@pytest.fixture(scope="module")
def sampler() -> Sampler:
    torch.manual_seed(0)
    tokenizer = CharTokenizer(TEXT)
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size, n_embd=32, n_head=4, n_layer=2,
        block_size=16, dropout=0.0,
    )
    return Sampler(TransformerLM(config), tokenizer)


# --- filters ---------------------------------------------------------------


def test_temperature_below_one_sharpens() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    cold = F.softmax(apply_temperature(logits, 0.5), dim=-1)
    warm = F.softmax(apply_temperature(logits, 2.0), dim=-1)
    assert cold.max() > warm.max()


def test_temperature_of_one_is_identity() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(apply_temperature(logits, 1.0), logits)


def test_top_k_keeps_exactly_k() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    assert torch.isfinite(apply_top_k(logits, 2)).sum().item() == 2


def test_top_k_keeps_the_highest() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
    kept = torch.isfinite(apply_top_k(logits, 2))[0]
    assert bool(kept[1]) and bool(kept[4])  # values 5.0 and 4.0


def test_top_k_larger_than_vocab_is_harmless() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.equal(apply_top_k(logits, 99), logits)


def test_top_p_adapts_to_confidence() -> None:
    """A confident distribution keeps fewer tokens than an uncertain one."""
    confident = torch.tensor([[10.0, 1.0, 1.0, 1.0, 1.0]])
    uncertain = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0]])
    kept_confident = torch.isfinite(apply_top_p(confident, 0.9)).sum().item()
    kept_uncertain = torch.isfinite(apply_top_p(uncertain, 0.9)).sum().item()
    assert kept_confident < kept_uncertain


def test_top_p_always_keeps_at_least_one() -> None:
    """A token above p on its own must not be filtered out entirely."""
    logits = torch.tensor([[100.0, 1.0, 1.0]])
    assert torch.isfinite(apply_top_p(logits, 0.5)).sum().item() >= 1


# --- config validation -----------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"temperature": 0.0}, "temperature must be positive"),
        ({"temperature": -1.0}, "temperature must be positive"),
        ({"top_k": 0}, "top_k must be at least 1"),
        ({"top_p": 0.0}, "top_p must be in"),
        ({"top_p": 1.5}, "top_p must be in"),
    ],
)
def test_invalid_config_rejected(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SamplingConfig(**kwargs)  # type: ignore[arg-type]


# --- generation ------------------------------------------------------------


def test_output_length(sampler: Sampler) -> None:
    out = sampler.generate("ab", SamplingConfig(max_new_tokens=25, seed=0))
    assert len(out) == 2 + 25


def test_seed_makes_output_reproducible(sampler: Sampler) -> None:
    cfg = SamplingConfig(max_new_tokens=30, seed=7)
    assert sampler.generate("a", cfg) == sampler.generate("a", cfg)


def test_different_seeds_diverge(sampler: Sampler) -> None:
    a = sampler.generate("a", SamplingConfig(max_new_tokens=40, seed=1))
    b = sampler.generate("a", SamplingConfig(max_new_tokens=40, seed=2))
    assert a != b


def test_top_k_of_one_is_deterministic(sampler: Sampler) -> None:
    """With only one candidate the dice roll cannot change the outcome."""
    cfg_a = SamplingConfig(max_new_tokens=30, top_k=1, seed=1)
    cfg_b = SamplingConfig(max_new_tokens=30, top_k=1, seed=999)
    assert sampler.generate("a", cfg_a) == sampler.generate("a", cfg_b)


def test_generates_past_block_size(sampler: Sampler) -> None:
    """Cropping must keep working beyond the position-embedding table."""
    out = sampler.generate("a", SamplingConfig(max_new_tokens=60, seed=0))
    assert len(out) == 61


def test_output_stays_in_vocabulary(sampler: Sampler) -> None:
    out = sampler.generate("a", SamplingConfig(max_new_tokens=50, seed=0))
    assert set(out) <= set(sampler.tokenizer.chars)


def test_empty_prompt_is_allowed(sampler: Sampler) -> None:
    out = sampler.generate("", SamplingConfig(max_new_tokens=10, seed=0))
    assert len(out) == 11  # seeded with one padding token


def test_low_temperature_repeats_more(sampler: Sampler) -> None:
    """Sharper distributions concentrate on fewer characters."""
    cold = sampler.generate("a", SamplingConfig(max_new_tokens=200, temperature=0.1, seed=0))
    hot = sampler.generate("a", SamplingConfig(max_new_tokens=200, temperature=2.0, seed=0))
    assert len(set(cold)) < len(set(hot))
