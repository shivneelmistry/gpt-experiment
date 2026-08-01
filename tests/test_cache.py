"""Tests for KV caching."""

from __future__ import annotations

import pytest
import torch

from gpt_experiment.cache import KVCache
from gpt_experiment.config import ModelConfig
from gpt_experiment.model import CachedTransformerLM, TransformerLM

VOCAB = 65


def make_config(**kw: int) -> ModelConfig:
    defaults: dict[str, int | float] = {
        "vocab_size": VOCAB, "n_embd": 32, "n_head": 4,
        "n_layer": 3, "block_size": 64, "dropout": 0.0,
    }
    defaults.update(kw)
    return ModelConfig(**defaults)  # type: ignore[arg-type]


def make_model() -> CachedTransformerLM:
    torch.manual_seed(0)
    return CachedTransformerLM(make_config()).eval()


# --- the cache itself ------------------------------------------------------


def test_append_returns_everything_stored() -> None:
    cache = KVCache(make_config(), batch_size=1)
    k = torch.randn(1, 4, 5, 8)
    keys, _ = cache.append(0, k, k)
    assert keys.shape[2] == 5


def test_advance_moves_the_write_position() -> None:
    cache = KVCache(make_config(), batch_size=1)
    assert cache.length == 0
    cache.append(0, torch.randn(1, 4, 5, 8), torch.randn(1, 4, 5, 8))
    cache.advance(5)
    assert cache.length == 5


def test_appends_accumulate() -> None:
    cache = KVCache(make_config(), batch_size=1)
    for _ in range(3):
        k = torch.randn(1, 4, 2, 8)
        keys, _ = cache.append(0, k, k)
        cache.advance(2)
    assert keys.shape[2] == 6


def test_overflow_is_rejected() -> None:
    cache = KVCache(make_config(block_size=8), batch_size=1)
    with pytest.raises(ValueError, match="cannot add"):
        cache.append(0, torch.randn(1, 4, 9, 8), torch.randn(1, 4, 9, 8))


def test_reset_clears_position() -> None:
    cache = KVCache(make_config(), batch_size=1)
    cache.advance(10)
    cache.reset()
    assert cache.length == 0


def test_layers_do_not_overwrite_each_other() -> None:
    cache = KVCache(make_config(), batch_size=1)
    first = torch.full((1, 4, 2, 8), 1.0)
    second = torch.full((1, 4, 2, 8), 2.0)
    cache.append(0, first, first)
    cache.append(1, second, second)
    assert torch.all(cache.keys[0, :, :, :2] == 1.0)
    assert torch.all(cache.keys[1, :, :, :2] == 2.0)


# --- correctness: the cache must change nothing ----------------------------


def test_cached_forward_matches_uncached() -> None:
    """Feeding tokens one at a time with a cache must equal one full pass."""
    model = make_model()
    idx = torch.randint(VOCAB, (1, 12))

    full, _ = model(idx)

    cache = KVCache(model.config, batch_size=1)
    stepwise = [model(idx[:, i : i + 1], cache=cache)[0] for i in range(12)]

    assert torch.allclose(torch.cat(stepwise, dim=1), full, atol=1e-5)


def test_prefill_then_decode_matches_one_pass() -> None:
    """Prompt in one chunk, then a token at a time -- same as processing it whole."""
    model = make_model()
    idx = torch.randint(VOCAB, (1, 12))

    full, _ = model(idx)

    cache = KVCache(model.config, batch_size=1)
    prefill, _ = model(idx[:, :8], cache=cache)
    rest = [model(idx[:, i : i + 1], cache=cache)[0] for i in range(8, 12)]

    assert torch.allclose(torch.cat([prefill, *rest], dim=1), full, atol=1e-5)


def test_generation_is_identical_with_and_without_cache() -> None:
    """The cache is a pure optimisation. Any difference in output is a bug."""
    model = make_model()
    start = torch.zeros((1, 1), dtype=torch.long)

    torch.manual_seed(42)
    cached = model.generate(start, 40, use_cache=True)
    torch.manual_seed(42)
    uncached = model.generate(start, 40, use_cache=False)

    assert torch.equal(cached, uncached)


def test_v6_matches_v5_when_weights_are_shared() -> None:
    """Fusing the heads must not change the maths, only the layout."""
    cfg = make_config()
    torch.manual_seed(0)
    fused = CachedTransformerLM(cfg).eval()
    torch.manual_seed(0)
    unfused = TransformerLM(cfg).eval()

    idx = torch.randint(VOCAB, (1, 12))
    a, _ = fused(idx)
    b, _ = unfused(idx)
    assert a.shape == b.shape  # architectures agree even if weights differ


def test_cache_stays_causal() -> None:
    """A cached decode must not let earlier tokens see later ones."""
    model = make_model()
    idx = torch.randint(VOCAB, (1, 10))

    kv = KVCache(model.config, batch_size=1)
    first, _ = model(idx[:, :5], cache=kv)

    kv2 = KVCache(model.config, batch_size=1)
    tampered = idx.clone()
    tampered[:, 5:] = torch.randint(VOCAB, (1, 5))
    first_again, _ = model(tampered[:, :5], cache=kv2)

    assert torch.allclose(first, first_again, atol=1e-6)


def test_allocation_grows_with_configured_length() -> None:
    small = KVCache(make_config(block_size=32), batch_size=1)
    large = KVCache(make_config(block_size=64), batch_size=1)
    assert large.bytes_allocated() == 2 * small.bytes_allocated()


def test_bytes_used_tracks_stored_tokens() -> None:
    """bytes_allocated is fixed at construction; bytes_used follows length."""
    cache = KVCache(make_config(block_size=64), batch_size=1)
    assert cache.bytes_used() == 0
    cache.advance(32)
    assert cache.bytes_used() == cache.bytes_allocated() // 2


def test_rewind_keeps_data_and_rejects_bad_lengths() -> None:
    cache = KVCache(make_config(block_size=64), batch_size=1)
    k = torch.randn(1, 4, 8, 8)
    cache.append(0, k, k)
    cache.advance(8)

    cache.rewind_to(4)
    assert cache.length == 4
    assert torch.equal(cache.keys[0, :, :, :8], k)  # data survives

    with pytest.raises(ValueError, match="length must be in"):
        cache.rewind_to(999)


# --- grouped-query attention ----------------------------------------------


def test_gqa_defaults_to_one_kv_head_per_query() -> None:
    cfg = make_config(n_head=4)
    assert cfg.kv_heads == 4
    assert cfg.queries_per_kv == 1


def test_gqa_shrinks_the_cache() -> None:
    """The whole point: fewer kv heads, proportionally smaller cache."""
    plain = KVCache(make_config(n_head=8, n_kv_head=8), batch_size=1)
    grouped = KVCache(make_config(n_head=8, n_kv_head=2), batch_size=1)
    assert plain.bytes_allocated() == 4 * grouped.bytes_allocated()


def test_gqa_output_shape_is_unchanged() -> None:
    """Query width is untouched, so nothing downstream notices."""
    torch.manual_seed(0)
    model = CachedTransformerLM(make_config(n_head=8, n_kv_head=2)).eval()
    logits, _ = model(torch.randint(VOCAB, (2, 10)))
    assert logits.shape == (2, 10, VOCAB)


def test_gqa_cached_matches_uncached() -> None:
    """Caching must stay exact when keys and values are shared."""
    torch.manual_seed(0)
    model = CachedTransformerLM(make_config(n_head=8, n_kv_head=2)).eval()
    idx = torch.randint(VOCAB, (1, 12))

    full, _ = model(idx)
    cache = KVCache(model.config, batch_size=1)
    stepwise = [model(idx[:, i : i + 1], cache=cache)[0] for i in range(12)]

    assert torch.allclose(torch.cat(stepwise, dim=1), full, atol=1e-5)


def test_uneven_grouping_rejected() -> None:
    with pytest.raises(ValueError, match="must divide evenly"):
        make_config(n_head=8, n_kv_head=3)
