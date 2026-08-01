"""Tests for the measurement harness."""

from __future__ import annotations

import time

import torch

from gpt_experiment.benchmark import (
    Measurement,
    batch_scaling,
    cache_memory,
    generation_latency,
    measure,
    prefill_vs_decode,
    synchronize,
)
from gpt_experiment.config import ModelConfig
from gpt_experiment.model import CachedTransformerLM


def make_model() -> CachedTransformerLM:
    torch.manual_seed(0)
    config = ModelConfig(
        vocab_size=65, n_embd=32, n_head=4, n_layer=2, block_size=64, dropout=0.0
    )
    return CachedTransformerLM(config).eval()


def test_percentiles_are_ordered() -> None:
    m = Measurement("x", [i / 1000 for i in range(1, 101)])
    assert m.p50 < m.p95 < m.p99


def test_percentile_beats_mean_on_a_tail() -> None:
    """One slow outlier moves the mean but not the median. That is the point."""
    samples = [0.001] * 99 + [10.0]
    m = Measurement("x", samples)
    assert m.p50 < 0.002
    assert m.mean > 0.1


def test_measure_discards_warmup() -> None:
    """Warmup runs must not appear in the samples."""
    calls = 0

    def fn() -> None:
        nonlocal calls
        calls += 1

    result = measure("x", fn, warmup=5, runs=10)
    assert calls == 15
    assert len(result.samples) == 10


def test_measure_records_real_time() -> None:
    result = measure("sleep", lambda: time.sleep(0.002), warmup=1, runs=5)
    assert result.p50 >= 0.002


def test_synchronize_is_safe_on_cpu() -> None:
    synchronize("cpu")  # must not raise


def test_cache_memory_matches_prediction() -> None:
    """Measured footprint must equal the closed-form formula, or the table lies."""
    config = ModelConfig(
        vocab_size=65, n_embd=32, n_head=4, n_layer=2, block_size=64, dropout=0.0
    )
    assert all(row.matches for row in cache_memory(config))


def test_generation_latency_covers_both_paths() -> None:
    results = generation_latency(make_model(), lengths=(8, 16), runs=2)
    labels = [r.label for r in results]
    assert sum("cached" in label and "un" not in label for label in labels) == 2
    assert sum("uncached" in label for label in labels) == 2


def test_cached_beats_uncached_at_length() -> None:
    """The headline claim, asserted rather than eyeballed."""
    results = generation_latency(make_model(), lengths=(48,), runs=5)
    cached, uncached = results
    assert cached.p50 < uncached.p50


def test_prefill_and_decode_are_reported_separately() -> None:
    prefill, decode = prefill_vs_decode(make_model(), prompt_length=32, runs=3)
    assert "prefill" in prefill.label
    assert "decode" in decode.label
    assert prefill.p50 > decode.p50  # many tokens versus one


def test_batch_scaling_reports_throughput() -> None:
    rows = batch_scaling(make_model(), batch_sizes=(1, 4), tokens=8, runs=2)
    assert [batch for batch, _, _ in rows] == [1, 4]
    assert all(throughput > 0 for _, _, throughput in rows)


def test_larger_batches_raise_throughput() -> None:
    """Decode is memory bound, so extra sequences ride along nearly free."""
    rows = batch_scaling(make_model(), batch_sizes=(1, 8), tokens=8, runs=3)
    assert rows[1][2] > rows[0][2]
