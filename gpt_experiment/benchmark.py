"""Latency and throughput measurement."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import torch

from gpt_experiment.cache import KVCache
from gpt_experiment.config import ModelConfig
from gpt_experiment.model import CachedTransformerLM

__all__ = [
    "Measurement",
    "MemoryRow",
    "batch_scaling",
    "cache_memory",
    "generation_latency",
    "measure",
    "prefill_vs_decode",
    "synchronize",
]

_DEFAULT_WARMUP = 10
_DEFAULT_RUNS = 50


def synchronize(device: torch.device | str) -> None:
    """Wait for queued GPU work to finish.

    MPS and CUDA dispatch asynchronously, so timing without this measures queue
    submission rather than execution.
    """
    name = torch.device(device).type
    if name == "cuda":
        torch.cuda.synchronize()
    elif name == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class Measurement:
    """Timing distribution for one configuration.

    Percentiles, not a mean -- latency is right-skewed and the tail is the part
    that matters.
    """

    label: str
    samples: list[float] = field(repr=False)

    @property
    def p50(self) -> float:
        return statistics.quantiles(self.samples, n=100)[49]

    @property
    def p95(self) -> float:
        return statistics.quantiles(self.samples, n=100)[94]

    @property
    def p99(self) -> float:
        return statistics.quantiles(self.samples, n=100)[98]

    @property
    def mean(self) -> float:
        return statistics.fmean(self.samples)

    def row(self) -> str:
        return (
            f"{self.label:<28} {self.p50 * 1000:>9.2f} {self.p95 * 1000:>9.2f} "
            f"{self.p99 * 1000:>9.2f}"
        )

    @staticmethod
    def header() -> str:
        return f"{'':<28} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9}"


def measure(
    label: str,
    fn: Callable[[], object],
    device: torch.device | str = "cpu",
    warmup: int = _DEFAULT_WARMUP,
    runs: int = _DEFAULT_RUNS,
) -> Measurement:
    """Time fn, discarding warmup iterations and synchronizing around each run."""
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        synchronize(device)
        samples.append(time.perf_counter() - start)

    return Measurement(label, samples)


@torch.no_grad()
def generation_latency(
    model: CachedTransformerLM,
    lengths: tuple[int, ...] = (16, 32, 64, 128),
    device: str = "cpu",
    runs: int = 20,
) -> list[Measurement]:
    """Latency against sequence length. Cached grows linearly, uncached quadratically."""
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=device)

    results = []
    for length in lengths:
        for use_cache in (True, False):
            tag = "cached" if use_cache else "uncached"
            results.append(
                measure(
                    f"generate {length:>4} tok, {tag}",
                    partial(model.generate, start, length, use_cache=use_cache),
                    device=device,
                    warmup=3,
                    runs=runs,
                )
            )
    return results


@torch.no_grad()
def prefill_vs_decode(
    model: CachedTransformerLM,
    prompt_length: int = 128,
    device: str = "cpu",
    runs: int = 30,
) -> list[Measurement]:
    """Separate the two phases of inference.

    Prefill runs the whole prompt in parallel and is compute bound. Decode runs one
    token and is memory-bandwidth bound. Most serving decisions follow from that
    split.
    """
    model.eval()
    prompt = torch.randint(
        model.config.vocab_size, (1, prompt_length), device=device
    )

    def prefill() -> None:
        cache = KVCache(model.config, 1, device=device)
        model(prompt, cache=cache)

    warm_cache = KVCache(model.config, 1, device=device)
    model(prompt, cache=warm_cache)
    one_token = torch.zeros((1, 1), dtype=torch.long, device=device)

    def decode() -> None:
        warm_cache.rewind_to(prompt_length)  # replay the same step repeatedly
        model(one_token, cache=warm_cache)

    return [
        measure(f"prefill {prompt_length} tok", prefill, device, 5, runs),
        measure("decode 1 tok", decode, device, 5, runs),
    ]


@dataclass(frozen=True)
class MemoryRow:
    """Predicted against measured cache footprint for one batch size."""

    batch: int
    predicted: int
    measured: int

    @property
    def matches(self) -> bool:
        return self.predicted == self.measured


def cache_memory(
    config: ModelConfig,
    batch_sizes: tuple[int, ...] = (1, 4, 16),
    bytes_per_element: int = 4,
) -> list[MemoryRow]:
    """Allocated cache footprint against the closed-form prediction."""
    rows = []
    for batch in batch_sizes:
        predicted = (
            2  # keys and values
            * config.n_layer
            * batch
            * config.n_head
            * config.block_size
            * config.head_size
            * bytes_per_element
        )
        rows.append(
            MemoryRow(batch, predicted, KVCache(config, batch).bytes_allocated())
        )
    return rows


@torch.no_grad()
def batch_scaling(
    model: CachedTransformerLM,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16),
    tokens: int = 32,
    device: str = "cpu",
    runs: int = 10,
) -> list[tuple[int, Measurement, float]]:
    """Throughput against batch size. Returns (batch, timing, tokens per second).

    Decode is memory bound at batch 1, so extra sequences ride along nearly free.
    """
    model.eval()
    out = []
    for batch in batch_sizes:
        start = torch.zeros((batch, 1), dtype=torch.long, device=device)
        result = measure(
            f"batch {batch:>3}, {tokens} tok",
            partial(model.generate, start, tokens, use_cache=True),
            device=device,
            warmup=2,
            runs=runs,
        )
        out.append((batch, result, batch * tokens / result.p50))
    return out
