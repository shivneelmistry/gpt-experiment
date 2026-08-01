"""Run the benchmark suite."""

from __future__ import annotations

import argparse

import torch

from gpt_experiment.benchmark import (
    Measurement,
    batch_scaling,
    cache_memory,
    generation_latency,
    prefill_vs_decode,
)
from gpt_experiment.config import ModelConfig
from gpt_experiment.model import CachedTransformerLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--runs", type=int, default=20)
    args = parser.parse_args()

    config = ModelConfig(
        vocab_size=65,
        n_embd=args.n_embd,
        n_head=4,
        n_layer=args.n_layer,
        block_size=args.block_size,
        dropout=0.0,
    )
    torch.manual_seed(0)
    model = CachedTransformerLM(config).to(args.device).eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"device {args.device} | {params:,} parameters | {config.n_layer} layers")
    print("percentiles over independent runs, warmup discarded, device synchronized\n")

    print("--- generation latency ---")
    print(Measurement.header())
    for row in generation_latency(model, device=args.device, runs=args.runs):
        print(row.row())

    print("\n--- prefill vs decode ---")
    print(Measurement.header())
    phases = prefill_vs_decode(model, device=args.device, runs=args.runs)
    for row in phases:
        print(row.row())
    prefill, decode = phases
    print(
        f"\nprefill throughput  {128 / prefill.p50:>10,.0f} tok/s"
        f"\ndecode  throughput  {1 / decode.p50:>10,.0f} tok/s"
        f"\nratio               {(128 / prefill.p50) / (1 / decode.p50):>10.1f}x"
    )

    print("\n--- kv cache memory ---")
    print(f"{'batch':>6} {'predicted':>12} {'measured':>12} {'match':>7}")
    for memory in cache_memory(config):
        print(
            f"{memory.batch:>6} {memory.predicted / 1024:>10.1f}kB "
            f"{memory.measured / 1024:>10.1f}kB {'yes' if memory.matches else 'NO':>7}"
        )

    print("\n--- batch scaling ---")
    print(f"{'batch':>6} {'p50 ms':>10} {'tok/s':>10} {'per-seq cost':>14}")
    baseline = None
    for batch, timing, throughput in batch_scaling(model, device=args.device, runs=8):
        baseline = baseline or timing.p50
        print(
            f"{batch:>6} {timing.p50 * 1000:>10.1f} {throughput:>10,.0f} "
            f"{timing.p50 / batch / baseline:>13.2f}x"
        )


if __name__ == "__main__":
    main()
