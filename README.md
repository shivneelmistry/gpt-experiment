# gpt-experiment

A decoder-only transformer built from scratch, then measured. No `nn.Transformer`,
no `nn.MultiheadAttention`.

## Architecture

6000 iterations, lr 3e-3, tiny Shakespeare.

| | Model | Params | Val loss |
|---|---|---|---|
| — | random | — | 4.17 |
| v1 | `BigramLM` | 4.2K | 2.45 |
| v2 | `SingleHeadAttentionLM` | 28.9K | 2.38 |
| v3 | `MultiHeadAttentionLM` | 33.0K | 2.16 |
| v4 | `PlainBlockLM` | 66.1K | 2.07 |
| v5 | `TransformerLM` | 215.9K | **1.66** |
| v6 | `CachedTransformerLM` | 215.9K | 1.66 |
| v7 | `BidirectionalLM` | 215.9K | encoder, MLM |

v3 beat v2 by 9% at the same parameter count.

## KV cache

p50 over 15 runs, 841K params, CPU.

| Tokens | Cached | Uncached | Speedup |
|---:|---:|---:|---:|
| 16 | 13.2 ms | 19.9 ms | 1.5× |
| 32 | 26.8 ms | 45.5 ms | 1.7× |
| 64 | 54.0 ms | 104.3 ms | 1.9× |
| 128 | 106.4 ms | 296.2 ms | **2.8×** |

Cached grows linearly, uncached quadratically.

## Prefill vs decode

| Phase | p50 | Throughput |
|---|---:|---:|
| Prefill, 128 tok | 3.27 ms | 39,128 tok/s |
| Decode, 1 tok | 0.82 ms | 1,219 tok/s |

32× apart. Prefill saturates compute; decode is memory bound.

## Batching

| Batch | Throughput | Cost per sequence |
|---:|---:|---:|
| 1 | 1,206 tok/s | 1.00× |
| 4 | 2,371 tok/s | 0.51× |
| 16 | 6,899 tok/s | **0.17×** |

## GQA

| kv heads | Cache | Params |
|---:|---:|---:|
| 8 | 32.0 MB | 3.32M |
| 2 | 8.0 MB | 2.93M |
| 1 | 4.0 MB | 2.86M |

Outputs stay exact. Latency unchanged on CPU — the win is memory.

## Encoder

v7 is v5 with the causal mask off, trained on masked reconstruction. Identical
parameter count, asserted in tests.

| | |
|---|---:|
| Masked accuracy | **49.1%** |
| Unigram baseline | 14.9% |

## Residuals

Not vanishing gradients — pre-LayerNorm already prevents that. Gradient norm at the
embedding layer:

| Depth | With | Without |
|---:|---:|---:|
| 2 | 0.51 | 0.69 |
| 8 | 0.54 | 0.42 |
| 32 | 0.57 | 2.24 |
| 64 | 0.45 | 0.11 |

Residuals buy scale stability, so one learning rate works at any depth.

## Measurement

- Warmup iterations discarded
- Device synchronized around every run — MPS and CUDA dispatch async, so naive timing
  measures queue submission, not execution
- p50/p95/p99, never the mean — latency is right-skewed
- Fixed seeds, config saved with each checkpoint

## Layout

```
gpt_experiment/
    tokenizer.py    characters <-> ids
    config.py       frozen hyperparameters
    data.py         batching, targets shifted one position
    model.py        v1 through v7
    masking.py      MLM corruption and accuracy
    cache.py        preallocated KV storage
    trainer.py      loop, schedule, checkpoints
    sampler.py      temperature, top-k, top-p
    benchmark.py    measurement harness
```

## Run

```bash
uv sync
curl -o corpus.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

uv run python scripts/train.py --model v5 --lr 3e-3 --iters 6000
uv run python scripts/train_mlm.py --iters 4000
uv run python scripts/generate.py --model v5 --temperature 0.8
uv run python scripts/bench.py --runs 20

uv run pytest      # 133 tests
```

`--model` takes `v1`…`v7`, plus `v5-noresid`.

## Next

Paged attention · continuous batching · INT8 and AWQ quantization · speculative decoding.
