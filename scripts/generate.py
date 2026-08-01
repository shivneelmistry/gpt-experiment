"""Sample from a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gpt_experiment.config import ModelConfig
from gpt_experiment.model import MODELS
from gpt_experiment.sampler import Sampler, SamplingConfig
from gpt_experiment.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), default="v4")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = args.checkpoint or ROOT / "out" / f"{args.model}.pt"
    ckpt = torch.load(path, weights_only=False)

    tokenizer = CharTokenizer("".join(ckpt["vocab"]))
    config: ModelConfig = ckpt["model_config"]
    model = MODELS[args.model](config)
    model.load_state_dict(ckpt["model"])

    sampler = Sampler(model, tokenizer)
    text = sampler.generate(
        args.prompt,
        SamplingConfig(
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        ),
    )
    print(text)


if __name__ == "__main__":
    main()
