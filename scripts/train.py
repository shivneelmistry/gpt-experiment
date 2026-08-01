"""Train a model on the corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gpt_experiment.config import ModelConfig, TrainConfig
from gpt_experiment.data import Dataset
from gpt_experiment.model import MODELS
from gpt_experiment.tokenizer import CharTokenizer
from gpt_experiment.trainer import Trainer

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(MODELS), default="v1")
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus.txt")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample", type=int, default=300)
    args = parser.parse_args()

    text = args.corpus.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    dataset = Dataset(text, tokenizer)

    model_config = ModelConfig(vocab_size=tokenizer.vocab_size)
    train_config = TrainConfig(
        max_iters=args.iters, learning_rate=args.lr, device=args.device
    )

    torch.manual_seed(train_config.seed)  # before the model, so init is seeded
    model = MODELS[args.model](model_config)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} parameters")

    trainer = Trainer(model, dataset, model_config, train_config)
    trainer.train()

    out = args.out or ROOT / "out" / f"{args.model}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(out)
    print(f"\nsaved {out}")

    if args.sample:
        start = torch.zeros((1, 1), dtype=torch.long, device=train_config.device)
        ids = model.generate(start, args.sample)[0].tolist()
        print(f"\n--- sample ---\n{tokenizer.decode(ids)}")


if __name__ == "__main__":
    main()
