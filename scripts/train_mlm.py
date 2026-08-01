"""Train the encoder on masked-token reconstruction."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gpt_experiment.config import ModelConfig, TrainConfig
from gpt_experiment.data import Dataset
from gpt_experiment.masking import unigram_accuracy
from gpt_experiment.model import BidirectionalLM
from gpt_experiment.tokenizer import CharTokenizer
from gpt_experiment.trainer import MaskedTrainer

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=ROOT / "corpus.txt")
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "v7.pt")
    parser.add_argument("--iters", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    text = args.corpus.read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    dataset = Dataset(text, tokenizer)

    # One slot past the tokenizer for [MASK], so random replacements can never
    # produce it and the model can still emit every real character.
    model_config = ModelConfig(vocab_size=tokenizer.vocab_size + 1)
    train_config = TrainConfig(
        max_iters=args.iters, learning_rate=args.lr, device=args.device
    )

    torch.manual_seed(train_config.seed)
    trainer = MaskedTrainer(
        BidirectionalLM(model_config), dataset, model_config, train_config
    )

    encoded = torch.tensor(tokenizer.encode(text[:200_000]))
    baseline = unigram_accuracy(encoded, tokenizer.vocab_size)
    print(f"vocabulary {tokenizer.vocab_size} + 1 for [MASK]")
    print(f"unigram baseline: {baseline:.1%} (always guess the commonest character)\n")

    trainer.train()

    accuracy = trainer.masked_accuracy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.out)

    print(f"\nmasked accuracy  {accuracy:.1%}")
    print(f"unigram baseline {baseline:.1%}")
    print(f"lift             {accuracy / baseline:.1f}x")


if __name__ == "__main__":
    main()
