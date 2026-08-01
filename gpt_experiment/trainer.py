"""Training loop."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from gpt_experiment.config import ModelConfig, TrainConfig
from gpt_experiment.data import Dataset, Split
from gpt_experiment.masking import MaskingConfig, apply_mlm_masking, masked_accuracy

__all__ = ["MaskedTrainer", "Trainer"]

_MIN_LR_FRACTION = 0.1  # cosine decays to this share of peak, not to zero


class Trainer:
    """Owns the model, optimizer and schedule for one run.

    Seeds batching and dropout only -- weights are initialised before the Trainer
    exists, so seed before constructing the model for a fully reproducible run.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        model_config: ModelConfig,
        train_config: TrainConfig,
    ) -> None:
        torch.manual_seed(train_config.seed)

        self.model = model.to(train_config.device)
        self.dataset = dataset
        self.model_config = model_config
        self.train_config = train_config
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_config.learning_rate,
            weight_decay=train_config.weight_decay,
            betas=(0.9, 0.95),  # 0.95 not the 0.999 default; standard for LLMs
        )
        self.history: list[dict[str, float]] = []

    def learning_rate_at(self, step: int) -> float:
        """Linear warmup, then cosine decay to a floor."""
        cfg = self.train_config
        if step < cfg.warmup_iters:
            return cfg.learning_rate * (step + 1) / cfg.warmup_iters

        span = max(1, cfg.max_iters - cfg.warmup_iters)
        progress = min(1.0, (step - cfg.warmup_iters) / span)
        floor = cfg.learning_rate * _MIN_LR_FRACTION
        return floor + 0.5 * (cfg.learning_rate - floor) * (1 + math.cos(math.pi * progress))

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Mean loss per split, averaged over batches. One batch is noise."""
        self.model.eval()
        out: dict[str, float] = {}
        for split in ("train", "val"):
            losses = torch.zeros(self.train_config.eval_iters)
            for i in range(self.train_config.eval_iters):
                x, y = self._batch(split)
                _, loss = self.model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def train(self) -> list[dict[str, float]]:
        cfg = self.train_config
        self.model.train()

        for step in range(cfg.max_iters):
            lr = self.learning_rate_at(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            x, y = self._batch("train")
            _, loss = self.model(x, y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

            if step % cfg.eval_interval == 0 or step == cfg.max_iters - 1:
                losses = self.estimate_loss()
                record = {"step": float(step), "lr": lr, **losses}
                self.history.append(record)
                print(
                    f"step {step:>5}  train {losses['train']:.4f}  "
                    f"val {losses['val']:.4f}  lr {lr:.2e}"
                )

        return self.history

    def save_checkpoint(self, path: str | Path) -> None:
        """Weights, config and vocabulary. Without the vocabulary it decodes to garbage."""
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": self.model_config,
                "train_config": self.train_config,
                "vocab": self.dataset.tokenizer.chars,
                "history": self.history,
            },
            Path(path),
        )

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        ckpt: dict[str, Any] = torch.load(Path(path), weights_only=False)
        if ckpt["vocab"] != self.dataset.tokenizer.chars:
            raise ValueError("checkpoint vocabulary does not match this tokenizer")
        self.model.load_state_dict(ckpt["model"])
        self.history = ckpt["history"]
        return ckpt

    def _batch(self, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dataset.get_batch(
            split,
            self.train_config.batch_size,
            self.model_config.block_size,
            self.train_config.device,
        )


class MaskedTrainer(Trainer):
    """Trains an encoder on masked reconstruction. Only _batch differs from Trainer.

    The mask token sits one slot past the tokenizer's vocabulary, so random
    replacements can never produce it.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        model_config: ModelConfig,
        train_config: TrainConfig,
        masking: MaskingConfig | None = None,
    ) -> None:
        super().__init__(model, dataset, model_config, train_config)
        self.masking = masking or MaskingConfig()
        self.text_vocab = dataset.tokenizer.vocab_size
        self.mask_token_id = self.text_vocab

    def _batch(self, split: Split) -> tuple[torch.Tensor, torch.Tensor]:
        windows, _ = super()._batch(split)  # shifted targets unused here
        return apply_mlm_masking(
            windows, self.text_vocab, self.mask_token_id, self.masking
        )

    @torch.no_grad()
    def masked_accuracy(self, split: Split = "val", batches: int = 20) -> float:
        """Mean reconstruction accuracy over several batches."""
        self.model.eval()
        scores = []
        for _ in range(batches):
            inputs, labels = self._batch(split)
            logits, _ = self.model(inputs, labels)
            scores.append(masked_accuracy(logits, labels))
        self.model.train()
        return sum(scores) / len(scores)
