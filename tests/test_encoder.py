"""Tests for the bidirectional encoder and MLM masking."""

from __future__ import annotations

import math

import pytest
import torch

from gpt_experiment.config import ModelConfig, TrainConfig
from gpt_experiment.data import Dataset
from gpt_experiment.masking import (
    IGNORE_INDEX,
    MaskingConfig,
    apply_mlm_masking,
    masked_accuracy,
    unigram_accuracy,
)
from gpt_experiment.model import BidirectionalLM, Head, TransformerLM
from gpt_experiment.tokenizer import CharTokenizer
from gpt_experiment.trainer import MaskedTrainer

VOCAB = 65
MASK_ID = 64


def make_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=VOCAB, n_embd=32, n_head=4, n_layer=2, block_size=16, dropout=0.0
    )


def attention_weights(head: Head, x: torch.Tensor) -> torch.Tensor:
    _, time, _ = x.shape
    k, q = head.key(x), head.query(x)
    scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
    if head.causal:
        scores = scores.masked_fill(head.tril[:time, :time] == 0, float("-inf"))
    return torch.softmax(scores, dim=-1)


# --- the one-line difference -----------------------------------------------


def test_encoder_attends_in_both_directions() -> None:
    """The whole point of 1.1: only the mask separates encoder from decoder."""
    cfg = make_config()
    torch.manual_seed(0)
    x = torch.randn(1, 16, cfg.n_embd)

    causal = attention_weights(Head(cfg, 8, causal=True), x)
    bidirectional = attention_weights(Head(cfg, 8, causal=False), x)

    upper = torch.triu(torch.ones(16, 16), diagonal=1).bool()
    assert torch.all(causal.masked_select(upper) == 0)
    assert torch.all(bidirectional.masked_select(upper) > 0)


def test_encoder_output_depends_on_later_tokens() -> None:
    """Changing token 10 must change position 0 -- the opposite of the decoder test."""
    torch.manual_seed(0)
    model = BidirectionalLM(make_config()).eval()

    idx = torch.randint(VOCAB, (1, 16))
    tampered = idx.clone()
    tampered[:, 10:] = torch.randint(VOCAB, (1, 6))

    original, _ = model(idx)
    changed, _ = model(tampered)
    assert not torch.allclose(original[:, :10], changed[:, :10], atol=1e-4)


def test_encoder_refuses_to_generate() -> None:
    model = BidirectionalLM(make_config())
    with pytest.raises(NotImplementedError, match="no next token"):
        model.generate(torch.zeros((1, 1), dtype=torch.long), 5)


def test_encoder_and_decoder_have_identical_parameter_counts() -> None:
    """Same architecture. Only the mask and the objective differ."""
    cfg = make_config()
    torch.manual_seed(0)
    encoder = sum(p.numel() for p in BidirectionalLM(cfg).parameters())
    torch.manual_seed(0)
    decoder = sum(p.numel() for p in TransformerLM(cfg).parameters())
    assert encoder == decoder


# --- masking ---------------------------------------------------------------


def test_labels_are_ignored_where_nothing_was_selected() -> None:
    torch.manual_seed(0)
    tokens = torch.randint(VOCAB, (8, 32))
    _, labels = apply_mlm_masking(tokens, VOCAB, MASK_ID)

    selected = labels != IGNORE_INDEX
    assert torch.equal(labels[selected], tokens[selected])
    assert selected.sum() > 0


def test_selection_rate_is_about_fifteen_percent() -> None:
    torch.manual_seed(0)
    tokens = torch.randint(VOCAB, (64, 128))
    _, labels = apply_mlm_masking(tokens, VOCAB, MASK_ID)

    rate = (labels != IGNORE_INDEX).float().mean().item()
    assert 0.13 < rate < 0.17


def test_split_is_roughly_eighty_ten_ten() -> None:
    """Of selected tokens: 80% masked, 10% randomised, 10% left alone."""
    torch.manual_seed(0)
    tokens = torch.randint(VOCAB - 1, (128, 128))  # never the mask id itself
    inputs, labels = apply_mlm_masking(tokens, VOCAB - 1, MASK_ID)

    selected = labels != IGNORE_INDEX
    masked = ((inputs == MASK_ID) & selected).float().sum()
    untouched = ((inputs == tokens) & selected).float().sum()
    total = selected.float().sum()

    assert 0.77 < (masked / total).item() < 0.83
    # 10% kept, plus the ~1/vocab of randomised tokens that land on themselves
    assert 0.08 < (untouched / total).item() < 0.13


def test_unselected_tokens_are_untouched() -> None:
    torch.manual_seed(0)
    tokens = torch.randint(VOCAB, (16, 64))
    inputs, labels = apply_mlm_masking(tokens, VOCAB, MASK_ID)

    unselected = labels == IGNORE_INDEX
    assert torch.equal(inputs[unselected], tokens[unselected])


def test_shapes_are_preserved() -> None:
    tokens = torch.randint(VOCAB, (4, 12))
    inputs, labels = apply_mlm_masking(tokens, VOCAB, MASK_ID)
    assert inputs.shape == labels.shape == tokens.shape


def test_keep_prob_is_the_remainder() -> None:
    assert MaskingConfig().keep_prob == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mask_prob": 0.0}, "mask_prob must be in"),
        ({"mask_prob": 1.5}, "mask_prob must be in"),
        ({"replace_prob": 0.8, "random_prob": 0.5}, "must not exceed 1"),
    ],
)
def test_invalid_masking_config_rejected(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MaskingConfig(**kwargs)


# --- the objective ---------------------------------------------------------


def test_loss_only_scores_selected_positions() -> None:
    """IGNORE_INDEX positions must not contribute, or the model is graded on copying."""
    torch.manual_seed(0)
    model = BidirectionalLM(make_config()).eval()
    idx = torch.randint(VOCAB, (4, 16))

    labels = torch.full_like(idx, IGNORE_INDEX)
    labels[:, 0] = idx[:, 0]  # score one position

    _, scored_one = model(idx, labels)
    _, scored_all = model(idx, idx)

    assert scored_one is not None and scored_all is not None
    assert scored_one.item() != pytest.approx(scored_all.item())


def test_untrained_loss_matches_uniform_guess() -> None:
    torch.manual_seed(0)
    model = BidirectionalLM(make_config())
    tokens = torch.randint(VOCAB, (8, 16))
    inputs, labels = apply_mlm_masking(tokens, VOCAB, MASK_ID)

    _, loss = model(inputs, labels)
    assert loss is not None
    assert abs(loss.item() - math.log(VOCAB)) < 0.1


# --- accuracy metrics ------------------------------------------------------


def test_masked_accuracy_scores_only_selected_positions() -> None:
    """Perfect on the masked tokens, wrong everywhere else: still 100%."""
    logits = torch.zeros(1, 4, VOCAB)
    logits[0, 1, 7] = 10.0  # predicts 7 at position 1
    logits[0, 3, 9] = 10.0  # predicts 9 at position 3

    labels = torch.tensor([[IGNORE_INDEX, 7, IGNORE_INDEX, 9]])
    assert masked_accuracy(logits, labels) == pytest.approx(1.0)


def test_masked_accuracy_is_zero_when_all_wrong() -> None:
    logits = torch.zeros(1, 2, VOCAB)
    logits[0, 0, 5] = 10.0
    labels = torch.tensor([[6, IGNORE_INDEX]])
    assert masked_accuracy(logits, labels) == pytest.approx(0.0)


def test_masked_accuracy_handles_an_empty_selection() -> None:
    logits = torch.zeros(1, 2, VOCAB)
    labels = torch.full((1, 2), IGNORE_INDEX)
    assert masked_accuracy(logits, labels) == 0.0


def test_unigram_accuracy_is_the_commonest_token_frequency() -> None:
    tokens = torch.tensor([0, 0, 0, 1, 2])  # 3 of 5 are token 0
    assert unigram_accuracy(tokens, vocab_size=3) == pytest.approx(0.6)


def test_masked_trainer_corrupts_its_batches() -> None:
    """Inputs must differ from targets, or the task is a copy."""
    text = "".join(chr(ord("a") + i % 26) for i in range(20_000))
    tokenizer = CharTokenizer(text)
    dataset = Dataset(text, tokenizer)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size + 1, n_embd=32, n_head=4,
        n_layer=2, block_size=16, dropout=0.0,
    )
    train_cfg = TrainConfig(batch_size=8, max_iters=50, warmup_iters=5, device="cpu")

    torch.manual_seed(0)
    trainer = MaskedTrainer(BidirectionalLM(cfg), dataset, cfg, train_cfg)
    inputs, labels = trainer._batch("train")

    assert trainer.mask_token_id == tokenizer.vocab_size
    assert (labels != IGNORE_INDEX).any()
    assert (inputs == trainer.mask_token_id).any()
    assert int(inputs.max()) <= trainer.mask_token_id
