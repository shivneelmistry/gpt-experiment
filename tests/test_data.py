"""Tests for batching."""

from __future__ import annotations

import pytest
import torch

from gpt_experiment.data import Dataset
from gpt_experiment.tokenizer import CharTokenizer

TEXT = "".join(chr(ord("a") + i % 26) for i in range(10_000))


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    return Dataset(TEXT, CharTokenizer(TEXT))


def test_shapes(dataset: Dataset) -> None:
    x, y = dataset.get_batch("train", batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)


def test_y_is_x_shifted_by_one(dataset: Dataset) -> None:
    x, y = dataset.get_batch("train", batch_size=4, block_size=8)
    assert torch.equal(y[:, :-1], x[:, 1:])


def test_split_sizes(dataset: Dataset) -> None:
    assert dataset.size("train") == 9000
    assert dataset.size("val") == 1000
    assert len(dataset) == 10_000


def test_same_seed_same_batch(dataset: Dataset) -> None:
    torch.manual_seed(0)
    x1, y1 = dataset.get_batch("train", 4, 8)
    torch.manual_seed(0)
    x2, y2 = dataset.get_batch("train", 4, 8)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_ids_are_in_vocab(dataset: Dataset) -> None:
    x, y = dataset.get_batch("train", 16, 32)
    vocab_size = dataset.tokenizer.vocab_size
    assert int(x.min()) >= 0
    assert int(x.max()) < vocab_size
    assert int(y.max()) < vocab_size


def test_batch_does_not_alias_corpus(dataset: Dataset) -> None:
    """Mutating a batch must not corrupt the underlying data."""
    x, _ = dataset.get_batch("train", 2, 8)
    before = dataset.get_batch
    x[0, 0] = 999
    torch.manual_seed(0)
    fresh, _ = before("train", 2, 8)
    assert int(fresh.max()) < dataset.tokenizer.vocab_size


def test_block_too_large_rejected(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="needs at least"):
        dataset.get_batch("val", batch_size=1, block_size=5000)


def test_bad_train_frac_rejected() -> None:
    with pytest.raises(ValueError, match="must be in"):
        Dataset(TEXT, CharTokenizer(TEXT), train_frac=1.5)
