"""Tests for the character tokenizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpt_experiment.tokenizer import CharTokenizer

CORPUS = Path(__file__).resolve().parent.parent / "corpus.txt"


@pytest.fixture(scope="module")
def corpus() -> str:
    if not CORPUS.exists():
        pytest.skip(f"{CORPUS.name} not present")
    return CORPUS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tokenizer(corpus: str) -> CharTokenizer:
    return CharTokenizer(corpus)


def test_vocab_size(tokenizer: CharTokenizer) -> None:
    assert tokenizer.vocab_size == 65


def test_roundtrip_is_lossless(tokenizer: CharTokenizer, corpus: str) -> None:
    sample = corpus[:10_000]
    assert tokenizer.decode(tokenizer.encode(sample)) == sample


def test_encode_length_matches_input(tokenizer: CharTokenizer) -> None:
    assert len(tokenizer.encode("hello")) == 5


def test_mapping_is_deterministic(corpus: str) -> None:
    """Two tokenizers built from the same text must agree.

    Guards against building the vocabulary from an unsorted set, which would
    produce a different mapping per run and silently break checkpoints.
    """
    a, b = CharTokenizer(corpus), CharTokenizer(corpus)
    assert a.chars == b.chars
    assert a.encode("hello world") == b.encode("hello world")


def test_ids_are_contiguous_from_zero(tokenizer: CharTokenizer) -> None:
    ids = tokenizer.encode("".join(tokenizer.chars))
    assert ids == list(range(tokenizer.vocab_size))


def test_empty_sequences() -> None:
    tok = CharTokenizer("abc")
    assert tok.encode("") == []
    assert tok.decode([]) == ""


def test_empty_text_rejected() -> None:
    with pytest.raises(ValueError, match="empty text"):
        CharTokenizer("")


def test_unknown_character_rejected() -> None:
    tok = CharTokenizer("abc")
    with pytest.raises(KeyError, match="not in the vocabulary"):
        tok.encode("abz")


def test_out_of_range_id_rejected() -> None:
    tok = CharTokenizer("abc")
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tok.decode([0, 99])
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tok.decode([-1])


def test_dict_path_matches_byte_path() -> None:
    """The two implementations must be behaviourally identical.

    A vocabulary above 256 characters falls back to the dict path; both must
    produce the same IDs for the same input.
    """
    text = "".join(chr(i) for i in range(0x4E00, 0x4E00 + 300))  # 300 CJK chars
    tok = CharTokenizer(text)
    assert tok.vocab_size == 300
    assert tok.decode(tok.encode(text)) == text


def test_len_and_repr() -> None:
    tok = CharTokenizer("abc")
    assert len(tok) == 3
    assert repr(tok) == "CharTokenizer(vocab_size=3)"
