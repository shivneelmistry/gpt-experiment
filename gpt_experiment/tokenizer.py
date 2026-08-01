"""Character-level tokenization."""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["CharTokenizer"]

_LATIN1 = "latin-1"
_BYTE_VOCAB_LIMIT = 256  # above this an ID no longer fits in one byte


class CharTokenizer:
    """
    Maps characters to integer IDs and back.
    """

    def __init__(self, text: str) -> None:
        if not text:
            raise ValueError("cannot build a vocabulary from empty text")

        # sorted: set order varies per run, which would remap every id
        self._chars: tuple[str, ...] = tuple(sorted(set(text)))
        self._stoi: dict[str, int] = {ch: i for i, ch in enumerate(self._chars)}
        self._use_byte_path: bool = len(self._chars) <= _BYTE_VOCAB_LIMIT

        self._to_byte_chars: dict[int, str] = {}
        self._from_byte_chars: dict[int, str] = {}
        if self._use_byte_path:
            self._to_byte_chars = str.maketrans(
                {ch: chr(i) for i, ch in enumerate(self._chars)}
            )
            self._from_byte_chars = str.maketrans(
                {chr(i): ch for i, ch in enumerate(self._chars)}
            )

    @property
    def vocab_size(self) -> int:
        return len(self._chars)

    @property
    def chars(self) -> tuple[str, ...]:
        return self._chars

    def encode(self, text: str) -> list[int]:
        """Text to token IDs. Raises KeyError on unknown characters."""
        if self._use_byte_path:
            try:
                ids = list(text.translate(self._to_byte_chars).encode(_LATIN1))
            except UnicodeEncodeError:
                raise self._unknown_character_error(text) from None
            # translate leaves unknown characters alone rather than failing, so one
            # can slip through as its own codepoint. Valid ids are < vocab_size.
            if ids and max(ids) >= self.vocab_size:
                raise self._unknown_character_error(text) from None
            return ids

        stoi = self._stoi
        try:
            return [stoi[ch] for ch in text]
        except KeyError:
            raise self._unknown_character_error(text) from None

    def decode(self, ids: Sequence[int]) -> str:
        """Token IDs back to text. Raises ValueError on out-of-range IDs."""
        if not ids:
            return ""

        self._validate_ids(ids)

        if self._use_byte_path:
            return bytes(ids).decode(_LATIN1).translate(self._from_byte_chars)

        chars = self._chars
        return "".join([chars[i] for i in ids])

    def _validate_ids(self, ids: Sequence[int]) -> None:
        # same silent-failure risk as encode, in reverse
        lowest, highest = min(ids), max(ids)
        if lowest < 0 or highest >= self.vocab_size:
            offender = lowest if lowest < 0 else highest
            raise ValueError(
                f"token ID {offender} is outside the vocabulary "
                f"(expected 0..{self.vocab_size - 1})"
            )

    def _unknown_character_error(self, text: str) -> KeyError:
        for ch in text:
            if ch not in self._stoi:
                return KeyError(
                    f"character {ch!r} is not in the vocabulary of "
                    f"{self.vocab_size} characters"
                )
        return KeyError("failed to encode text for an unknown reason")

    def __len__(self) -> int:
        return len(self._chars)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(vocab_size={self.vocab_size})"
