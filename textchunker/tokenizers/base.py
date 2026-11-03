"""Tokenizer interface used for counting complete source substrings."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    @property
    def name(self) -> str:
        ...

    def count(self, text: str) -> int:
        ...


class CharacterTokenizer:
    name = "character"

    def count(self, text: str) -> int:
        return len(text)


class LegacyTokenizer:
    name = "legacy:cl100k_base-or-char"

    def count(self, text: str) -> int:
        from ..utils import count_tokens
        return count_tokens(text)
