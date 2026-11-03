"""Pluggable, explicit token-counting strategies."""

from .base import Tokenizer, CharacterTokenizer, LegacyTokenizer
from .tiktoken_adapter import TiktokenTokenizer
from .hf_adapter import HFTokenizer


def create_tokenizer(name: str) -> Tokenizer:
    if name in ("char", "character"):
        return CharacterTokenizer()
    if name == "legacy":
        return LegacyTokenizer()
    if name.startswith("tiktoken:"):
        return TiktokenTokenizer(name.split(":", 1)[1])
    if name.startswith("hf:"):
        return HFTokenizer(name.split(":", 1)[1])
    raise ValueError(f"Unknown tokenizer: {name!r}")


__all__ = [
    "Tokenizer", "CharacterTokenizer", "LegacyTokenizer",
    "TiktokenTokenizer", "HFTokenizer", "create_tokenizer",
]
