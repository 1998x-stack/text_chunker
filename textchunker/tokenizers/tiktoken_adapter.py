"""Exact token count with a named tiktoken encoding."""
from __future__ import annotations


class TiktokenTokenizer:
    def __init__(self, encoding: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "Install the optional token dependency: pip install -e '.[token]'"
            ) from exc
        self._encoding = tiktoken.get_encoding(encoding)
        self.name = f"tiktoken:{encoding}"

    def count(self, text: str) -> int:
        return len(self._encoding.encode(text))
