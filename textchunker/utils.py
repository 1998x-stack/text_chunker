"""Token and sentence utilities.

Sentence offsets are half-open Python character indexes into original input.
No whitespace or separators are silently removed from source spans.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Tuple

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except ImportError:  # pragma: no cover - optional dependency
    _enc = None


def count_tokens(text: str) -> int:
    """Legacy shared budget counter; explicit tokenizer injection is v0.3."""
    return len(_enc.encode(text)) if _enc is not None else len(text)


def whitespace_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Split on sentence punctuation and keep separator whitespace in spans.

    Chinese punctuation is recognized without requiring trailing spaces.
    The returned nonempty spans concatenate to the complete input.
    """
    if not text:
        return []
    pattern = r"(?<=[。！？!?；;])\s*|(?<=\.)\s+"
    results: List[Tuple[str, int, int]] = []
    cursor = 0
    for match in re.finditer(pattern, text):
        stop = match.end()
        if stop > cursor:
            results.append((text[cursor:stop], cursor, stop))
            cursor = stop
    if cursor < len(text):
        results.append((text[cursor:], cursor, len(text)))
    return results


def whitespace_sentences(text: str) -> List[str]:
    return [piece.strip() for piece, _, _ in
            whitespace_sentences_with_offsets(text) if piece.strip()]


def clamp(n: int, a: int, b: int) -> int:
    return max(a, min(n, b))


def sliding_windows(seq: List[str], k: int) -> Iterable[Tuple[int, List[str]]]:
    for index in range(0, len(seq) - k + 1):
        yield index, seq[index:index + k]
