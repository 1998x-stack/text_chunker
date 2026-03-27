from __future__ import annotations
import re
from typing import Iterable, List, Tuple, Optional

# 尝试使用 tiktoken 进行 token 计数，不可用则退化到字符级
try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _enc = None


def count_tokens(text: str) -> int:
    """统计文本的 token 数（优先 tiktoken；否则返回字符数）。"""
    if _enc:
        return len(_enc.encode(text))
    return len(text)


def whitespace_sentences(text: str) -> List[str]:
    """轻量级句子切分（中文/英文兼容的标点启发式）。"""
    pat = r'(?<=[。！？!?；;])\s+|(?<=\.)\s+'
    parts = re.split(pat, text.strip())
    return [p for p in parts if p]


def whitespace_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Sentence splitting that returns (sentence, start, end) tuples.

    Uses the same regex as whitespace_sentences but tracks character offsets,
    eliminating the need for text.find() which fails on duplicate sentences.
    """
    pat = r'(?<=[。！？!?；;])\s+|(?<=\.)\s+'
    stripped = text.strip()
    if not stripped:
        return []
    parts = re.split(pat, stripped)
    offset = text.index(stripped[0]) if stripped else 0
    results: List[Tuple[str, int, int]] = []
    cursor = offset
    for part in parts:
        if not part:
            continue
        start = text.find(part, cursor)
        if start == -1:
            start = cursor
        end = start + len(part)
        results.append((part, start, end))
        cursor = end
    return results


def clamp(n: int, a: int, b: int) -> int:
    return max(a, min(n, b))


def sliding_windows(seq: List[str], k: int) -> Iterable[Tuple[int, List[str]]]:
    """滑动窗口，返回 (起始索引, 片段列表)。"""
    for i in range(0, len(seq) - k + 1):
        yield i, seq[i:i + k]
