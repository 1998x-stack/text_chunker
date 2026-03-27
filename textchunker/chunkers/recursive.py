from __future__ import annotations

from typing import List

from loguru import logger

from ..exceptions import ChunkerError
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..utils import count_tokens


def _split_by_separators(text: str, seps: List[str], size: int) -> List[str]:
    """Recursively split text by separator priority until chunks fit within size."""
    if count_tokens(text) <= size or not seps:
        return [text]
    sep = seps[0]
    parts = text.split(sep) if sep else list(text)
    chunks: List[str] = []
    buf = ""
    for i, p in enumerate(parts):
        piece = buf + (p + sep if i < len(parts) - 1 else p)
        if count_tokens(piece) <= size:
            buf = piece
        else:
            if buf:
                chunks.append(buf)
            if count_tokens(p) > size:
                chunks.extend(_split_by_separators(p, seps[1:], size))
                buf = ""
            else:
                buf = p + (sep if i < len(parts) - 1 else "")
    if buf:
        chunks.append(buf)
    out: List[str] = []
    for ch in chunks:
        if count_tokens(ch) > size and len(seps) > 1:
            out.extend(_split_by_separators(ch, seps[1:], size))
        else:
            out.append(ch)
    return out


def _build_offset_map(text: str, parts: List[str]) -> List[int]:
    """Build start offsets for each part using cumulative cursor.

    This is safe for duplicate parts because the cursor only moves forward.
    """
    offsets: List[int] = []
    cursor = 0
    for part in parts:
        idx = text.find(part, cursor)
        if idx == -1:
            idx = cursor
        offsets.append(idx)
        cursor = idx + len(part)
    return offsets


@register("recursive")
class RecursiveChunker(BaseChunker):
    """Recursive chunker: splits by paragraph > line > word > character priority."""

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        r = self.cfg.recursive
        size = int(c.get("chunk_size", 512))
        overlap = int(c.get("chunk_overlap", 80))
        seps = list(r.get("separators", ["\n\n", "\n", " ", ""]))
        max_chunks = c.get("max_chunks")

        if not text.strip():
            return []

        parts = _split_by_separators(text, seps, size)
        offsets = _build_offset_map(text, parts)

        chunks: List[Chunk] = []
        for i, part in enumerate(parts):
            start = offsets[i]
            end = start + len(part)
            chunks.append(Chunk(
                id=len(chunks), text=part, start=start, end=end,
                meta={"strategy": "recursive"},
            ))
            if max_chunks and len(chunks) >= max_chunks:
                break

        if overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev, cur = chunks[i - 1], chunks[i]
                head_start = max(cur.start - overlap, prev.start)
                head = text[head_start:cur.start]
                cur.text = head + cur.text
                cur.start = head_start

        logger.info("Recursive chunking: {} chunks, size={}, overlap={}", len(chunks), size, overlap)
        return chunks
