from __future__ import annotations
from typing import List
from loguru import logger
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..utils import count_tokens


def _split_by_separators(text: str, seps: List[str], size: int) -> List[str]:
    """按优先级分隔符递归切分，直到块不超过目标大小。"""
    if count_tokens(text) <= size or not seps:
        return [text]
    sep = seps[0]
    parts = text.split(sep) if sep else list(text)  # 最后一个分隔符为空字符串时按字符切
    chunks: List[str] = []
    buf = ""
    for i, p in enumerate(parts):
        piece = (buf + (p + sep if i < len(parts) - 1 else p))
        if count_tokens(piece) <= size:
            buf = piece
        else:
            if buf:
                chunks.append(buf)
            # 如果单个 part 自身就很大，递归继续拆
            if count_tokens(p) > size:
                chunks.extend(_split_by_separators(p, seps[1:], size))
                buf = ""
            else:
                buf = p + (sep if i < len(parts) - 1 else "")
    if buf:
        chunks.append(buf)
    # 二次遍历，若仍有过大的块，继续递归下一层分隔符
    out: List[str] = []
    for ch in chunks:
        if count_tokens(ch) > size and len(seps) > 1:
            out.extend(_split_by_separators(ch, seps[1:], size))
        else:
            out.append(ch)
    return out


@register("recursive")
class RecursiveChunker(BaseChunker):
    """递归分块：按段落→句子→词→字符等优先级递归切分。"""

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        r = self.cfg.recursive
        size = int(c.get("chunk_size", 512))
        overlap = int(c.get("chunk_overlap", 80))
        seps = list(r.get("separators", ["\n\n", "\n", " ", ""]))
        max_chunks = c.get("max_chunks")

        logger.info(f"Recursive chunking: size={size}, overlap={overlap}, separators={seps}")

        parts = _split_by_separators(text, seps, size)
        chunks: List[Chunk] = []
        cursor = 0
        for i, part in enumerate(parts):
            start = text.find(part, cursor)
            end = start + len(part)
            chunks.append(Chunk(id=len(chunks), text=part, start=start, end=end, meta={"strategy": "recursive"}))
            cursor = end
            if max_chunks and len(chunks) >= max_chunks:
                break

        # 简单重叠：合并相邻片段的尾/头（字符级）
        if overlap > 0 and chunks:
            for i in range(1, len(chunks)):
                prev, cur = chunks[i - 1], chunks[i]
                head = text[max(cur.start - overlap, prev.start):cur.start]
                cur.text = head + cur.text
                cur.start = cur.start - len(head)

        return chunks
