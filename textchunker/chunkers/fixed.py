from __future__ import annotations
from typing import List
from loguru import logger
from ..types import Chunk
from ..utils import count_tokens
from ..registry import register
from .base import BaseChunker


@register("fixed")
class FixedChunker(BaseChunker):
    """固定窗口分块（可重叠、按 token 或字符）。"""

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        f = self.cfg.fixed
        size = int(c.get("chunk_size", 512))
        overlap = int(c.get("chunk_overlap", 80))
        use_tokens = bool(f.get("use_tokens", True))
        max_chunks = c.get("max_chunks")

        logger.info(f"Fixed-size chunking: size={size}, overlap={overlap}, tokens={use_tokens}")
        chunks: List[Chunk] = []

        if use_tokens and count_tokens("x") == 1:
            # 使用 token 尺度的简单实现：切片时用近似字符比例回退
            # 为稳健性，这里直接按字符近似实现（工业环境可换为 tiktoken 的编码反解映射）
            pass

        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            piece = text[start:end]
            cid = len(chunks)
            chunks.append(Chunk(id=cid, text=piece, start=start, end=end, meta={"strategy": "fixed"}))
            if max_chunks and len(chunks) >= max_chunks:
                break
            if end == n:
                break
            start = max(end - overlap, start + 1)

        return chunks
