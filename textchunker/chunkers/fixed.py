from __future__ import annotations

from typing import List

from loguru import logger

from ..exceptions import ChunkerError
from ..types import Chunk
from .. import utils as _utils_mod
from ..utils import count_tokens
from ..registry import register
from .base import BaseChunker


@register("fixed")
class FixedChunker(BaseChunker):
    """Fixed-size sliding window chunker (character or token based)."""

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        f = self.cfg.fixed
        size = int(c.get("chunk_size", 512))
        overlap = int(c.get("chunk_overlap", 80))
        use_tokens = bool(f.get("use_tokens", True))
        max_chunks = c.get("max_chunks")

        if not text:
            return []

        enc = _utils_mod._enc
        if use_tokens and enc is not None:
            return self._chunk_by_tokens(text, size, overlap, max_chunks, enc)

        if use_tokens and enc is None:
            logger.warning("tiktoken unavailable, falling back to character-based chunking")

        return self._chunk_by_chars(text, size, overlap, max_chunks)

    def _chunk_by_chars(self, text: str, size: int, overlap: int, max_chunks: int | None) -> List[Chunk]:
        chunks: List[Chunk] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            piece = text[start:end]
            chunks.append(Chunk(
                id=len(chunks), text=piece, start=start, end=end,
                meta={"strategy": "fixed", "mode": "char"},
            ))
            if max_chunks and len(chunks) >= max_chunks:
                break
            if end == n:
                break
            start = end - overlap if overlap < (end - start) else start + 1
        logger.info("Fixed chunking (char): {} chunks from {} chars", len(chunks), n)
        return chunks

    def _chunk_by_tokens(self, text: str, size: int, overlap: int, max_chunks: int | None, enc) -> List[Chunk]:
        tokens = enc.encode(text)
        chunks: List[Chunk] = []
        tok_start = 0
        n = len(tokens)
        while tok_start < n:
            tok_end = min(tok_start + size, n)
            chunk_tokens = tokens[tok_start:tok_end]
            piece = enc.decode(chunk_tokens)
            char_start = len(enc.decode(tokens[:tok_start]))
            char_end = len(enc.decode(tokens[:tok_end]))
            chunks.append(Chunk(
                id=len(chunks), text=piece, start=char_start, end=char_end,
                meta={"strategy": "fixed", "mode": "token"},
            ))
            if max_chunks and len(chunks) >= max_chunks:
                break
            if tok_end == n:
                break
            tok_start = tok_end - overlap if overlap < (tok_end - tok_start) else tok_start + 1
        logger.info("Fixed chunking (token): {} chunks from {} tokens", len(chunks), n)
        return chunks
