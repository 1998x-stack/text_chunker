"""Lossless fixed-size character/token chunking."""

from __future__ import annotations

from typing import List

from loguru import logger

from .. import utils as _utils_mod
from ..core.token_budget import largest_fitting_end
from ..core.validation import validate_parameters
from ..registry import register
from ..types import Chunk
from .base import BaseChunker


@register("fixed")
class FixedChunker(BaseChunker):
    """Windowed text splitting with code-point-safe token mode."""

    def chunk(self, text: str) -> List[Chunk]:
        common, options = self.cfg.common, self.cfg.fixed
        size = int(common.get("chunk_size", 512))
        overlap = int(common.get("chunk_overlap", 80))
        max_chunks = common.get("max_chunks")
        validate_parameters(size, overlap, max_chunks)
        use_tokens = bool(options.get("use_tokens", True))

        if not text:
            return []
        enc = _utils_mod._enc
        if use_tokens and enc is not None:
            return self._chunk_by_tokens(text, size, overlap, max_chunks, enc)
        if use_tokens:
            logger.warning(
                "tiktoken unavailable; explicitly falling back to character windows"
            )
        return self._chunk_by_chars(text, size, overlap, max_chunks)

    def _chunk_by_chars(self, text: str, size: int, overlap: int,
                        max_chunks: int | None) -> List[Chunk]:
        chunks: List[Chunk] = []
        cursor = 0
        while cursor < len(text):
            stop = min(cursor + size, len(text))
            chunks.append(Chunk(
                id=len(chunks), text=text[cursor:stop],
                start=cursor, end=stop,
                meta={"strategy": "fixed", "mode": "char"},
            ))
            if stop == len(text) or (
                    max_chunks is not None and len(chunks) >= max_chunks):
                break
            cursor = stop - overlap
        return chunks

    def _chunk_by_tokens(self, text: str, size: int, overlap: int,
                         max_chunks: int | None, enc) -> List[Chunk]:
        """Use real source slices, never decode arbitrary partial UTF-8 tokens."""
        count = lambda value: len(enc.encode(value))
        chunks: List[Chunk] = []
        cursor = 0
        while cursor < len(text):
            stop = largest_fitting_end(text, cursor, size, count)
            chunks.append(Chunk(
                id=len(chunks), text=text[cursor:stop],
                start=cursor, end=stop,
                meta={"strategy": "fixed", "mode": "token"},
            ))
            if stop == len(text) or (
                    max_chunks is not None and len(chunks) >= max_chunks):
                break

            # Reuse a bounded token suffix without stalling forward progress.
            next_start = stop
            for candidate in range(stop - 1, cursor, -1):
                if count(text[candidate:stop]) > overlap:
                    break
                next_start = candidate
            cursor = next_start
        return chunks
