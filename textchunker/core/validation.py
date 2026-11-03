"""Correctness contracts shared by every strategy and evaluation pipeline."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from ..exceptions import ChunkerError, ConfigError
from ..types import Chunk


def validate_parameters(size: int, overlap: int,
                        max_chunks: Optional[int] = None) -> None:
    if size <= 0:
        raise ConfigError("chunk_size must be positive")
    if overlap < 0 or overlap >= size:
        raise ConfigError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
    if max_chunks is not None and (not isinstance(max_chunks, int)
                                   or max_chunks <= 0):
        raise ConfigError("max_chunks must be a positive integer or null")


def validate_chunks(text: str, chunks: Sequence[Chunk],
                    count: Optional[Callable[[str], int]] = None,
                    max_size: Optional[int] = None,
                    require_full_coverage: bool = True) -> None:
    """Reject missing source intervals, broken offsets and oversized chunks.

    Overlap is legal; every source character must appear in at least one
    chunk when require_full_coverage is true.
    """
    covered_until = 0
    last_start = -1
    for index, chunk in enumerate(chunks):
        if chunk.id != index:
            raise ChunkerError(f"Chunk id mismatch at index {index}")
        if not (0 <= chunk.start < chunk.end <= len(text)):
            raise ChunkerError(f"Invalid source span in chunk {index}")
        if chunk.start < last_start:
            raise ChunkerError(f"Out-of-order source spans at chunk {index}")
        if text[chunk.start:chunk.end] != chunk.text:
            raise ChunkerError(f"Source mapping mismatch in chunk {index}")
        if require_full_coverage and chunk.start > covered_until:
            raise ChunkerError(f"Uncovered source interval starts at {covered_until}")
        if max_size is not None and count is not None:
            if count(chunk.text) > max_size:
                raise ChunkerError(f"Chunk {index} exceeds budget {max_size}")
        last_start = chunk.start
        covered_until = max(covered_until, chunk.end)

    if require_full_coverage and covered_until != len(text):
        raise ChunkerError(
            f"Document not fully covered: covered [0,{covered_until}) "
            f"of [0,{len(text)})"
        )
