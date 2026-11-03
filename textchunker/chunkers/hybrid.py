"""Structure-aware, budget-safe chunking over canonical DocumentIR blocks.

Heading boundaries are hard section boundaries. Markdown tables/code fences
and HTML tables/pre blocks remain atomic when they fit; oversized elements
are deterministically split with visible fallback metadata.
"""
from __future__ import annotations

from typing import List

from loguru import logger

from ..core.models import DocumentBlock, DocumentIR
from ..core.spans import Span
from ..core.token_budget import overlap_start
from ..core.validation import validate_parameters
from ..exceptions import ChunkerError, ConfigError
from ..parsers.markdown import parse_markdown
from ..registry import register
from ..types import Chunk
from .base import BaseChunker
from .recursive import _split_spans


_ATOMIC = frozenset({"table", "code"})


def _ordered_blocks(document: DocumentIR) -> List[DocumentBlock]:
    """Fill parser gaps rather than silently discarding canonical characters."""
    blocks: List[DocumentBlock] = []
    cursor = 0
    for block in document.blocks:
        if block.span.start < cursor:
            raise ChunkerError("DocumentIR has overlapping structural blocks")
        if block.span.start > cursor:
            blocks.append(DocumentBlock(
                block_id=f"raw:{cursor}", kind="raw",
                span=Span(cursor, block.span.start),
            ))
        if block.span.end > block.span.start:
            blocks.append(block)
        cursor = block.span.end
    if cursor < len(document.text):
        blocks.append(DocumentBlock(
            block_id=f"raw:{cursor}", kind="raw",
            span=Span(cursor, len(document.text)),
        ))
    return blocks


@register("hybrid")
class HybridChunker(BaseChunker):
    """Greedily pack sections, then recurse only for oversized blocks."""

    def chunk(self, text: str) -> List[Chunk]:
        """Legacy string API assumes Markdown; prefer chunk_document(DocumentIR)."""
        return self.chunk_document(parse_markdown(text))

    def chunk_document(self, document: DocumentIR) -> List[Chunk]:
        size = int(self.cfg.common.get("chunk_size", 512))
        overlap = int(self.cfg.common.get("chunk_overlap", 80))
        max_chunks = self.cfg.common.get("max_chunks")
        validate_parameters(size, overlap, max_chunks)

        options = self.cfg.hybrid
        overflow_policy = options.get("oversized_element", "split")
        if overflow_policy not in {"split", "error"}:
            raise ConfigError("hybrid.oversized_element must be split or error")
        if (document.format == "markdown"
                and not document.metadata.get("ast", False)):
            raise ConfigError(
                "Hybrid Markdown requires markdown-it-py; "
                "install structure extras: pip install -e '.[structure]'"
            )

        text, count = document.text, self.count_tokens
        blocks = _ordered_blocks(document)
        chunks: List[Chunk] = []
        pending: List[DocumentBlock] = []
        section = 0

        def emit(start: int, end: int, group: List[DocumentBlock],
                 oversized: str | None = None) -> None:
            kinds = [item.kind for item in group]
            chunks.append(Chunk(
                id=len(chunks), text=text[start:end], start=start, end=end,
                meta={
                    "strategy": "hybrid",
                    "section": section,
                    "block_ids": [item.block_id for item in group],
                    "block_kinds": kinds,
                    "contains_atomic": any(k in _ATOMIC for k in kinds),
                    "oversized_block": oversized,
                    "atomic_overflow": bool(oversized),
                },
            ))

        def flush() -> None:
            nonlocal pending
            if pending:
                emit(pending[0].span.start, pending[-1].span.end, pending)
                pending = []

        for block in blocks:
            if block.kind == "heading":
                flush()
                section += 1

            source = text[block.span.start:block.span.end]
            if count(source) > size:
                flush()
                if overflow_policy == "error" and block.kind in _ATOMIC:
                    raise ChunkerError(
                        f"Atomic {block.kind} {block.block_id} exceeds budget {size}"
                    )
                spans = _split_spans(
                    source, ["\n\n", "\n", " ", ""], size, count
                )
                for span in spans:
                    emit(
                        block.span.start + span.start,
                        block.span.start + span.end,
                        [block],
                        oversized=block.block_id if block.kind in _ATOMIC else None,
                    )
                continue

            if pending:
                combined = text[pending[0].span.start:block.span.end]
                if count(combined) > size:
                    flush()
            pending.append(block)
        flush()

        # Overlap is permitted only inside a section and does not cross a
        # code/table boundary. Overflow segments may overlap each other.
        if overlap:
            for index in range(1, len(chunks)):
                previous, current = chunks[index - 1], chunks[index]
                if previous.meta["section"] != current.meta["section"]:
                    continue
                left_over = previous.meta["oversized_block"]
                right_over = current.meta["oversized_block"]
                if left_over or right_over:
                    if not left_over or left_over != right_over:
                        continue
                elif (previous.meta["contains_atomic"]
                      or current.meta["contains_atomic"]):
                    continue
                new_start = overlap_start(
                    text, previous.start, current.start, current.end,
                    overlap, size, count,
                )
                current.start = new_start
                current.text = text[new_start:current.end]

        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        logger.info("Hybrid chunking: {} chunks, {} document blocks",
                    len(chunks), len(blocks))
        return chunks
