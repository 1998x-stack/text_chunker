"""Independent structural quality checks, separate from the hybrid algorithm."""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

from ..core.contracts import Diagnostic
from ..core.models import DocumentIR
from ..core.spans import Span
from ..types import Chunk


def evaluate_structure(
    document: DocumentIR, chunks: Sequence[Chunk],
    count: Callable[[str], int], max_size: int,
) -> Tuple[Diagnostic, ...]:
    issues = []
    for block in document.blocks:
        if block.kind == "heading":
            # A chunk must never bridge across a *new* heading boundary.
            for chunk in chunks:
                if chunk.start < block.span.start < chunk.end:
                    issues.append(Diagnostic(
                        code="HEADING_CROSSING",
                        message=f"Chunk {chunk.id} crosses heading {block.block_id}",
                        span=block.span,
                    ))
                    break
        if block.kind not in {"code", "table"}:
            continue
        source = document.text[block.span.start:block.span.end]
        if count(source) <= max_size:
            intact = any(
                chunk.start <= block.span.start and chunk.end >= block.span.end
                for chunk in chunks
            )
            # An intentional max_chunks limit may omit whole later blocks.
            if not intact and any(
                chunk.start < block.span.end and chunk.end > block.span.start
                for chunk in chunks
            ):
                issues.append(Diagnostic(
                    code="ATOMIC_SPLIT",
                    message=f"Fits budget but was split: {block.block_id}",
                    span=block.span,
                ))
        elif any(
            chunk.start < block.span.end and chunk.end > block.span.start
            for chunk in chunks
        ):
            issues.append(Diagnostic(
                code="ATOMIC_OVERFLOW",
                message=f"Oversized {block.kind} split: {block.block_id}",
                severity="warning", span=block.span,
            ))
    return tuple(issues)
