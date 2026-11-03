"""Independent correctness evaluator; does not trust chunker metadata.

Evaluation only consumes canonical text, legacy chunks and an explicit counter.
It never calls the implementation's source-span or token-budget validators.
"""
from __future__ import annotations

from typing import Callable, Sequence

from ..core.contracts import Diagnostic, QualityReport
from ..core.spans import Span
from ..types import Chunk


def evaluate_correctness(
    text: str, chunks: Sequence[Chunk], count: Callable[[str], int],
    max_size: int, require_full_coverage: bool = True,
) -> QualityReport:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    diagnostics = []
    intervals = []
    last_start = -1

    for index, chunk in enumerate(chunks):
        if chunk.id != index:
            diagnostics.append(Diagnostic(
                "CHUNK_ID", f"Unexpected chunk index at {index}"
            ))
        if not (0 <= chunk.start < chunk.end <= len(text)):
            diagnostics.append(Diagnostic(
                "SPAN_BOUNDS", f"Out of bounds span at {index}"
            ))
            continue
        if chunk.start < last_start:
            diagnostics.append(Diagnostic(
                "SPAN_ORDER", f"Out of order chunk at {index}"
            ))
        last_start = chunk.start

        if text[chunk.start:chunk.end] != chunk.text:
            diagnostics.append(Diagnostic(
                "SOURCE_MISMATCH", f"Chunk {index} differs from source",
                span=Span(chunk.start, chunk.end),
            ))
        if count(chunk.text) > max_size:
            diagnostics.append(Diagnostic(
                "BUDGET_EXCEEDED", f"Chunk {index} exceeds budget",
                span=Span(chunk.start, chunk.end),
            ))
        intervals.append((chunk.start, chunk.end))

    intervals.sort()
    cursor, covered, total = 0, 0, 0
    missing = []
    for start, end in intervals:
        if start > cursor:
            missing.append(Span(cursor, start))
        covered += max(0, end - max(start, cursor))
        total += end - start
        cursor = max(cursor, end)
    if cursor < len(text):
        missing.append(Span(cursor, len(text)))
    if len(text) == 0:
        coverage = 1.0
    else:
        coverage = covered / len(text)
    complete = not missing

    if missing:
        diagnostics.append(Diagnostic(
            "INCOMPLETE", f"{len(missing)} uncovered source interval(s)",
            severity="error" if require_full_coverage else "warning",
            span=missing[0],
        ))
    ok = all(item.severity != "error" for item in diagnostics)
    return QualityReport(
        ok=ok, complete=complete,
        unique_coverage_ratio=coverage,
        duplicated_characters=max(0, total - covered),
        missing_spans=tuple(missing),
        diagnostics=tuple(diagnostics),
    )
