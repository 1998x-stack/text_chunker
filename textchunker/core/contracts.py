"""Immutable results for the DocumentIR -> ChunkingResult pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from .models import SourceLocation
from .spans import Span


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    severity: str = "error"
    span: Span | None = None

    def __post_init__(self) -> None:
        if self.severity not in {"error", "warning", "info"}:
            raise ValueError("invalid diagnostic severity")


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    document_id: str
    index: int
    text: str
    span: Span
    token_count: int
    source_locations: Tuple[SourceLocation, ...]
    heading_path: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QualityReport:
    ok: bool
    complete: bool
    unique_coverage_ratio: float
    duplicated_characters: int
    missing_spans: Tuple[Span, ...]
    diagnostics: Tuple[Diagnostic, ...]


@dataclass(frozen=True)
class ChunkingResult:
    document_id: str
    strategy: str
    tokenizer_id: str
    chunks: Tuple[ChunkRecord, ...]
    report: QualityReport

    @property
    def complete(self) -> bool:
        return self.report.complete
