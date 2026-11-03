"""Canonical document and source provenance models.

Canonical spans index Python Unicode code points in DocumentIR.text, never
UTF-8 bytes or raw PDF byte offsets. Extracted formats retain explicit
unknown original offsets instead of inventing a raw-source mapping.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple

from .spans import Span


@dataclass(frozen=True)
class SourceLocation:
    source_uri: str
    canonical_span: Span
    source_span: Optional[Span] = None
    page_number: Optional[int] = None
    block_id: Optional[str] = None
    mapping: str = "extracted"
    bbox: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.mapping not in {"exact", "extracted"}:
            raise ValueError("mapping must be exact or extracted")
        if self.mapping == "exact" and self.source_span is None:
            raise ValueError("exact provenance requires a source span")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number is 1-based")


@dataclass(frozen=True)
class SourceMapEntry:
    canonical_span: Span
    source_span: Optional[Span] = None
    page_number: Optional[int] = None
    block_id: Optional[str] = None
    bbox: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.source_span is not None and (
                self.source_span.end - self.source_span.start
                != self.canonical_span.end - self.canonical_span.start):
            raise ValueError("exact offset maps require equal span lengths")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number is 1-based")


@dataclass(frozen=True)
class SourceMap:
    source_uri: str
    text_length: int
    entries: Tuple[SourceMapEntry, ...] = ()

    def __post_init__(self) -> None:
        if self.text_length < 0:
            raise ValueError("negative document length")
        cursor = 0
        for entry in self.entries:
            if entry.canonical_span.start != cursor:
                raise ValueError("source map must be contiguous and ordered")
            if entry.canonical_span.end <= cursor:
                raise ValueError("empty source map entry")
            cursor = entry.canonical_span.end
        if cursor != self.text_length:
            raise ValueError("source map must cover entire canonical text")

    @classmethod
    def identity(cls, source_uri: str, length: int) -> SourceMap:
        if length < 0:
            raise ValueError("negative document length")
        entries = (SourceMapEntry(Span(0, length), Span(0, length)),) if length else ()
        return cls(source_uri=source_uri, text_length=length, entries=entries)

    @classmethod
    def extracted(cls, source_uri: str, length: int) -> SourceMap:
        if length < 0:
            raise ValueError("negative document length")
        entries = (SourceMapEntry(Span(0, length)),) if length else ()
        return cls(source_uri=source_uri, text_length=length, entries=entries)

    def resolve(self, span: Span) -> Tuple[SourceLocation, ...]:
        if span.end > self.text_length:
            raise ValueError("span outside canonical text")
        if span.start == span.end:
            return ()
        locations = []
        cursor = span.start
        for entry in self.entries:
            begin = max(span.start, entry.canonical_span.start)
            end = min(span.end, entry.canonical_span.end)
            if begin >= end:
                continue
            if begin != cursor:
                raise ValueError("unmapped canonical span")
            origin = None
            if entry.source_span is not None:
                delta = begin - entry.canonical_span.start
                origin = Span(entry.source_span.start + delta,
                              entry.source_span.start + delta + end - begin)
            locations.append(SourceLocation(
                source_uri=self.source_uri, canonical_span=Span(begin, end),
                source_span=origin, page_number=entry.page_number,
                block_id=entry.block_id,
                mapping="exact" if origin is not None else "extracted",
                bbox=entry.bbox,
            ))
            cursor = end
        if cursor != span.end:
            raise ValueError("unmapped canonical span")
        return tuple(locations)


@dataclass(frozen=True)
class DocumentBlock:
    block_id: str
    kind: str
    span: Span
    heading_level: Optional[int] = None
    title: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.block_id or not self.kind:
            raise ValueError("block_id and kind must not be empty")
        if self.heading_level is not None and not 1 <= self.heading_level <= 6:
            raise ValueError("heading_level must be between 1 and 6")


@dataclass(frozen=True)
class DocumentIR:
    document_id: str
    text: str
    source_uri: str
    format: str
    source_map: SourceMap
    blocks: Tuple[DocumentBlock, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.document_id or not self.source_uri:
            raise ValueError("document_id and source_uri are required")
        if self.source_map.source_uri != self.source_uri:
            raise ValueError("source map URI mismatch")
        if self.source_map.text_length != len(self.text):
            raise ValueError("source map length mismatch")
        last_end = 0
        ids = set()
        for block in self.blocks:
            if block.block_id in ids:
                raise ValueError("duplicate block_id")
            if block.span.start < last_end or block.span.end > len(self.text):
                raise ValueError("document blocks overlap or exceed text")
            last_end = block.span.end
            ids.add(block.block_id)


def document_digest(source_uri: str, text: str) -> str:
    return hashlib.sha256(
        (source_uri + "\0" + text).encode("utf-8")
    ).hexdigest()


def chunk_digest(document_id: str, span: Span, text: str) -> str:
    content = f"{document_id}\0{span.start}:{span.end}\0{text}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
