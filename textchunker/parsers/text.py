"""Plain text and already-extracted text parsers."""

from __future__ import annotations

from ..core.models import (
    DocumentBlock, DocumentIR, SourceMap, document_digest,
)
from ..core.spans import Span


def parse_text(text: str, source_uri: str = "memory://text") -> DocumentIR:
    blocks = (
        DocumentBlock("block:0", "text", Span(0, len(text))),
    ) if text else ()
    return DocumentIR(
        document_id=document_digest(source_uri, text),
        text=text, source_uri=source_uri, format="text",
        source_map=SourceMap.identity(source_uri, len(text)),
        blocks=blocks,
    )


def parse_extracted_text(
    text: str, source_uri: str, format: str,
) -> DocumentIR:
    """Text exists but no truthful raw-source character offsets are known."""
    blocks = (
        DocumentBlock("block:0", "extracted_text", Span(0, len(text))),
    ) if text else ()
    return DocumentIR(
        document_id=document_digest(source_uri, text),
        text=text, source_uri=source_uri, format=format,
        source_map=SourceMap.extracted(source_uri, len(text)),
        blocks=blocks,
        metadata={"source_offsets": "unavailable"},
    )
