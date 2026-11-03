"""Extracted PDF page text with page provenance, no raw-PDF offsets."""

from __future__ import annotations

from typing import Sequence

from ..core.models import (
    DocumentBlock, DocumentIR, SourceMap, SourceMapEntry, document_digest,
)
from ..core.spans import Span


def parse_pdf_pages(
    pages: Sequence[str], source_uri: str = "memory://pdf",
) -> DocumentIR:
    """Page numbering starts at 1; separators belong to prior page.

    Empty pages are represented by no character interval and are retained in
    document metadata rather than fabricated zero-length source entries.
    """
    pieces = []
    entries = []
    blocks = []
    cursor = 0
    for index, page in enumerate(pages):
        is_last = index == len(pages) - 1
        content = page + ("" if is_last else "\n\n")
        pieces.append(content)
        if content:
            entry = SourceMapEntry(
                canonical_span=Span(cursor, cursor + len(content)),
                source_span=None, page_number=index + 1,
                block_id=f"page:{index + 1}",
            )
            entries.append(entry)
            blocks.append(DocumentBlock(
                block_id=f"page:{index + 1}", kind="page",
                span=entry.canonical_span,
            ))
            cursor += len(content)
    text = "".join(pieces)
    return DocumentIR(
        document_id=document_digest(source_uri, text),
        text=text, source_uri=source_uri, format="pdf",
        source_map=SourceMap(source_uri, len(text), tuple(entries)),
        blocks=tuple(blocks),
        metadata={"page_count": len(pages), "source_offsets": "unavailable"},
    )
