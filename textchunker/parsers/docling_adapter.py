"""Optional native DoclingDocument → DocumentIR bridge.

Walk native document items in reading order. Preserve official page and bbox
provenance, never invent raw source character offsets. No heavy dependency is
imported unless converting a real file with parse_docling_file().
"""
from __future__ import annotations

from typing import Any

from ..core.models import (
    DocumentBlock, DocumentIR, SourceMap, SourceMapEntry, document_digest,
)
from ..core.spans import Span

_HEADINGS = {"title", "section_header"}
_KINDS = {
    "title": "heading", "section_header": "heading",
    "text": "paragraph", "paragraph": "paragraph", "list_item": "list",
    "code": "code", "table": "table", "formula": "formula",
    "caption": "caption", "picture": "picture",
    "footnote": "footnote", "page_header": "header",
    "page_footer": "footer",
}


def _label(item: Any) -> str:
    value = getattr(item, "label", "text")
    return str(getattr(value, "value", value)).lower()


def _bbox(item: Any) -> dict | None:
    provenance = getattr(item, "prov", None) or []
    if not provenance or getattr(provenance[0], "bbox", None) is None:
        return None
    box = provenance[0].bbox
    fields = {key: getattr(box, key, None) for key in ("l", "t", "r", "b")}
    if not all(isinstance(x, (int, float)) for x in fields.values()):
        return None
    origin = getattr(box, "coord_origin", None)
    if origin is not None:
        fields["coord_origin"] = str(getattr(origin, "value", origin))
    return fields


def _text(item: Any, document: Any, kind: str) -> str:
    if kind == "table" and hasattr(item, "export_to_markdown"):
        return str(item.export_to_markdown(doc=document))
    return str(getattr(item, "text", "") or "")


def parse_docling(document: Any, source_uri: str = "memory://docling") -> DocumentIR:
    """Convert a *native DoclingDocument* with iterate_items() to DocumentIR."""
    if not callable(getattr(document, "iterate_items", None)):
        raise TypeError("Expected DoclingDocument with iterate_items()")
    rows = []
    skipped = []
    for index, (item, level) in enumerate(document.iterate_items()):
        kind = _KINDS.get(_label(item), "paragraph")
        content = _text(item, document, kind)
        ref = str(getattr(item, "self_ref", f"docling:{index}"))
        if not content.strip():
            skipped.append(ref)
            continue
        provenance = getattr(item, "prov", None) or []
        page = getattr(provenance[0], "page_no", None) if provenance else None
        if not isinstance(page, int) or page < 1:
            page = None
        heading_level = None
        if kind == "heading":
            level_from_item = getattr(item, "level", None)
            candidate = level_from_item if isinstance(level_from_item, int) else level
            heading_level = min(6, max(1, int(candidate))) if (
                isinstance(candidate, int)
            ) else 1
        rows.append((kind, content, ref, page, _bbox(item), heading_level, int(level)))

    pieces = []
    entries = []
    blocks = []
    cursor = 0
    for index, (kind, content, ref, page, bbox, heading_level, depth) in enumerate(rows):
        # Delimiter belongs to the preceding element, not to nonexistent
        # character offsets in the binary source document.
        part = content + ("\n\n" if index < len(rows) - 1 else "")
        span = Span(cursor, cursor + len(part))
        pieces.append(part)
        entries.append(SourceMapEntry(
            canonical_span=span, source_span=None,
            page_number=page, block_id=ref, bbox=bbox,
        ))
        blocks.append(DocumentBlock(
            block_id=ref, kind=kind, span=span,
            heading_level=heading_level,
            title=content if kind == "heading" else None,
            metadata={"depth": depth, "atomic": kind in {"code", "table"},
                      "page": page, "bbox": bbox},
        ))
        cursor += len(part)
    text = "".join(pieces)
    return DocumentIR(
        document_id=document_digest(source_uri, text),
        text=text, source_uri=source_uri, format="docling",
        source_map=SourceMap(source_uri, len(text), tuple(entries)),
        blocks=tuple(blocks),
        metadata={"parser": "docling-native-iter-v1",
                  "source_offsets": "unavailable",
                  "omitted_nontext_items": tuple(skipped)},
    )


def parse_docling_file(path: str) -> DocumentIR:
    """Opt-in heavy conversion. No implicit OCR, download, or network call."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ImportError(
            "Install Docling explicitly with pip install -e '.[docling]'"
        ) from exc
    conversion = DocumentConverter().convert(path)
    if not getattr(conversion, "document", None):
        raise ValueError("Docling did not return a document")
    return parse_docling(conversion.document, source_uri=path)
