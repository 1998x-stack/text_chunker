"""DOM-aware HTML-to-DocumentIR adapter with honest source provenance.

Canonical text is intentionally extracted/normalized. Source character
offsets into raw HTML are unknown; each span is anchored to a DOM block id.
"""
from __future__ import annotations

from ..core.models import (
    DocumentBlock, DocumentIR, SourceMap, SourceMapEntry, document_digest,
)
from ..core.spans import Span


_TAG_KINDS = {
    "h1": "heading", "h2": "heading", "h3": "heading",
    "h4": "heading", "h5": "heading", "h6": "heading",
    "p": "paragraph", "pre": "code", "table": "table",
    "li": "list", "blockquote": "quote",
}
_BLOCK_TAGS = set(_TAG_KINDS)


def _table_text(table) -> str:
    rows = []
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if cells:
            rows.append(" | ".join(cell.get_text(" ", strip=True)
                                   for cell in cells))
    return "\n".join(rows) if rows else table.get_text(" ", strip=True)


def _dom_blocks(soup):
    """Select outermost semantic blocks, never double-count nested children."""
    for element in soup.find_all(list(_BLOCK_TAGS)):
        if any(ancestor.name in _BLOCK_TAGS for ancestor in element.parents):
            continue
        kind = _TAG_KINDS[element.name]
        if kind == "table":
            content = _table_text(element)
        elif kind == "code":
            content = element.get_text()
        else:
            content = element.get_text(" ", strip=True)
        if content:
            yield kind, content, element.name


def parse_html(html: str, source_uri: str = "memory://html") -> DocumentIR:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "Install HTML dependencies with pip install -e '.[structure]'"
        ) from exc

    soup = BeautifulSoup(html, "html.parser")
    for excluded in soup(["script", "style", "template"]):
        excluded.decompose()

    elements = list(_dom_blocks(soup))
    if not elements:
        plain = soup.get_text(" ", strip=True)
        elements = [("paragraph", plain, "fallback")] if plain else []

    pieces, entries, blocks = [], [], []
    cursor = 0
    for index, (kind, content, tag) in enumerate(elements):
        part = content + ("\n\n" if index + 1 < len(elements) else "")
        block_id = f"dom:{index}:{tag}"
        span = Span(cursor, cursor + len(part))
        pieces.append(part)
        entries.append(SourceMapEntry(
            canonical_span=span, block_id=block_id,
        ))
        blocks.append(DocumentBlock(
            block_id=block_id, kind=kind, span=span,
            title=content if kind == "heading" else None,
            heading_level=int(tag[1:]) if kind == "heading" else None,
            metadata={"tag": tag, "atomic": kind in {"code", "table"}},
        ))
        cursor += len(part)
    text = "".join(pieces)
    return DocumentIR(
        document_id=document_digest(source_uri, text),
        text=text, source_uri=source_uri, format="html",
        source_map=SourceMap(source_uri, len(text), tuple(entries)),
        blocks=tuple(blocks),
        metadata={"parser": "bs4-dom-v0.4", "source_offsets": "unavailable"},
    )
