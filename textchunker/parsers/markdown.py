"""Lossless line-block Markdown adapter; NOT a full Markdown AST parser."""

from __future__ import annotations

import re

from ..core.models import (
    DocumentBlock, DocumentIR, SourceMap, document_digest,
)
from ..core.spans import Span


_HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^[ \t]{0,3}(\x60{3,}|~{3,})(.*)$")


def parse_markdown(text: str, source_uri: str = "memory://markdown") -> DocumentIR:
    blocks = []
    cursor = 0
    fence_char, fence_width = None, 0
    for index, line in enumerate(text.splitlines(keepends=True)):
        content = line.rstrip("\r\n")
        kind = "paragraph"
        heading_level, title = None, None
        fence = _FENCE.match(content)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char, fence_width = marker[0], len(marker)
                kind = "code"
            elif marker[0] == fence_char and len(marker) >= fence_width:
                fence_char, fence_width = None, 0
                kind = "code"
            else:
                kind = "code"
        elif fence_char is not None:
            kind = "code"
        else:
            heading = _HEADING.match(content)
            if heading:
                kind = "heading"
                heading_level = len(heading.group(1))
                title = heading.group(2)
        end = cursor + len(line)
        blocks.append(DocumentBlock(
            block_id=f"line:{index}", kind=kind,
            span=Span(cursor, end),
            heading_level=heading_level, title=title,
        ))
        cursor = end
    return DocumentIR(
        document_id=document_digest(source_uri, text),
        text=text, source_uri=source_uri, format="markdown",
        source_map=SourceMap.identity(source_uri, len(text)),
        blocks=tuple(blocks),
        metadata={"parser": "line-block-v0.3", "ast": False},
    )
