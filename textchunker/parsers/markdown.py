"""MarkdownIt-backed, source-lossless top-level structural blocks.

We use CommonMark/GFM table token maps for block boundaries, rather than
normalizing rendered Markdown back into text. All canonical characters,
including blank lines and delimiters, remain in DocumentIR.text.
"""
from __future__ import annotations

import re

from ..core.models import (
    DocumentBlock, DocumentIR, SourceMap, document_digest,
)
from ..core.spans import Span


_HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)\s*$")
_FENCE = re.compile(r"^[ \t]{0,3}(\x60{3,}|~{3,})(.*)$")
_TOP_LEVEL = {
    "heading_open": "heading",
    "paragraph_open": "paragraph",
    "fence": "code",
    "code_block": "code",
    "table_open": "table",
    "bullet_list_open": "list",
    "ordered_list_open": "list",
    "blockquote_open": "quote",
    "hr": "thematic_break",
    "html_block": "html",
}


def _line_blocks(text: str) -> tuple[DocumentBlock, ...]:
    """Dependency-free fallback: retain full offsets and fenced-code safety."""
    blocks = []
    cursor, fence_char, fence_width = 0, None, 0
    for index, line in enumerate(text.splitlines(keepends=True)):
        content = line.rstrip("\r\n")
        kind, level, title = "paragraph", None, None
        fence = _FENCE.match(content)
        if fence:
            marker = fence.group(1)
            if fence_char is None:
                fence_char, fence_width = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_width:
                fence_char, fence_width = None, 0
            kind = "code"
        elif fence_char is not None:
            kind = "code"
        else:
            heading = _HEADING.match(content)
            if heading:
                kind, level = "heading", len(heading.group(1))
                title = heading.group(2)
        end = cursor + len(line)
        blocks.append(DocumentBlock(
            block_id=f"line:{index}", kind=kind, span=Span(cursor, end),
            heading_level=level, title=title,
        ))
        cursor = end
    return tuple(blocks)


def _markdown_it_blocks(text: str) -> tuple[DocumentBlock, ...]:
    from markdown_it import MarkdownIt

    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    tokens = MarkdownIt("commonmark").enable("table").parse(text)
    blocks = []
    cursor = 0
    for index, token in enumerate(tokens):
        kind = _TOP_LEVEL.get(token.type)
        if kind is None or token.level != 0 or token.map is None:
            continue
        begin_line, end_line = token.map
        if not (0 <= begin_line < end_line <= len(lines)):
            continue
        begin, end = offsets[begin_line], offsets[end_line]
        if begin < cursor:
            # A parent token already owns this text. Never duplicate spans.
            if end <= cursor:
                continue
            begin = cursor
        if begin > cursor:
            blocks.append(DocumentBlock(
                block_id=f"gap:{len(blocks)}", kind="whitespace",
                span=Span(cursor, begin),
            ))
        level, title = None, None
        if kind == "heading":
            level = int(token.tag[1:]) if token.tag.startswith("h") else None
            if index + 1 < len(tokens) and tokens[index + 1].type == "inline":
                title = tokens[index + 1].content
        blocks.append(DocumentBlock(
            block_id=f"md:{len(blocks)}", kind=kind,
            span=Span(begin, end), heading_level=level, title=title,
            metadata={"atomic": kind in {"table", "code"},
                      "token_type": token.type},
        ))
        cursor = end
    if cursor < len(text):
        blocks.append(DocumentBlock(
            block_id=f"gap:{len(blocks)}", kind="whitespace",
            span=Span(cursor, len(text)),
        ))
    return tuple(blocks)


def parse_markdown(text: str,
                   source_uri: str = "memory://markdown") -> DocumentIR:
    parser = "line-block-fallback"
    ast = False
    try:
        blocks = _markdown_it_blocks(text)
        parser = "markdown-it-py"
        ast = True
    except ImportError:
        blocks = _line_blocks(text)
    return DocumentIR(
        document_id=document_digest(source_uri, text),
        text=text, source_uri=source_uri, format="markdown",
        source_map=SourceMap.identity(source_uri, len(text)),
        blocks=blocks, metadata={"parser": parser, "ast": ast},
    )
