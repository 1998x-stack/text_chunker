from __future__ import annotations

import re
from typing import List, Tuple

from loguru import logger

from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..utils import count_tokens
from .recursive import _split_by_separators


def _build_offset_map(text: str, parts: List[str]) -> List[int]:
    """Build start offsets using cumulative cursor (safe for duplicates)."""
    offsets: List[int] = []
    cursor = 0
    for part in parts:
        idx = text.find(part, cursor)
        if idx == -1:
            idx = cursor
        offsets.append(idx)
        cursor = idx + len(part)
    return offsets


@register("structure")
class StructureChunker(BaseChunker):
    """Structure-based chunker: splits by document headings (Markdown/HTML)."""

    def _md_sections(self, text: str) -> List[Tuple[str, int, int]]:
        lines = text.splitlines(keepends=True)
        heads: List[int] = []
        for i, ln in enumerate(lines):
            if re.match(r"^\s*#{1,6}\s+", ln):
                heads.append(i)
        heads.append(len(lines))
        spans: List[Tuple[str, int, int]] = []
        for i in range(len(heads) - 1):
            beg_line = heads[i]
            end_line = heads[i + 1]
            beg = sum(len(l) for l in lines[:beg_line])
            end = sum(len(l) for l in lines[:end_line])
            title = re.sub(r"^#+\s+", "", lines[beg_line]).strip()
            spans.append((title, beg, end))
        if not spans:
            spans = [("Document", 0, len(text))]
        return spans

    def _html_sections(self, text: str) -> List[Tuple[str, int, int]]:
        return [("HTML", 0, len(text))]

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        st = self.cfg.structure
        size = int(c.get("chunk_size", 800))
        overlap = int(c.get("chunk_overlap", 100))
        prefer = st.get("prefer", "auto")
        sub_split = bool(st.get("sub_split", True))
        max_chunks = c.get("max_chunks")

        if prefer in {"md", "auto"}:
            spans = self._md_sections(text)
        elif prefer == "html":
            spans = self._html_sections(text)
        else:
            spans = self._md_sections(text)

        chunks: List[Chunk] = []
        for title, beg, end in spans:
            seg = text[beg:end]
            if count_tokens(seg) <= size or not sub_split:
                chunks.append(Chunk(
                    id=len(chunks), text=seg, start=beg, end=end,
                    meta={"strategy": "structure", "title": title},
                ))
            else:
                parts = _split_by_separators(seg, ["\n\n", "\n", " ", ""], size)
                offsets = _build_offset_map(seg, parts)
                for j, p in enumerate(parts):
                    s = beg + offsets[j]
                    e = s + len(p)
                    chunks.append(Chunk(
                        id=len(chunks), text=p, start=s, end=e,
                        meta={"strategy": "structure", "title": title},
                    ))
            if max_chunks and len(chunks) >= max_chunks:
                break

        if overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev, cur = chunks[i - 1], chunks[i]
                head_start = max(cur.start - overlap, prev.start)
                head = text[head_start:cur.start]
                cur.text = head + cur.text
                cur.start = head_start

        logger.info("Structure chunking: {} chunks from {} sections", len(chunks), len(spans))
        return chunks
