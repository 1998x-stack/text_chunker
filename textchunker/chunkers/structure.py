"""Structure-first Markdown chunker preserving source coverage."""

from __future__ import annotations

import re
from typing import List, Tuple

from loguru import logger

from ..core.validation import validate_parameters
from ..exceptions import ChunkerError
from ..registry import register
from ..types import Chunk
from ..utils import count_tokens
from .base import BaseChunker
from .recursive import _split_by_separators, add_bounded_overlap


@register("structure")
class StructureChunker(BaseChunker):
    """Markdown heading sections, including any text before first heading.

    HTML input is currently treated as one section. A proper AST-based parser
    is planned for v0.4; do not mistake this fallback for structural HTML.
    """

    def _md_sections(self, text: str) -> List[Tuple[str, int, int]]:
        if not text:
            return []
        lines = text.splitlines(keepends=True)
        positions = [0]
        for line in lines:
            positions.append(positions[-1] + len(line))
        heads = [
            index for index, line in enumerate(lines)
            if re.match(r"^[ \t]{0,3}#{1,6}[ \t]+", line)
        ]
        if not heads:
            return [("Document", 0, len(text))]

        sections: List[Tuple[str, int, int]] = []
        if positions[heads[0]] > 0:
            sections.append(("Preamble", 0, positions[heads[0]]))
        for position, head in enumerate(heads):
            stop_line = heads[position + 1] if position + 1 < len(heads) else len(lines)
            title = re.sub(r"^[ \t]{0,3}#{1,6}[ \t]+", "", lines[head]).strip()
            sections.append((title, positions[head], positions[stop_line]))
        return sections

    def _html_sections(self, text: str) -> List[Tuple[str, int, int]]:
        return [("HTML (unparsed)", 0, len(text))] if text else []

    def chunk(self, text: str) -> List[Chunk]:
        common, options = self.cfg.common, self.cfg.structure
        size = int(common.get("chunk_size", 800))
        overlap = int(common.get("chunk_overlap", 100))
        max_chunks = common.get("max_chunks")
        validate_parameters(size, overlap, max_chunks)
        prefer = options.get("prefer", "auto")
        sub_split = bool(options.get("sub_split", True))
        sections = (self._html_sections(text) if prefer == "html"
                    else self._md_sections(text))

        chunks: List[Chunk] = []
        for title, begin, end in sections:
            source = text[begin:end]
            if count_tokens(source) > size and not sub_split:
                raise ChunkerError(
                    "Section exceeds budget while sub_split is disabled"
                )
            if count_tokens(source) <= size:
                parts = [source]
            else:
                parts = _split_by_separators(
                    source, ["\n\n", "\n", " ", ""], size
                )

            cursor = begin
            for part in parts:
                chunks.append(Chunk(
                    id=len(chunks), text=part, start=cursor,
                    end=cursor + len(part),
                    meta={"strategy": "structure", "title": title},
                ))
                cursor += len(part)
                if max_chunks is not None and len(chunks) >= max_chunks:
                    break
            if max_chunks is not None and len(chunks) >= max_chunks:
                break

        add_bounded_overlap(text, chunks, size, overlap, same_title_only=True)
        logger.info("Structure chunking: {} chunks from {} sections",
                    len(chunks), len(sections))
        return chunks
