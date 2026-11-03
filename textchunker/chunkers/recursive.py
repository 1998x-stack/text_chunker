"""Separator-aware chunking with lossless Unicode source spans."""

from __future__ import annotations

from typing import List

from loguru import logger

from ..core.spans import Span
from ..core.token_budget import largest_fitting_end, overlap_start
from ..core.validation import validate_parameters
from ..exceptions import ChunkerError
from ..registry import register
from ..types import Chunk
from ..utils import count_tokens
from .base import BaseChunker


def _merge_spans(text: str, spans: List[Span], size: int, count=None) -> List[Span]:
    """Greedily merge contiguous segments without exceeding token budget."""
    counter = count or count_tokens
    merged: List[Span] = []
    for span in spans:
        if merged and merged[-1].end != span.start:
            raise ChunkerError("A recursive split introduced a source gap")
        if merged and counter(text[merged[-1].start:span.end]) <= size:
            previous = merged.pop()
            merged.append(Span(previous.start, span.end))
        else:
            merged.append(span)
    return merged


def _split_span(text: str, start: int, end: int,
                seps: List[str], size: int, count=None) -> List[Span]:
    counter = count or count_tokens
    if start == end:
        return []
    if counter(text[start:end]) <= size:
        return [Span(start, end)]

    if not seps or seps[0] == "":
        results: List[Span] = []
        cursor = start
        while cursor < end:
            # Limit the search to this source interval, not later document text.
            stop = largest_fitting_end(text[:end], cursor, size, counter)
            results.append(Span(cursor, stop))
            cursor = stop
        return results

    sep = seps[0]
    units: List[Span] = []
    cursor = start
    while cursor < end:
        idx = text.find(sep, cursor, end)
        stop = idx + len(sep) if idx != -1 else end
        units.append(Span(cursor, stop))
        cursor = stop

    if len(units) == 1:
        return _split_span(text, start, end, seps[1:], size, counter)

    refined: List[Span] = []
    for item in units:
        if counter(item.extract(text)) <= size:
            refined.append(item)
        else:
            refined.extend(_split_span(text, item.start, item.end,
                                       seps[1:], size, counter))
    return _merge_spans(text, refined, size, counter)


def _split_spans(text: str, seps: List[str], size: int, count=None) -> List[Span]:
    if not text:
        return []
    return _split_span(text, 0, len(text), seps, size, count)


def _split_by_separators(text: str, seps: List[str], size: int, count=None) -> List[str]:
    """Legacy helper retained for structure chunker and external callers."""
    return [span.extract(text) for span in _split_spans(text, seps, size, count)]


def _build_offset_map(text: str, parts: List[str]) -> List[int]:
    """Strict cumulative offsets: missing or reordered content is an error."""
    offsets: List[int] = []
    cursor = 0
    for part in parts:
        if not text.startswith(part, cursor):
            raise ChunkerError("Parts do not match contiguous source text")
        offsets.append(cursor)
        cursor += len(part)
    if cursor != len(text):
        raise ChunkerError("Parts did not cover the entire source text")
    return offsets


def add_bounded_overlap(text: str, chunks: List[Chunk], size: int,
                        overlap: int, same_title_only: bool = False,
                        count=None) -> None:
    """Mutate chunk offsets/text, preserving span integrity and size limits."""
    counter = count or count_tokens
    if not overlap:
        return
    for index in range(1, len(chunks)):
        prev, cur = chunks[index - 1], chunks[index]
        if same_title_only and prev.meta.get("title") != cur.meta.get("title"):
            continue
        begin = overlap_start(text, prev.start, cur.start, cur.end,
                              overlap, size, counter)
        cur.start = begin
        cur.text = text[begin:cur.end]


@register("recursive")
class RecursiveChunker(BaseChunker):
    """Paragraph > line > word > Unicode character splitting."""

    def chunk(self, text: str) -> List[Chunk]:
        common = self.cfg.common
        size = int(common.get("chunk_size", 512))
        overlap = int(common.get("chunk_overlap", 80))
        max_chunks = common.get("max_chunks")
        validate_parameters(size, overlap, max_chunks)
        seps = list(self.cfg.recursive.get(
            "separators", ["\n\n", "\n", " ", ""]
        ))

        spans = _split_spans(text, seps, size, self.count_tokens)
        if max_chunks is not None:
            spans = spans[:max_chunks]
        chunks = [
            Chunk(id=i, text=span.extract(text), start=span.start,
                  end=span.end, meta={"strategy": "recursive"})
            for i, span in enumerate(spans)
        ]
        add_bounded_overlap(text, chunks, size, overlap,
                            count=self.count_tokens)
        logger.info("Recursive chunking: {} chunks, size={}, overlap={}",
                    len(chunks), size, overlap)
        return chunks
