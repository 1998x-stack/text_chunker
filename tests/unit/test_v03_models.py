"""DocumentIR and provenance invariants."""
from dataclasses import replace

import pytest

from textchunker.core.models import (
    DocumentBlock, DocumentIR, SourceMap, SourceMapEntry,
)
from textchunker.core.spans import Span
from textchunker.parsers import (
    parse_document, parse_html, parse_markdown, parse_pdf_pages, parse_text,
)


def test_text_offsets_exact_with_unicode():
    doc = parse_text("A中🚀B", "docs/a.txt")
    assert doc.source_map.resolve(Span(1, 3))[0].source_span == Span(1, 3)
    assert doc.source_map.resolve(Span(1, 3))[0].mapping == "exact"
    assert len(doc.blocks) == 1


def test_source_map_splits_at_original_segment_boundaries():
    mapping = SourceMap(
        "x", 8, (
            SourceMapEntry(Span(0, 4), Span(10, 14)),
            SourceMapEntry(Span(4, 8), Span(20, 24)),
        ),
    )
    locations = mapping.resolve(Span(2, 7))
    assert [(part.canonical_span, part.source_span) for part in locations] == [
        (Span(2, 4), Span(12, 14)), (Span(4, 7), Span(20, 23)),
    ]


def test_source_map_rejects_gaps_or_wrong_lengths():
    with pytest.raises(ValueError, match="contiguous"):
        SourceMap("x", 4, (SourceMapEntry(Span(1, 4)),))
    with pytest.raises(ValueError, match="equal span lengths"):
        SourceMapEntry(Span(0, 4), Span(0, 3))


def test_markdown_line_blocks_are_lossless_and_fences_hide_headings():
    source = "# Parent\nIntro\n## Child\n" + "~~~py\n# not heading\n~~~\n"
    doc = parse_markdown(source, "docs/manual.md")
    assert "".join(doc.text[block.span.start:block.span.end]
                   for block in doc.blocks) == source
    assert [block.title for block in doc.blocks if block.kind == "heading"] == [
        "Parent", "Child"
    ]
    assert doc.metadata["ast"] is True


def test_html_extracts_text_without_inventing_raw_html_offsets():
    doc = parse_html("<h1>Title</h1><p>Some <b>text</b></p>", "x.html")
    assert doc.text == "Title\n\nSome text"
    assert doc.source_map.resolve(Span(0, 5))[0].source_span is None
    assert doc.source_map.resolve(Span(0, 5))[0].mapping == "extracted"


def test_pdf_pages_are_one_based_and_split_spanning_chunks():
    doc = parse_pdf_pages(["第一页", "Second page"], "file.pdf")
    assert doc.text == "第一页\n\nSecond page"
    places = doc.source_map.resolve(Span(2, len(doc.text)))
    assert [loc.page_number for loc in places] == [1, 2]
    assert all(loc.source_span is None for loc in places)
    assert doc.metadata["page_count"] == 2


def test_empty_document_is_valid():
    doc = parse_text("", "empty.txt")
    assert doc.blocks == ()
    assert doc.source_map.resolve(Span(0, 0)) == ()


def test_document_rejects_wrong_map_uri():
    doc = parse_text("abc", "abc.txt")
    with pytest.raises(ValueError, match="URI mismatch"):
        replace(doc, source_map=SourceMap.identity("other.txt", 3))


def test_document_rejects_overlapping_blocks():
    doc = parse_text("abcdef", "x")
    with pytest.raises(ValueError, match="overlap"):
        replace(doc, blocks=(
            DocumentBlock("a", "text", Span(0, 4)),
            DocumentBlock("b", "text", Span(3, 6)),
        ))


def test_stable_id_depends_on_source_and_content():
    first = parse_text("same", "one.txt")
    assert first.document_id == parse_text("same", "one.txt").document_id
    assert first.document_id != parse_text("same", "two.txt").document_id
    assert first.document_id != parse_text("different", "one.txt").document_id


def test_parse_document_rejects_unknown_format():
    with pytest.raises(ValueError, match="Unsupported"):
        parse_document("text", "source.bin", "binary")
