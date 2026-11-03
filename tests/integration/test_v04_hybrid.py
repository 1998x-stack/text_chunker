"""Hybrid integrity tests: atomic structures, headings, budgets, provenance."""
import pytest

from textchunker.core.spans import Span
from textchunker.evaluation.structure import evaluate_structure
from textchunker.exceptions import ChunkerError, ConfigError
from textchunker.parsers import parse_html, parse_markdown, parse_pdf_pages
from textchunker.pipeline import run_chunking
from textchunker.tokenizers import CharacterTokenizer
from textchunker.types import Chunk, StrategyConfig


class DoubleTokenizer:
    name = "test:double"

    def count(self, text):
        return len(text) * 2


def _cfg(size=64, overlap=0, limit=None, policy="split"):
    return StrategyConfig(
        name="hybrid",
        common={
            "chunk_size": size, "chunk_overlap": overlap, "max_chunks": limit,
        },
        hybrid={"oversized_element": policy},
    )


def _assert_atomic_intact(document, result, size):
    for block in document.blocks:
        if block.kind not in {"code", "table"}:
            continue
        if len(block.span.extract(document.text)) > size:
            continue
        assert any(
            chunk.span.start <= block.span.start
            and chunk.span.end >= block.span.end
            for chunk in result.chunks
        ), block.block_id


def test_hybrid_keeps_fitting_code_and_table_atomic():
    text = (
        "# One\n\nIntro.\n\n"
        "~~~py\nx = 1\n~~~\n\n"
        "| A | B |\n| - | - |\n| 1 | 2 |\n\n"
        "# Two\nNew section.\n"
    )
    doc = parse_markdown(text, "manual.md")
    result = run_chunking(doc, _cfg(size=45, overlap=6),
                          CharacterTokenizer())
    assert result.report.ok and result.complete
    assert result.report.unique_coverage_ratio == 1.0
    _assert_atomic_intact(doc, result, 45)
    assert not any(issue.code in {"ATOMIC_SPLIT", "HEADING_CROSSING"}
                   for issue in result.report.diagnostics)
    assert all(chunk.token_count <= 45 for chunk in result.chunks)
    assert "Two" in {part for chunk in result.chunks
                     for part in chunk.heading_path}


def test_hybrid_source_coverage_and_budget_with_custom_tokenizer():
    doc = parse_markdown("# A\n中文🚀abc\n\n# B\nSecond line.\n", "a.md")
    result = run_chunking(doc, _cfg(size=24, overlap=4),
                          DoubleTokenizer())
    assert result.complete and result.report.ok
    assert result.tokenizer_id == "test:double"
    assert all(chunk.token_count <= 24 for chunk in result.chunks)
    assert all(chunk.text == doc.text[chunk.span.start:chunk.span.end]
               for chunk in result.chunks)


def test_hybrid_oversized_code_reports_warning_and_recovers():
    doc = parse_markdown(
        "# Guide\n\n~~~python\n" + ("verylongline()" * 15) +
        "\nnext line\n~~~\n", "big.md"
    )
    result = run_chunking(doc, _cfg(size=22, overlap=3),
                          CharacterTokenizer())
    assert result.report.ok and result.complete
    assert all(chunk.token_count <= 22 for chunk in result.chunks)
    assert any(issue.code == "ATOMIC_OVERFLOW"
               for issue in result.report.diagnostics)
    assert any(chunk.metadata["atomic_overflow"] for chunk in result.chunks)


def test_hybrid_oversized_table_can_fail_closed():
    doc = parse_markdown("| A | B |\n| - | - |\n| " + "x" * 90 +
                         " | value |\n", "table.md")
    with pytest.raises(ChunkerError, match="Atomic table"):
        run_chunking(doc, _cfg(size=30, policy="error"),
                     CharacterTokenizer())


def test_hybrid_preserves_html_dom_block_ids():
    doc = parse_html(
        "<h1>Intro</h1><p>Welcome.</p><pre>example code</pre>"
        "<table><tr><td>A</td><td>B</td></tr></table>",
        "page.html",
    )
    result = run_chunking(doc, _cfg(size=48, overlap=6),
                          CharacterTokenizer())
    assert result.report.ok and result.complete
    _assert_atomic_intact(doc, result, 48)
    assert all(loc.source_span is None
               for chunk in result.chunks for loc in chunk.source_locations)
    assert any(loc.block_id and loc.block_id.startswith("dom:")
               for chunk in result.chunks for loc in chunk.source_locations)


def test_hybrid_keeps_pdf_page_locations():
    doc = parse_pdf_pages(["First page " * 8, "第二页 " * 10], "paper.pdf")
    result = run_chunking(doc, _cfg(size=32, overlap=0),
                          CharacterTokenizer())
    assert result.complete and result.report.ok
    assert {loc.page_number for chunk in result.chunks
            for loc in chunk.source_locations} == {1, 2}


def test_hybrid_truncated_run_is_explicitly_incomplete():
    doc = parse_markdown("# A\n" + ("one line\n" * 30), "partial.md")
    result = run_chunking(doc, _cfg(size=24, limit=2),
                          CharacterTokenizer())
    assert len(result.chunks) == 2
    assert result.report.ok
    assert result.complete is False
    assert result.report.missing_spans


def test_hybrid_rejects_invalid_policy():
    doc = parse_markdown("# A\nText.", "a.md")
    with pytest.raises(ConfigError, match="oversized_element"):
        run_chunking(doc, _cfg(policy="ignore"), CharacterTokenizer())


def test_structure_evaluator_rejects_splitting_fitting_code():
    source = "~~~\nabc\n~~~\n"
    doc = parse_markdown(source, "code.md")
    pieces = [
        Chunk(0, source[:5], 0, 5),
        Chunk(1, source[5:], 5, len(source)),
    ]
    issues = evaluate_structure(doc, pieces, len, 99)
    assert "ATOMIC_SPLIT" in {item.code for item in issues}
