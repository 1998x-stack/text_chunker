"""All deterministic strategies accept DocumentIR and explicit tokenizers."""

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from textchunker.core.models import SourceMap
from textchunker.exceptions import ChunkerError
from textchunker.parsers import parse_markdown, parse_pdf_pages, parse_text
from textchunker.pipeline import run_chunking
from textchunker.tokenizers import (
    CharacterTokenizer, LegacyTokenizer, TiktokenTokenizer, create_tokenizer,
)
from textchunker.types import StrategyConfig


@dataclass
class DoubleTokenizer:
    name: str = "test:double"

    def count(self, text: str) -> int:
        return len(text) * 2


def _cfg(name, size=10, overlap=0, limit=None, **options):
    return StrategyConfig(
        name=name, common={
            "chunk_size": size, "chunk_overlap": overlap, "max_chunks": limit,
        },
        fixed=options if name == "fixed" else {},
        structure=options if name == "structure" else {},
        recursive={"separators": ["\n\n", "\n", " ", ""]},
        llm=options if name == "llm" else {},
    )


@pytest.mark.parametrize("name,opts", [
    ("fixed", {"use_tokens": True}),
    ("recursive", {}),
    ("structure", {"prefer": "md"}),
])
def test_tokenizer_is_injected_into_deterministic_chunkers(name, opts):
    doc = parse_markdown("# Intro\nHello world.\n\n# More\nSecond part.", "a.md")
    result = run_chunking(doc, _cfg(name, size=20, overlap=4, **opts),
                          tokenizer=DoubleTokenizer())
    assert result.report.ok and result.complete
    assert all(chunk.token_count <= 20 for chunk in result.chunks)
    assert all(chunk.source_locations for chunk in result.chunks)
    assert result.report.unique_coverage_ratio == 1.0


def test_char_mode_is_explicit_and_ignores_other_tokenizer():
    doc = parse_text("abcdefghijk", "a.txt")
    result = run_chunking(doc, _cfg(
        "fixed", size=4, overlap=1, use_tokens=False
    ), tokenizer=DoubleTokenizer())
    assert result.tokenizer_id == "character"
    assert max(chunk.token_count for chunk in result.chunks) <= 4


def test_chunk_ids_are_repeatable_and_source_sensitive():
    doc = parse_text("content\n" * 20, "one.txt")
    config = _cfg("recursive", size=18, overlap=2)
    first = run_chunking(doc, config, CharacterTokenizer())
    second = run_chunking(doc, config, CharacterTokenizer())
    assert [chunk.chunk_id for chunk in first.chunks] == [
        chunk.chunk_id for chunk in second.chunks
    ]
    other = run_chunking(parse_text(doc.text, "other.txt"), config,
                         CharacterTokenizer())
    assert [chunk.chunk_id for chunk in first.chunks] != [
        chunk.chunk_id for chunk in other.chunks
    ]


def test_markdown_heading_path_tracks_nested_headings():
    doc = parse_markdown("# Parent\nA\n## Child\nB\n# Next\nC", "a.md")
    result = run_chunking(doc, _cfg("fixed", size=12, use_tokens=False),
                          CharacterTokenizer())
    assert any(chunk.heading_path == ("Parent",) for chunk in result.chunks)
    assert any(chunk.heading_path == ("Parent", "Child")
               for chunk in result.chunks)


def test_truncated_output_is_explicitly_incomplete():
    doc = parse_text("abcdef" * 8, "many.txt")
    result = run_chunking(doc, _cfg(
        "fixed", size=8, limit=1, use_tokens=False
    ))
    assert result.report.ok
    assert not result.complete
    assert result.report.missing_spans
    assert result.report.diagnostics[0].code == "INCOMPLETE"


def test_pdf_provenance_survives_chunking():
    doc = parse_pdf_pages(["a" * 6, "b" * 6], "paper.pdf")
    result = run_chunking(doc, _cfg(
        "fixed", size=10, use_tokens=False
    ))
    assert result.complete
    page_numbers = {
        location.page_number
        for chunk in result.chunks
        for location in chunk.source_locations
    }
    assert page_numbers == {1, 2}
    assert all(location.source_span is None
               for chunk in result.chunks
               for location in chunk.source_locations)


def test_bad_strategy_result_fails_closed(monkeypatch):
    from textchunker import pipeline
    from textchunker.pipeline import runner

    class BrokenChunker:
        def chunk(self, text):
            from textchunker.types import Chunk
            return [Chunk(0, "bad", 0, 3)]

    monkeypatch.setattr(runner, "create_chunker",
                        lambda cfg, tokenizer: BrokenChunker())
    with pytest.raises(ChunkerError, match="SOURCE_MISMATCH"):
        pipeline.run_chunking(parse_text("abcdef", "a"), _cfg(
            "recursive", size=12
        ), CharacterTokenizer())


def test_model_tokenizers_are_explicit_or_fail_at_construction():
    assert create_tokenizer("char").count("A🚀") == 2
    assert isinstance(create_tokenizer("legacy"), LegacyTokenizer)
    with pytest.raises(ValueError):
        create_tokenizer("unknown:tokenizer")
    tiktoken = pytest.importorskip("tiktoken")
    adapter = TiktokenTokenizer()
    assert adapter.count("hello world") == len(
        tiktoken.get_encoding("cl100k_base").encode("hello world")
    )


def test_llm_respects_injected_counter():
    from textchunker.chunkers import llm_based

    class NoSplitProvider:
        def __init__(self, **kwargs):
            pass

        def propose_spans(self, text, system_prompt, max_chars):
            return [(0, len(text), None)]

    with patch.object(llm_based, "DashScopeProvider", NoSplitProvider):
        doc = parse_text("abcdefghij" * 3, "a")
        cfg = _cfg("llm", size=8, overlap=0, max_chars_per_call=10)
        result = run_chunking(doc, cfg, DoubleTokenizer())
        assert result.complete
        assert all(chunk.token_count <= 8 for chunk in result.chunks)
