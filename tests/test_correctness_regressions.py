"""Previously uncovered data-loss and token-budget regressions."""

import json
import sys
import types

import pytest

from textchunker.core.validation import validate_chunks
from textchunker.types import StrategyConfig


def config(name, size=20, overlap=0, max_chunks=None, **options):
    return StrategyConfig(
        name=name,
        common={
            "chunk_size": size,
            "chunk_overlap": overlap,
            "max_chunks": max_chunks,
        },
        fixed=options if name == "fixed" else {},
        recursive={"separators": ["\n\n", "\n", " ", ""]},
        structure=options if name == "structure" else {},
        semantic=options if name == "semantic" else {},
        llm=options if name == "llm" else {},
    )


def test_recursive_preserves_double_newline(monkeypatch):
    import textchunker.utils as utils
    from textchunker.chunkers.recursive import RecursiveChunker

    monkeypatch.setattr(utils, "_enc", None)
    # The legacy implementation silently dropped both newline characters.
    source = "abcdef\n\nXYZ"
    chunks = RecursiveChunker(config("recursive", size=3)).chunk(source)
    assert "".join(chunk.text for chunk in chunks) == source
    validate_chunks(source, chunks, count=len, max_size=3)


@pytest.mark.parametrize("source", [
    "前言\n\n# 第一章\n介绍\n\n# 第二章\n内容",
    "  \n\n",
    "Same paragraph.\n\nSame paragraph.\n\nAnother.",
    "中文没有空格。第二句话！第三句话？",
    "123 \n\n 🚀 café 中 文 \n code",
])
def test_recursive_full_coverage_and_budget(monkeypatch, source):
    import textchunker.utils as utils
    from textchunker.chunkers.recursive import RecursiveChunker

    monkeypatch.setattr(utils, "_enc", None)
    chunks = RecursiveChunker(config("recursive", size=18, overlap=4)).chunk(source)
    validate_chunks(source, chunks, count=len, max_size=18)


def test_structure_keeps_preamble_and_sections(monkeypatch):
    import textchunker.utils as utils
    from textchunker.chunkers.structure import StructureChunker

    monkeypatch.setattr(utils, "_enc", None)
    source = "A preamble\n\n# Heading\nFirst line.\n\n# Next\nSecond line."
    chunks = StructureChunker(
        config("structure", size=18, overlap=3, prefer="md")
    ).chunk(source)
    validate_chunks(source, chunks, count=len, max_size=18)
    assert chunks[0].meta["title"] == "Preamble"


def test_token_mode_never_decodes_broken_utf8(monkeypatch):
    tiktoken = pytest.importorskip("tiktoken")
    import textchunker.utils as utils
    from textchunker.chunkers.fixed import FixedChunker

    encoder = tiktoken.get_encoding("cl100k_base")
    monkeypatch.setattr(utils, "_enc", encoder)
    source = "🚀你好🙂café\n" * 20
    chunks = FixedChunker(
        config("fixed", size=12, overlap=3, use_tokens=True)
    ).chunk(source)
    validate_chunks(source, chunks, count=lambda t: len(encoder.encode(t)),
                    max_size=12)
    assert all("\ufffd" not in chunk.text for chunk in chunks)


def test_llm_invalid_spans_fallback_without_losing_source(monkeypatch):
    from textchunker.chunkers import llm_based

    class BadProvider:
        def __init__(self, **kwargs):
            pass

        def propose_spans(self, text, system_prompt, max_chars):
            return [(0, 2, "gap"), (5, 7, "skip")]

    monkeypatch.setattr(llm_based, "DashScopeProvider", BadProvider)
    source = "first sentence\n\nsecond sentence\n\nthird sentence"
    chunks = llm_based.LLMChunker(
        config("llm", size=14, overlap=2, max_chars_per_call=11)
    ).chunk(source)
    validate_chunks(source, chunks, count=None, max_size=None)
    assert any(chunk.meta["fallback"] for chunk in chunks)


def test_cli_multi_document_jsonl_does_not_overwrite(tmp_path, monkeypatch):
    from textchunker.cli import main

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "a.txt").write_text("Document one.", encoding="utf-8")
    (inputs / "b.txt").write_text("Document two.", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "project:\n  visualize: false\n  export_format: jsonl\n"
        "strategy:\n  name: fixed\n  common:\n"
        "    chunk_size: 100\n    chunk_overlap: 0\n"
        "  fixed:\n    use_tokens: false\n",
        encoding="utf-8",
    )
    output = tmp_path / "chunks.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "textchunker", "--config", str(config_file),
        "--input", str(inputs), "--output", str(output),
    ])
    main()
    lines = [json.loads(line) for line in
             output.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert {line["text"] for line in lines} == {
        "Document one.", "Document two."
    }


def test_sentence_offsets_cover_unspaced_chinese():
    from textchunker.utils import whitespace_sentences_with_offsets

    source = "第一句。第二句！第三句？"
    spans = whitespace_sentences_with_offsets(source)
    assert len(spans) == 3
    assert "".join(part for part, _, _ in spans) == source
