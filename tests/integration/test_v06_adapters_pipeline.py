"""Adapters and run-scoped tools keep canonical source text untouched."""
from __future__ import annotations

import json
import sys
import types

from textchunker.adapters import to_langchain_documents, to_llamaindex_nodes
from textchunker.export import save_result_jsonl
from textchunker.parsers import parse_markdown, parse_text
from textchunker.pipeline import run_chunking
from textchunker.pipeline.context_builder import contextualize_chunk
from textchunker.pipeline.profiler import profile_chunking
from textchunker.pipeline.router import recommend_strategy
from textchunker.tokenizers import CharacterTokenizer
from textchunker.types import StrategyConfig


def _result():
    doc = parse_markdown("# Overview\n\nSome content.", "example.md")
    cfg = StrategyConfig(
        "recursive", common={"chunk_size": 100, "chunk_overlap": 0}
    )
    return run_chunking(doc, cfg, CharacterTokenizer())


def test_contextual_text_does_not_mutate_spans():
    result = _result()
    chunk = result.chunks[0]
    source_text = chunk.text
    contextual = contextualize_chunk(chunk)
    assert "Overview" in contextual
    assert contextual.endswith(source_text)
    assert chunk.text == source_text


def test_deterministic_router_does_not_load_model():
    assert recommend_strategy(parse_text("plain", "a.txt")) == "recursive"
    assert recommend_strategy(parse_markdown("# T\nBody", "a.md")) == "hybrid"


def test_run_profile_reports_measured_span_integrity():
    doc = parse_text("abcdef", "a.txt")
    cfg = StrategyConfig(
        "fixed", common={"chunk_size": 4, "chunk_overlap": 0},
        fixed={"use_tokens": False},
    )
    result, profile = profile_chunking(doc, cfg, CharacterTokenizer())
    assert result.report.ok
    assert profile.integrity_ok
    assert profile.canonical_characters == 6
    assert profile.chunks == len(result.chunks)
    assert profile.elapsed_seconds >= 0


def test_langchain_adapter_is_lazy_and_uses_real_chunk_ids(monkeypatch):
    class FakeDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata

    root = types.ModuleType("langchain_core")
    module = types.ModuleType("langchain_core.documents")
    module.Document = FakeDocument
    monkeypatch.setitem(sys.modules, "langchain_core", root)
    monkeypatch.setitem(sys.modules, "langchain_core.documents", module)
    result = _result()
    documents = to_langchain_documents(result)
    assert documents[0].page_content == result.chunks[0].text
    assert documents[0].metadata["chunk_id"] == result.chunks[0].chunk_id


def test_llamaindex_adapter_sets_id_and_character_offsets(monkeypatch):
    class FakeTextNode:
        def __init__(self, **kwargs):
            self.payload = kwargs

    root = types.ModuleType("llama_index")
    core = types.ModuleType("llama_index.core")
    schema = types.ModuleType("llama_index.core.schema")
    schema.TextNode = FakeTextNode
    for name, module in [
        ("llama_index", root), ("llama_index.core", core),
        ("llama_index.core.schema", schema),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    result = _result()
    node = to_llamaindex_nodes(result)[0]
    assert node.payload["id_"] == result.chunks[0].chunk_id
    assert node.payload["start_char_idx"] == result.chunks[0].span.start
    assert node.payload["end_char_idx"] == result.chunks[0].span.end


def test_export_includes_source_bbox_field_as_optional(tmp_path):
    result = _result()
    path = tmp_path / "chunks.jsonl"
    save_result_jsonl(str(path), result, "example.md")
    record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert record["chunk_id"] == result.chunks[0].chunk_id
    assert record["source_locations"][0]["bbox"] is None
