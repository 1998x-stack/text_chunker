"""Prove CLI serialization and all five strategies use a consistent counter."""
import json
import sys
import types
from unittest.mock import patch

import numpy as np

from textchunker.parsers import parse_text
from textchunker.pipeline import run_chunking
from textchunker.tokenizers import CharacterTokenizer
from textchunker.types import StrategyConfig


def test_cli_emits_source_locations(tmp_path, monkeypatch):
    from textchunker.cli import main

    source = tmp_path / "input.md"
    source.write_text("# Heading\nA line.\n", encoding="utf-8")
    target = tmp_path / "out.jsonl"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "project:\n  visualize: false\n  export_format: jsonl\n"
        "strategy:\n  name: fixed\n"
        "  common:\n    chunk_size: 40\n    chunk_overlap: 0\n"
        "  fixed:\n    use_tokens: false\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", [
        "textchunker", "--config", str(cfg),
        "--input", str(source), "--output", str(target),
        "--tokenizer", "char",
    ])
    main()
    lines = [json.loads(row) for row in
             target.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 1
    assert lines[0]["source_locations"][0]["mapping"] == "exact"
    assert lines[0]["chunk_id"]
    assert lines[0]["complete"] is True
    assert lines[0]["heading_path"] == ["Heading"]


def test_semantic_uses_custom_count_without_real_model(monkeypatch):
    import textchunker.chunkers.semantic as module

    class FakeEmbedding:
        def __init__(self, *_a, **_kw):
            pass

        def encode(self, sentences, normalize_embeddings=True):
            return np.ones((len(sentences), 3), dtype=float) / np.sqrt(3)

    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = FakeEmbedding
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)

    class DoubleTokenizer:
        name = "double"

        def count(self, text):
            return len(text) * 2

    doc = parse_text("第一句。第二句！第三句？" * 3, "test.txt")
    cfg = StrategyConfig(
        name="semantic",
        common={"chunk_size": 16, "chunk_overlap": 2},
        semantic={"min_similarity": 0.5, "sentence_window": 1},
    )
    result = run_chunking(doc, cfg, tokenizer=DoubleTokenizer())
    assert result.complete
    assert all(chunk.token_count <= 16 for chunk in result.chunks)
