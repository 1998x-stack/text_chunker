"""Test semantic decision logic with deterministic, isolated embeddings."""

import sys
import types

import numpy as np
import pytest

from textchunker.core.validation import validate_chunks
from textchunker.types import StrategyConfig
from textchunker import utils


class DummySentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, sentences, normalize_embeddings=True):
        return np.eye(len(sentences), dtype=float)


@pytest.fixture
def chunker_factory(monkeypatch):
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = DummySentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", stub)
    monkeypatch.setattr(utils, "_enc", None)
    from textchunker.chunkers.semantic import SemanticChunker

    def create(size=15, limit=None):
        cfg = StrategyConfig(
            name="semantic",
            common={"chunk_size": size, "chunk_overlap": 0,
                    "max_chunks": limit},
            semantic={"min_similarity": 0, "sentence_window": 1},
        )
        return SemanticChunker(cfg)
    return create


def test_semantic_preserves_source_and_budget(chunker_factory):
    text = "First sentence.\n\nSecond sentence with more words.\n\nThird sentence."
    chunks = chunker_factory().chunk(text)
    validate_chunks(text, chunks, count=len, max_size=15)


def test_semantic_limits_chunk_count(chunker_factory):
    chunks = chunker_factory(size=5, limit=2).chunk(
        "One. Two. Three. Four."
    )
    assert len(chunks) == 2


def test_semantic_handles_duplicate_sentences(chunker_factory):
    text = "Hello world. Hello world. Goodbye world."
    chunks = chunker_factory().chunk(text)
    validate_chunks(text, chunks, count=len, max_size=15)
    assert [chunk.start for chunk in chunks] == sorted(
        chunk.start for chunk in chunks
    )


def test_semantic_chinese_without_spaces(chunker_factory):
    text = "第一句话。第二句话！第三句话？"
    chunks = chunker_factory(size=12).chunk(text)
    validate_chunks(text, chunks, count=len, max_size=12)
