import sys
import types

import numpy as np
import pytest

# 提前注入 sentence_transformers stub，避免真实依赖
stub_module = types.ModuleType("sentence_transformers")


class _DummySentenceTransformer:
    def __init__(self, *_args, **_kwargs):
        pass

    def encode(self, sentences, normalize_embeddings=True):
        vecs = np.eye(len(sentences), dtype=float)
        return vecs


stub_module.SentenceTransformer = _DummySentenceTransformer
sys.modules["sentence_transformers"] = stub_module

from textchunker.chunkers.semantic import SemanticChunker
from textchunker import utils as _utils
from textchunker.types import StrategyConfig

_utils._enc = None


def _strategy_cfg(**semantic_overrides):
    return StrategyConfig(
        name="semantic",
        common={"chunk_size": 15, "chunk_overlap": 0},
        semantic={"min_similarity": 0.0, "sentence_window": 1, **semantic_overrides},
        recursive={},
        fixed={},
        structure={},
        llm={},
    )


def test_semantic_chunker_preserves_sentence_offsets():
    text = "First sentence.\n\nSecond sentence with more words.\n\nThird sentence."
    chunker = SemanticChunker(_strategy_cfg())

    chunks = chunker.chunk(text)

    assert len(chunks) == 3
    second_start = text.index("Second sentence")
    third_start = text.index("Third sentence")

    assert chunks[1].start == second_start
    assert chunks[2].start == third_start
    assert text[chunks[0].start:chunks[0].end].strip() == chunks[0].text


def test_semantic_chunker_applies_max_chunks_limit():
    text = "One. Two. Three. Four."
    cfg = _strategy_cfg()
    cfg.common["chunk_size"] = 5
    cfg.common["max_chunks"] = 2

    chunker = SemanticChunker(cfg)
    chunks = chunker.chunk(text)

    assert len(chunks) == 2
