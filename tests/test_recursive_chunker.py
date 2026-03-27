import pytest
from textchunker.types import StrategyConfig
from textchunker.chunkers.recursive import RecursiveChunker

def _cfg(**overrides):
    common = {"chunk_size": 50, "chunk_overlap": 0}
    recursive = {"separators": ["\n\n", "\n", " ", ""]}
    common.update(overrides.get("common", {}))
    recursive.update(overrides.get("recursive", {}))
    return StrategyConfig(
        name="recursive", common=common, recursive=recursive,
        semantic={}, fixed={}, structure={}, llm={},
    )

def test_recursive_chunker_basic():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 100, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1

def test_recursive_chunker_offset_correctness():
    text = "AAA.\n\nBBB.\n\nCCC."
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 10, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    for c in chunks:
        assert c.start >= 0
        assert c.end <= len(text)
        assert text[c.start:c.end] == c.text

def test_recursive_chunker_with_overlap():
    text = "Word " * 50
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 50, "chunk_overlap": 10}))
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2

def test_recursive_chunker_duplicate_content():
    text = "Same line.\n\nSame line.\n\nSame line."
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 20, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    starts = [c.start for c in chunks]
    assert len(starts) == len(set(starts))

def test_recursive_chunker_max_chunks():
    text = "Word " * 100
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 20, "chunk_overlap": 0, "max_chunks": 3}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 3

def test_recursive_chunker_empty_text():
    chunker = RecursiveChunker(_cfg())
    chunks = chunker.chunk("")
    assert len(chunks) <= 1
