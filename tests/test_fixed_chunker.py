import pytest
from textchunker.types import StrategyConfig
from textchunker.chunkers.fixed import FixedChunker

def _cfg(**overrides):
    common = {"chunk_size": 50, "chunk_overlap": 10}
    fixed = {"use_tokens": False}
    common.update(overrides.get("common", {}))
    fixed.update(overrides.get("fixed", {}))
    return StrategyConfig(
        name="fixed", common=common, fixed=fixed,
        semantic={}, recursive={}, structure={}, llm={},
    )

def test_fixed_chunker_basic():
    text = "a" * 120
    chunker = FixedChunker(_cfg())
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 50
        assert c.start >= 0
        assert c.end <= len(text)
        assert text[c.start:c.end] == c.text

def test_fixed_chunker_overlap():
    text = "a" * 100
    chunker = FixedChunker(_cfg(common={"chunk_size": 50, "chunk_overlap": 20}))
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    assert chunks[1].start == chunks[0].end - 20

def test_fixed_chunker_empty_text():
    chunker = FixedChunker(_cfg())
    chunks = chunker.chunk("")
    assert chunks == []

def test_fixed_chunker_text_smaller_than_size():
    text = "short"
    chunker = FixedChunker(_cfg(common={"chunk_size": 100, "chunk_overlap": 10}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0].text == "short"

def test_fixed_chunker_max_chunks():
    text = "a" * 500
    chunker = FixedChunker(_cfg(common={"chunk_size": 50, "chunk_overlap": 0, "max_chunks": 3}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 3

def test_fixed_chunker_token_mode():
    from textchunker.utils import count_tokens, _enc
    if _enc is None:
        pytest.skip("tiktoken not available")
    text = "Hello world. " * 100
    chunker = FixedChunker(_cfg(
        common={"chunk_size": 50, "chunk_overlap": 10},
        fixed={"use_tokens": True},
    ))
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert count_tokens(c.text) <= 55  # small tolerance
        assert c.start >= 0
        assert c.end <= len(text)

def test_fixed_chunker_token_mode_fallback():
    import textchunker.utils as utils_mod
    original_enc = utils_mod._enc
    utils_mod._enc = None
    try:
        text = "a" * 120
        chunker = FixedChunker(_cfg(
            common={"chunk_size": 50, "chunk_overlap": 10},
            fixed={"use_tokens": True},
        ))
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
    finally:
        utils_mod._enc = original_enc
