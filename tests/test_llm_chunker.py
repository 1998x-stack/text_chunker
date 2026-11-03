"""LLM provider output remains untrusted until validated."""

from unittest.mock import MagicMock, patch

from textchunker.types import StrategyConfig
from textchunker.chunkers.llm_based import LLMChunker
from textchunker.core.validation import validate_chunks


def _cfg(**overrides):
    options = {
        "provider": "dashscope",
        "llm_model": "qwen-max",
        "max_chars_per_call": 8000,
        "system_prompt": "Segment the text.",
    }
    options.update(overrides)
    return StrategyConfig(
        name="llm", llm=options,
        common={"chunk_size": 512, "chunk_overlap": 0},
    )


def test_llm_proposals_accepted_when_contiguous():
    text = "Hello World Testing Text Foo Bar"
    with patch("textchunker.chunkers.llm_based.DashScopeProvider") as provider:
        instance = MagicMock()
        instance.propose_spans.return_value = [
            (0, 10, "Part 1"),
            (10, len(text), "Part 2"),
        ]
        provider.return_value = instance
        chunks = LLMChunker(_cfg()).chunk(text)
        assert len(chunks) == 2
        assert chunks[0].meta["fallback"] is False
        validate_chunks(text, chunks)


def test_llm_empty_proposals_fall_back_losslessly():
    text = "Entire document as one chunk."
    with patch("textchunker.chunkers.llm_based.DashScopeProvider") as provider:
        instance = MagicMock()
        instance.propose_spans.return_value = []
        provider.return_value = instance
        chunks = LLMChunker(_cfg()).chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].meta["fallback"] is True


def test_llm_max_chunks_truncates_explicitly():
    cfg = _cfg()
    cfg.common["max_chunks"] = 1
    text = "Hello World Testing"
    with patch("textchunker.chunkers.llm_based.DashScopeProvider") as provider:
        instance = MagicMock()
        instance.propose_spans.return_value = [
            (0, 5, "A"), (5, 10, "B"), (10, len(text), "C")
        ]
        provider.return_value = instance
        chunks = LLMChunker(cfg).chunk(text)
        assert len(chunks) == 1
