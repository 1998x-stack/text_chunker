import pytest
from unittest.mock import MagicMock, patch

from textchunker.types import StrategyConfig
from textchunker.chunkers.llm_based import LLMChunker


def _cfg(**overrides):
    llm = {
        "provider": "dashscope",
        "llm_model": "qwen-max",
        "max_chars_per_call": 8000,
        "system_prompt": "Segment the text.",
    }
    llm.update(overrides)
    return StrategyConfig(
        name="llm", llm=llm, common={},
        semantic={}, recursive={}, fixed={}, structure={},
    )


def test_llm_chunker_with_mock_provider():
    cfg = _cfg()
    with patch("textchunker.chunkers.llm_based.DashScopeProvider") as MockProvider:
        mock_instance = MagicMock()
        mock_instance.propose_spans.return_value = [
            (0, 10, "Part 1"),
            (10, 20, "Part 2"),
        ]
        MockProvider.return_value = mock_instance

        chunker = LLMChunker(cfg)
        chunks = chunker.chunk("Hello World Testing Text Foo Bar")
        assert len(chunks) == 2
        assert chunks[0].meta["strategy"] == "llm"


def test_llm_chunker_fallback_on_empty_spans():
    cfg = _cfg()
    with patch("textchunker.chunkers.llm_based.DashScopeProvider") as MockProvider:
        mock_instance = MagicMock()
        mock_instance.propose_spans.return_value = []
        MockProvider.return_value = mock_instance

        chunker = LLMChunker(cfg)
        text = "Entire document as one chunk."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text


def test_llm_chunker_max_chunks():
    cfg = _cfg()
    cfg.common["max_chunks"] = 1
    with patch("textchunker.chunkers.llm_based.DashScopeProvider") as MockProvider:
        mock_instance = MagicMock()
        mock_instance.propose_spans.return_value = [
            (0, 5, "A"), (5, 10, "B"), (10, 15, "C"),
        ]
        MockProvider.return_value = mock_instance

        chunker = LLMChunker(cfg)
        chunks = chunker.chunk("Hello World Testing")
        assert len(chunks) == 1
