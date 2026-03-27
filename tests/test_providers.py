import json
import pytest
from unittest.mock import MagicMock, patch

from textchunker.providers.dashscope_provider import DashScopeProvider
from textchunker.providers.llm import HFProvider, BaseLLMProvider


def _make_dashscope_provider(mock_client, model="qwen-max"):
    """Create a DashScopeProvider without calling __init__ (avoids OpenAI import)."""
    provider = DashScopeProvider.__new__(DashScopeProvider)
    provider.model = model
    provider.client = mock_client
    return provider


def test_dashscope_provider_parses_response():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps([
        {"title": "Intro", "start": 0, "end": 50},
        {"title": "Body", "start": 50, "end": 200},
    ])

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    provider = _make_dashscope_provider(mock_client)
    spans = provider.propose_spans("test text", "system prompt", 8000)
    assert len(spans) == 2
    assert spans[0] == (0, 50, "Intro")
    assert spans[1] == (50, 200, "Body")


def test_dashscope_provider_handles_invalid_json():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "not valid json at all"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    provider = _make_dashscope_provider(mock_client)
    spans = provider.propose_spans("text", "prompt", 1000)
    assert spans == []


def test_dashscope_provider_handles_api_error():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    provider = _make_dashscope_provider(mock_client)
    spans = provider.propose_spans("text", "prompt", 1000)
    assert spans == []


def test_dashscope_parse_spans_extracts_embedded_json():
    content = 'Here is the result: [{"title": "A", "start": 0, "end": 10}] done.'
    spans = DashScopeProvider._parse_spans(content)
    assert len(spans) == 1
    assert spans[0] == (0, 10, "A")


def test_base_provider_is_abstract():
    provider = BaseLLMProvider()
    with pytest.raises(NotImplementedError):
        provider.propose_spans("text", "prompt", 100)
