"""Provider parsing, finite budgets, retry limits, semantic injection."""
from __future__ import annotations

import numpy as np
import pytest

from textchunker.exceptions import ProviderError
from textchunker.parsers import parse_text
from textchunker.pipeline import run_chunking
from textchunker.providers.response_parser import parse_span_response
from textchunker.tokenizers import CharacterTokenizer
from textchunker.types import StrategyConfig


@pytest.mark.parametrize("bad", [
    '[{"start":true,"end":4}]',
    '[{"start":0.5,"end":4}]',
    '[{"start":"0","end":4}]',
    '[{"start":0,"end":4,"title":5}]',
    '[{"start":4,"end":1}]',
    '[{"start":0,"end":0}]',
    '{"a": 3}',
    "not json",
])
def test_untrusted_llm_offsets_rejected_without_coercion(bad):
    assert parse_span_response(bad) == []


def test_bounded_parser_accepts_fenced_json_and_rejects_huge_response():
    fence = chr(96) * 3
    content = ('Result:\n' + fence + 'json\n'
               + '[{"start":0,"end":10,"title":"A"}]\n' + fence)
    assert parse_span_response(content) == [(0, 10, "A")]
    assert parse_span_response(content, max_response_chars=10) == []


def test_llm_provider_call_budget_keeps_source_complete():
    class Provider:
        def __init__(self):
            self.calls = 0

        def propose_spans(self, text, system_prompt, max_chars):
            self.calls += 1
            return [(0, len(text), None)]

    fake = Provider()
    cfg = StrategyConfig(
        "llm", common={"chunk_size": 8, "chunk_overlap": 0},
        llm={
            "max_chars_per_call": 8, "max_provider_calls": 1,
            "max_total_prompt_chars": 8, "retry_attempts": 1,
        },
    )
    doc = parse_text("0123456789abcdefghijklmnopqrstuvwxyz", "x")
    result = run_chunking(doc, cfg, CharacterTokenizer(), llm_provider=fake)
    assert result.report.ok and result.complete
    assert fake.calls == 1
    assert len(result.chunks) > 1
    assert all(chunk.token_count <= 8 for chunk in result.chunks)
    assert any(chunk.metadata["fallback_reason"] == "budget_exhausted"
               for chunk in result.chunks)


def test_llm_retry_is_bounded_and_can_recover():
    class Provider:
        calls = 0

        def propose_spans(self, text, system_prompt, max_chars):
            self.calls += 1
            return [] if self.calls == 1 else [(0, len(text), "ok")]

    fake = Provider()
    cfg = StrategyConfig(
        "llm", common={"chunk_size": 100, "chunk_overlap": 0},
        llm={"max_chars_per_call": 100, "max_provider_calls": 2,
             "retry_attempts": 2},
    )
    result = run_chunking(
        parse_text("One block.", "x"), cfg,
        CharacterTokenizer(), llm_provider=fake,
    )
    assert fake.calls == 2
    assert result.report.ok and result.complete
    assert result.chunks[0].metadata["fallback"] is False


def test_llm_strict_error_policy_refuses_fake_success():
    class NeverCalled:
        def propose_spans(self, *args, **kwargs):
            raise AssertionError("must not invoke")

    cfg = StrategyConfig(
        "llm", common={"chunk_size": 32, "chunk_overlap": 0},
        llm={"max_provider_calls": 0, "on_failure": "error"},
    )
    with pytest.raises(ProviderError, match="budget_exhausted"):
        run_chunking(parse_text("important", "x"), cfg,
                     CharacterTokenizer(), llm_provider=NeverCalled())


class FakeEmbeddings:
    def __init__(self, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def encode(self, sentences, normalize_embeddings=True):
        self.calls += 1
        if self.invalid:
            return np.full((len(sentences), 2), np.nan)
        return np.asarray([
            [2.0, 0.0] if "alpha" in s else [0.0, 7.0]
            for s in sentences
        ])


def _semantic_cfg(**options):
    return StrategyConfig(
        "semantic", common={"chunk_size": 70, "chunk_overlap": 0},
        semantic={"min_similarity": 0.5, "sentence_window": 1, **options},
    )


def test_semantic_provider_injection_normalizes_and_splits_topics():
    fake = FakeEmbeddings()
    doc = parse_text("alpha. alpha. beta. beta.", "topics")
    result = run_chunking(
        doc, _semantic_cfg(), CharacterTokenizer(), embedding_provider=fake,
    )
    assert result.report.ok and result.complete
    assert len(result.chunks) == 2
    assert fake.calls == 1


def test_semantic_nonfinite_embedding_fails_closed():
    with pytest.raises(ProviderError, match="invalid shape/values"):
        run_chunking(
            parse_text("alpha. beta.", "topics"), _semantic_cfg(),
            CharacterTokenizer(),
            embedding_provider=FakeEmbeddings(invalid=True),
        )


def test_semantic_unavailable_provider_has_explicit_recursive_fallback():
    class Broken:
        def encode(self, sentences, normalize_embeddings=True):
            raise RuntimeError("offline")

    result = run_chunking(
        parse_text("alpha. beta. gamma.", "topics"),
        _semantic_cfg(on_embedding_error="recursive"),
        CharacterTokenizer(), embedding_provider=Broken(),
    )
    assert result.report.ok and result.complete
    assert all(item.metadata["fallback"] == "recursive"
               for item in result.chunks)


def test_semantic_embedding_call_cap_is_enforced():
    with pytest.raises(ProviderError, match="budget exceeded"):
        run_chunking(
            parse_text("alpha. beta. alpha. gamma.", "topics"),
            _semantic_cfg(embedding_batch_size=1, max_embedding_calls=1),
            CharacterTokenizer(), embedding_provider=FakeEmbeddings(),
        )
