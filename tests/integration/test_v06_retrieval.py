"""Evidence-grounded offline retrieval evaluation and dataset validation."""
import json
from pathlib import Path

import pytest

from textchunker.core.spans import Span
from textchunker.evaluation.benchmark import (
    load_golden_dataset, run_benchmark,
)
from textchunker.evaluation.report import write_report
from textchunker.evaluation.retrieval import (
    EvidenceAnchor, GoldenQuestion, evaluate_retrieval,
)
from textchunker.parsers import parse_text
from textchunker.pipeline import run_chunking
from textchunker.tokenizers import CharacterTokenizer
from textchunker.types import StrategyConfig


def _fixed_result():
    text = "alpha beta gamma"
    doc = parse_text(text, "doc.txt")
    cfg = StrategyConfig(
        "fixed", common={"chunk_size": 6, "chunk_overlap": 0},
        fixed={"use_tokens": False},
    )
    return run_chunking(doc, cfg, CharacterTokenizer())


def test_retrieval_span_containment_not_keyword_guessing():
    result = _fixed_result()
    questions = (
        GoldenQuestion(
            "q1", "alpha", (EvidenceAnchor("doc.txt", Span(0, 5)),)
        ),
        GoldenQuestion(
            "q2", "beta gamma", (EvidenceAnchor("doc.txt", Span(6, 16)),)
        ),
    )
    report = evaluate_retrieval([result], questions, k=1)
    assert report.total_queries == 2
    assert report.recall_at_k == pytest.approx(0.5)
    assert report.mrr_at_k == pytest.approx(0.5)
    assert report.ndcg_at_k == pytest.approx(0.5)
    assert report.oracle_recall == pytest.approx(0.5)
    assert report.questions[0].ranked[0].relevant
    assert not report.questions[1].oracle_has_evidence


def test_unknown_evidence_source_fails_not_silent_miss():
    with pytest.raises(ValueError, match="unknown sources"):
        evaluate_retrieval(
            [_fixed_result()],
            [GoldenQuestion(
                "q", "anything",
                (EvidenceAnchor("not-available", Span(0, 3)),),
            )],
        )


def test_nonfinite_scorer_is_rejected():
    q = GoldenQuestion(
        "q", "alpha", (EvidenceAnchor("doc.txt", Span(0, 5)),)
    )
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_retrieval(
            [_fixed_result()], [q], scorer=lambda q, text: float("nan")
        )


def test_duplicate_query_ids_fail():
    q = GoldenQuestion(
        "duplicate", "alpha", (EvidenceAnchor("doc.txt", Span(0, 5)),)
    )
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_retrieval([_fixed_result()], [q, q])


def test_benchmark_golden_fixture_and_atomic_output(tmp_path):
    path = Path("benchmarks/datasets/golden_v1.json")
    documents, questions = load_golden_dataset(path)
    assert len(documents) == 2
    assert len(questions) == 5
    report = run_benchmark(
        path, strategies=("fixed", "recursive", "hybrid"),
        tokenizer="char", chunk_size=128, chunk_overlap=16, k=3,
    )
    assert report["schema_version"] == 1
    for row in report["strategies"].values():
        assert row["integrity_ok"]
        assert row["unique_coverage"] == [1.0, 1.0]
        assert row["retrieval"]["total_queries"] == 5
        for metric in ("recall_at_k", "mrr_at_k", "ndcg_at_k", "oracle_recall"):
            assert 0 <= row["retrieval"][metric] <= 1.0
    target = tmp_path / "nested" / "report.json"
    write_report(target, report)
    assert json.loads(target.read_text(encoding="utf-8"))["scorer"] == (
        "lexical_term_overlap_v1"
    )
    assert not list(target.parent.glob("*.tmp"))


def test_golden_dataset_requires_unique_quote(tmp_path):
    bad = {
        "schema_version": 1,
        "documents": [{
            "source_uri": "a.txt", "format": "text", "text": "dup dup"
        }],
        "queries": [{
            "id": "q", "query": "test",
            "evidence": [{"source_uri": "a.txt", "quote": "dup"}],
        }],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_golden_dataset(path)


def test_online_strategies_are_refused_by_offline_benchmark():
    with pytest.raises(ValueError, match="offline"):
        run_benchmark(
            "benchmarks/datasets/golden_v1.json",
            strategies=("llm",),
        )


def test_golden_queries_require_nonempty_evidence():
    with pytest.raises(ValueError):
        GoldenQuestion("q", "Some question", ())
