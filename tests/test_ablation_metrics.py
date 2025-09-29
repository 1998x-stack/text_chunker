import numpy as np
import pytest

from textchunker.experiments.semantic_ablation import Scenario, summarize_metrics


def test_summarize_metrics_basic():
    scenario = Scenario(chunk_size=500, chunk_overlap=100, min_similarity=0.6, sentence_window=1)
    metrics = summarize_metrics(
        scenario,
        doc_token_counts=[100, 120],
        doc_char_counts=[500, 620],
        chunk_token_counts=[80, 70, 90, 85],
        chunk_char_counts=[400, 350, 420, 360],
        chunk_counts=[2, 2],
        boundary_hits=1,
    )

    assert metrics.scenario == scenario
    assert metrics.total_docs == 2
    assert metrics.total_chunks == 4
    assert metrics.avg_chunks_per_doc == pytest.approx(2.0)
    assert metrics.avg_doc_tokens == pytest.approx(110.0)
    assert metrics.avg_chunk_tokens == pytest.approx(np.mean([80, 70, 90, 85]))
    assert metrics.avg_chunk_chars == pytest.approx(np.mean([400, 350, 420, 360]))
    assert metrics.p95_chunk_tokens == pytest.approx(np.percentile([80, 70, 90, 85], 95))
    assert metrics.std_chunk_tokens == pytest.approx(np.std([80, 70, 90, 85]))
    assert metrics.redundancy_ratio == pytest.approx((325 - 220) / 220)
    assert metrics.boundary_rate == pytest.approx(0.25)
    assert metrics.coverage_ratio == pytest.approx(1530 / 1120)


def test_summarize_metrics_requires_documents():
    scenario = Scenario(chunk_size=600, chunk_overlap=50, min_similarity=0.62, sentence_window=2)
    with pytest.raises(ValueError):
        summarize_metrics(
            scenario,
            doc_token_counts=[],
            doc_char_counts=[],
            chunk_token_counts=[],
            chunk_char_counts=[],
            chunk_counts=[],
            boundary_hits=0,
        )
