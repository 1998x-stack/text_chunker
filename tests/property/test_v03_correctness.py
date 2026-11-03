"""An independent oracle must detect gaps, overlaps, corruption and budgets."""

import pytest

from textchunker.evaluation import evaluate_correctness
from textchunker.types import Chunk


def _chunk(index, text, start, end):
    return Chunk(index, text, start, end)


def test_unique_coverage_does_not_exceed_one_with_overlap():
    report = evaluate_correctness("abcdefgh", [
        _chunk(0, "abcde", 0, 5),
        _chunk(1, "defgh", 3, 8),
    ], len, 5)
    assert report.ok and report.complete
    assert report.unique_coverage_ratio == 1.0
    assert report.duplicated_characters == 2


@pytest.mark.parametrize("source", ["A🚀你好", "\n\n", "same same same", "abcdef"])
def test_gap_is_always_detected(source):
    if len(source) < 2:
        return
    chunks = [_chunk(0, source[:1], 0, 1)]
    result = evaluate_correctness(source, chunks, len, 50)
    assert not result.ok
    assert not result.complete
    assert result.missing_spans


def test_known_partial_mode_is_warning_not_successful_full_coverage():
    report = evaluate_correctness("abcdef", [
        _chunk(0, "abc", 0, 3)
    ], len, 3, require_full_coverage=False)
    assert report.ok
    assert not report.complete
    assert report.diagnostics[0].code == "INCOMPLETE"
    assert report.diagnostics[0].severity == "warning"


def test_detect_corrupted_text_even_with_full_span():
    report = evaluate_correctness("abcd", [
        _chunk(0, "xxxx", 0, 4)
    ], len, 4)
    assert not report.ok
    assert any(issue.code == "SOURCE_MISMATCH" for issue in report.diagnostics)


def test_detect_budget_violation():
    report = evaluate_correctness("abcdef", [
        _chunk(0, "abcdef", 0, 6)
    ], len, 3)
    assert not report.ok
    assert any(issue.code == "BUDGET_EXCEEDED" for issue in report.diagnostics)


def test_empty_document_is_complete_with_no_chunks():
    report = evaluate_correctness("", [], len, 10)
    assert report.ok and report.complete
    assert report.unique_coverage_ratio == 1.0


def test_invalid_chunk_bounds_do_not_crash_or_exaggerate_coverage():
    report = evaluate_correctness("abc", [_chunk(0, "xyz", -5, 4)],
                                  len, 5)
    assert not report.ok
    assert report.unique_coverage_ratio == 0.0
    assert any(issue.code == "SPAN_BOUNDS" for issue in report.diagnostics)
