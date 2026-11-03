"""Invariants required for all future chunkers, independent of algorithms."""

import pytest

from textchunker.core import Span, validate_chunks, validate_parameters
from textchunker.exceptions import ChunkerError, ConfigError
from textchunker.types import Chunk


def test_source_span_is_half_open_and_lossless():
    text = "A中🚀B"
    assert Span(1, 3).extract(text) == "中🚀"
    with pytest.raises(ValueError):
        Span(-1, 2)
    with pytest.raises(ValueError):
        Span(3, 2)


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, -1), (10, 10)])
def test_invalid_budgets_fail_fast(size, overlap):
    with pytest.raises(ConfigError):
        validate_parameters(size, overlap)


def test_invalid_max_chunks_fails_fast():
    with pytest.raises(ConfigError):
        validate_parameters(20, 3, max_chunks=0)


def test_source_gap_is_rejected():
    text = "abcdef"
    chunks = [
        Chunk(0, "abc", 0, 3),
        Chunk(1, "ef", 4, 6),
    ]
    with pytest.raises(ChunkerError, match="Uncovered"):
        validate_chunks(text, chunks)


def test_overlap_is_legal_and_coverage_is_exact():
    text = "abcdefgh"
    chunks = [
        Chunk(0, "abcde", 0, 5),
        Chunk(1, "defgh", 3, 8),
    ]
    validate_chunks(text, chunks, count=len, max_size=5)


def test_corrupt_source_mapping_is_rejected():
    with pytest.raises(ChunkerError, match="mapping"):
        validate_chunks("abc", [Chunk(0, "xxx", 0, 3)])
