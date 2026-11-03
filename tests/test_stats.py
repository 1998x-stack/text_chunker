"""Statistics must distinguish overlap duplication from unique coverage."""

import json
import time
from pathlib import Path

import pytest

from textchunker.stats import StatsCollector, track_time, count_calls
from textchunker.types import Chunk


def test_stats_collector_singleton():
    first = StatsCollector.instance()
    assert first is StatsCollector.instance()
    first.reset()


def test_stats_record_chunks():
    stats = StatsCollector.instance()
    stats.reset()
    stats.record_chunks([100, 200, 150], [50, 100, 75])
    result = stats.summary()
    assert result["total_chunks"] == 3
    assert result["avg_chunk_chars"] == pytest.approx(150.0)
    assert result["avg_chunk_tokens"] == pytest.approx(75.0)
    assert result["min_chunk_chars"] == 100
    assert result["max_chunk_chars"] == 200


def test_stats_record_document():
    stats = StatsCollector.instance()
    stats.reset()
    stats.record_document(1000, 250, 5, 0.5)
    assert stats.summary()["total_processing_time"] == pytest.approx(0.5)


def test_coverage_uses_union_of_spans():
    stats = StatsCollector.instance()
    stats.reset()
    source = "abcdefghij"
    chunks = [
        Chunk(0, "abcdefgh", 0, 8),
        Chunk(1, "fghij", 5, 10),
    ]
    stats.record_document(len(source), 10, len(chunks), 0.1)
    stats.record_chunks([len(item.text) for item in chunks], [8, 5])
    stats.record_coverage(len(source), chunks)
    result = stats.summary()
    assert result["coverage_ratio"] == pytest.approx(1.0)
    assert result["output_char_ratio"] == pytest.approx(1.3)
    assert result["redundancy_ratio"] == pytest.approx(0.3)


def test_unmeasured_coverage_is_unknown_not_invented():
    stats = StatsCollector.instance()
    stats.reset()
    stats.record_document(100, 50, 1, 0.1)
    assert stats.summary()["coverage_ratio"] is None


def test_stats_boundary_rate():
    stats = StatsCollector.instance()
    stats.reset()
    stats.record_chunks([500, 500, 300], [100, 100, 60], 500)
    assert stats.summary()["boundary_rate"] == pytest.approx(2 / 3)


def test_stats_save(tmp_path):
    stats = StatsCollector.instance()
    stats.reset()
    stats.record_document(500, 100, 2, 0.3)
    stats.record_chunks([250, 250], [50, 50])
    target = tmp_path / "run.json"
    stats.save_run(str(target), {"strategy": "fixed"})
    assert Path(target).exists()
    assert json.loads(target.read_text())["summary"]["total_chunks"] == 2


def test_track_time_decorator():
    stats = StatsCollector.instance()
    stats.reset()

    @track_time
    def run():
        time.sleep(0.01)
        return 42

    assert run() == 42
    assert stats.timings["run"] > 0


def test_count_calls_decorator():
    stats = StatsCollector.instance()
    stats.reset()

    @count_calls
    def run():
        return "hello"

    run()
    run()
    assert stats.call_counts["run"] == 2


def test_comparison_table():
    from textchunker.stats import format_comparison_table
    assert format_comparison_table([
        {"strategy": "fixed", "chunks": 2, "avg_chars": 5, "time": 0.1}
    ])
