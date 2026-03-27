import json
import time
from pathlib import Path

import pytest

from textchunker.stats import StatsCollector, track_time, count_calls


def test_stats_collector_singleton():
    a = StatsCollector.instance()
    b = StatsCollector.instance()
    assert a is b
    a.reset()


def test_stats_record_chunks():
    sc = StatsCollector.instance()
    sc.reset()
    sc.record_chunks(chunk_sizes=[100, 200, 150], chunk_tokens=[50, 100, 75])
    summary = sc.summary()
    assert summary["total_chunks"] == 3
    assert summary["avg_chunk_chars"] == pytest.approx(150.0)
    assert summary["avg_chunk_tokens"] == pytest.approx(75.0)
    assert summary["min_chunk_chars"] == 100
    assert summary["max_chunk_chars"] == 200


def test_stats_record_document():
    sc = StatsCollector.instance()
    sc.reset()
    sc.record_document(char_count=1000, token_count=250, chunk_count=5, processing_time=0.5)
    summary = sc.summary()
    assert summary["total_docs"] == 1
    assert summary["total_processing_time"] == pytest.approx(0.5)


def test_stats_redundancy_and_coverage():
    sc = StatsCollector.instance()
    sc.reset()
    sc.record_document(char_count=1000, token_count=200, chunk_count=3, processing_time=0.1)
    sc.record_chunks(chunk_sizes=[400, 400, 400], chunk_tokens=[80, 80, 80])
    summary = sc.summary()
    assert summary["redundancy_ratio"] == pytest.approx((240 - 200) / 200)
    assert summary["coverage_ratio"] == pytest.approx(1200 / 1000)


def test_stats_boundary_rate():
    sc = StatsCollector.instance()
    sc.reset()
    sc.record_chunks(chunk_sizes=[500, 500, 300], chunk_tokens=[100, 100, 60], max_size=500)
    summary = sc.summary()
    assert summary["boundary_rate"] == pytest.approx(2 / 3)


def test_stats_save_and_load(tmp_path):
    sc = StatsCollector.instance()
    sc.reset()
    sc.record_document(char_count=500, token_count=100, chunk_count=2, processing_time=0.3)
    sc.record_chunks(chunk_sizes=[250, 250], chunk_tokens=[50, 50])

    save_path = str(tmp_path / "run.json")
    sc.save_run(save_path, metadata={"strategy": "fixed"})

    assert Path(save_path).exists()
    with open(save_path, "r") as f:
        data = json.load(f)
    assert data["metadata"]["strategy"] == "fixed"
    assert data["summary"]["total_chunks"] == 2


def test_track_time_decorator():
    sc = StatsCollector.instance()
    sc.reset()

    @track_time
    def slow_func():
        time.sleep(0.05)
        return 42

    result = slow_func()
    assert result == 42
    assert "slow_func" in sc.timings
    assert sc.timings["slow_func"] >= 0.04


def test_count_calls_decorator():
    sc = StatsCollector.instance()
    sc.reset()

    @count_calls
    def my_func():
        return "hello"

    my_func()
    my_func()
    my_func()
    assert sc.call_counts["my_func"] == 3


def test_compare_strategies_output():
    from textchunker.stats import format_comparison_table
    rows = [
        {"strategy": "fixed", "chunks": 10, "avg_chars": 200, "time": 0.1},
        {"strategy": "recursive", "chunks": 8, "avg_chars": 250, "time": 0.15},
    ]
    table = format_comparison_table(rows)
    assert table is not None
