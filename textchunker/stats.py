"""Lightweight run metrics and source-span coverage accounting."""

from __future__ import annotations

import functools
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from rich.table import Table

from .types import Chunk


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return float(ordered[lower] + (ordered[upper] - ordered[lower])
                 * (index - lower))


class StatsCollector:
    """Process-local singleton. v0.3 will replace this with a run-scoped object."""

    _instance: Optional["StatsCollector"] = None

    def __init__(self) -> None:
        self.reset()

    @classmethod
    def instance(cls) -> "StatsCollector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self) -> None:
        self.chunk_sizes: List[int] = []
        self.chunk_tokens: List[int] = []
        self.doc_char_counts: List[int] = []
        self.doc_token_counts: List[int] = []
        self.doc_chunk_counts: List[int] = []
        self.doc_times: List[float] = []
        self.timings: Dict[str, float] = {}
        self.call_counts: Dict[str, int] = {}
        self._max_size: Optional[int] = None
        self._covered_chars = 0
        self._measured_chars = 0

    def record_document(self, char_count: int, token_count: int,
                        chunk_count: int, processing_time: float) -> None:
        self.doc_char_counts.append(char_count)
        self.doc_token_counts.append(token_count)
        self.doc_chunk_counts.append(chunk_count)
        self.doc_times.append(processing_time)

    def record_chunks(self, chunk_sizes: Sequence[int],
                      chunk_tokens: Sequence[int],
                      max_size: Optional[int] = None) -> None:
        self.chunk_sizes.extend(chunk_sizes)
        self.chunk_tokens.extend(chunk_tokens)
        if max_size is not None:
            self._max_size = max_size

    def record_coverage(self, source_length: int,
                        chunks: Sequence[Chunk]) -> None:
        """Count unique covered characters, not overlap duplicates."""
        self._measured_chars += source_length
        covered_until = 0
        for chunk in sorted(chunks, key=lambda item: (item.start, item.end)):
            self._covered_chars += max(0, chunk.end - max(chunk.start, covered_until))
            covered_until = max(covered_until, chunk.end)

    def summary(self) -> Dict[str, Any]:
        n = len(self.chunk_sizes)
        docs = len(self.doc_char_counts)
        chars = self.chunk_sizes
        tokens = self.chunk_tokens
        total_doc_chars = sum(self.doc_char_counts)
        total_doc_tokens = sum(self.doc_token_counts)
        total_chunk_tokens = sum(tokens)
        output_ratio = (sum(chars) / total_doc_chars) if total_doc_chars else 0.0

        return {
            "total_docs": docs,
            "total_chunks": n,
            "avg_chunk_chars": statistics.mean(chars) if n else 0.0,
            "avg_chunk_tokens": statistics.mean(tokens) if n else 0.0,
            "p50_chunk_chars": _percentile(chars, 50),
            "p95_chunk_chars": _percentile(chars, 95),
            "std_chunk_chars": statistics.pstdev(chars) if n else 0.0,
            "min_chunk_chars": min(chars) if n else 0,
            "max_chunk_chars": max(chars) if n else 0,
            "redundancy_ratio": max(
                0.0, (total_chunk_tokens - total_doc_tokens)
                / total_doc_tokens
            ) if total_doc_tokens else 0.0,
            "output_char_ratio": output_ratio,
            "coverage_ratio": (
                self._covered_chars / self._measured_chars
                if self._measured_chars else None
            ),
            "boundary_rate": (
                sum(1 for item in chars if item >= self._max_size) / n
                if n and self._max_size is not None else 0.0
            ),
            "total_processing_time": sum(self.doc_times),
            "avg_chunks_per_doc": (
                sum(self.doc_chunk_counts) / docs if docs else 0.0
            ),
        }

    def save_run(self, path: str,
                 metadata: Optional[Dict[str, Any]] = None) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "summary": self.summary(),
        }
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")


def track_time(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        stats = StatsCollector.instance()
        stats.timings[func.__name__] = (
            stats.timings.get(func.__name__, 0.0) + elapsed
        )
        return result
    return wrapper


def count_calls(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        stats = StatsCollector.instance()
        stats.call_counts[func.__name__] = (
            stats.call_counts.get(func.__name__, 0) + 1
        )
        return func(*args, **kwargs)
    return wrapper


def format_comparison_table(rows: Sequence[Dict[str, Any]]) -> Table:
    table = Table(title="Strategy Comparison", show_header=True,
                  header_style="bold cyan")
    table.add_column("Strategy")
    table.add_column("Chunks", justify="right")
    table.add_column("Avg Chars", justify="right")
    table.add_column("Time (s)", justify="right")
    for row in rows:
        table.add_row(
            str(row.get("strategy", "")),
            str(row.get("chunks", "")),
            f"{row.get('avg_chars', 0):.1f}",
            f"{row.get('time', 0):.3f}",
        )
    return table
