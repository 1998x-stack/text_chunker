from __future__ import annotations

import json
import time
import functools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from rich.table import Table


class StatsCollector:
    """Singleton that collects chunking pipeline statistics."""

    _instance: Optional[StatsCollector] = None

    def __init__(self) -> None:
        self.reset()

    @classmethod
    def instance(cls) -> StatsCollector:
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

    def record_document(
        self,
        char_count: int,
        token_count: int,
        chunk_count: int,
        processing_time: float,
    ) -> None:
        self.doc_char_counts.append(char_count)
        self.doc_token_counts.append(token_count)
        self.doc_chunk_counts.append(chunk_count)
        self.doc_times.append(processing_time)

    def record_chunks(
        self,
        chunk_sizes: Sequence[int],
        chunk_tokens: Sequence[int],
        max_size: Optional[int] = None,
    ) -> None:
        self.chunk_sizes.extend(chunk_sizes)
        self.chunk_tokens.extend(chunk_tokens)
        if max_size is not None:
            self._max_size = max_size

    def summary(self) -> Dict[str, Any]:
        total_chunks = len(self.chunk_sizes)
        total_docs = len(self.doc_char_counts)
        total_doc_tokens = sum(self.doc_token_counts) if self.doc_token_counts else 0
        total_doc_chars = sum(self.doc_char_counts) if self.doc_char_counts else 0
        total_chunk_tokens = sum(self.chunk_tokens) if self.chunk_tokens else 0
        total_chunk_chars = sum(self.chunk_sizes) if self.chunk_sizes else 0

        if total_chunks > 0:
            arr_chars = np.array(self.chunk_sizes, dtype=float)
            arr_tokens = np.array(self.chunk_tokens, dtype=float)
            avg_chunk_chars = float(arr_chars.mean())
            avg_chunk_tokens = float(arr_tokens.mean())
            p50_chunk_chars = float(np.percentile(arr_chars, 50))
            p95_chunk_chars = float(np.percentile(arr_chars, 95))
            std_chunk_chars = float(arr_chars.std(ddof=0))
            min_chunk_chars = int(arr_chars.min())
            max_chunk_chars = int(arr_chars.max())
        else:
            avg_chunk_chars = avg_chunk_tokens = 0.0
            p50_chunk_chars = p95_chunk_chars = std_chunk_chars = 0.0
            min_chunk_chars = max_chunk_chars = 0

        redundancy_ratio = 0.0
        if total_doc_tokens > 0:
            redundancy_ratio = max(0.0, (total_chunk_tokens - total_doc_tokens) / total_doc_tokens)

        coverage_ratio = 0.0
        if total_doc_chars > 0:
            coverage_ratio = total_chunk_chars / total_doc_chars

        boundary_hits = 0
        if self._max_size is not None and total_chunks > 0:
            boundary_hits = sum(1 for s in self.chunk_sizes if s >= self._max_size)
        boundary_rate = boundary_hits / total_chunks if total_chunks > 0 else 0.0

        return {
            "total_docs": total_docs,
            "total_chunks": total_chunks,
            "avg_chunk_chars": avg_chunk_chars,
            "avg_chunk_tokens": avg_chunk_tokens,
            "p50_chunk_chars": p50_chunk_chars,
            "p95_chunk_chars": p95_chunk_chars,
            "std_chunk_chars": std_chunk_chars,
            "min_chunk_chars": min_chunk_chars,
            "max_chunk_chars": max_chunk_chars,
            "redundancy_ratio": redundancy_ratio,
            "coverage_ratio": coverage_ratio,
            "boundary_rate": boundary_rate,
            "total_processing_time": sum(self.doc_times),
            "avg_chunks_per_doc": sum(self.doc_chunk_counts) / total_docs if total_docs else 0.0,
        }

    def save_run(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
            "summary": self.summary(),
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


def track_time(func: Callable) -> Callable:
    """Decorator that records function execution time in StatsCollector."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        sc = StatsCollector.instance()
        sc.timings[func.__name__] = sc.timings.get(func.__name__, 0.0) + elapsed
        return result

    return wrapper


def count_calls(func: Callable) -> Callable:
    """Decorator that counts function invocations in StatsCollector."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sc = StatsCollector.instance()
        sc.call_counts[func.__name__] = sc.call_counts.get(func.__name__, 0) + 1
        return func(*args, **kwargs)

    return wrapper


def format_comparison_table(rows: Sequence[Dict[str, Any]]) -> Table:
    """Build a Rich Table comparing strategy results."""
    table = Table(title="Strategy Comparison", show_header=True, header_style="bold cyan")
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
