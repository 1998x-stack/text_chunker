# Text-Chunker Enhancement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enhance the text-chunker project to professional-grade quality — fix bugs, add DashScope/Settings integration, advanced logging, statistics system, comprehensive tests, experiments, bash scripts, Docker, and polished documentation.

**Architecture:** Layered enhancement in-place. New modules (`settings.py`, `logging.py`, `stats.py`, `exceptions.py`) are added alongside existing code. Providers refactored to support DashScope API with `DASHSCOPE_API_KEY`. All existing chunker modules receive bug fixes and code quality improvements. Tests expanded from 4 files to 14 files.

**Tech Stack:** Python 3.9+, loguru, rich, tiktoken, openai SDK (DashScope-compatible), sentence-transformers, numpy, pytest, Docker

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `textchunker/exceptions.py` | Custom exception hierarchy: `ChunkerError`, `ConfigError`, `ProviderError` |
| `textchunker/settings.py` | `Settings` dataclass — API keys, model names, log/stats config |
| `textchunker/logging.py` | `setup_logging()` — loguru console + JSON file sinks with rotation |
| `textchunker/stats.py` | `StatsCollector` singleton, `@track_time`/`@count_calls` decorators, comparison |
| `textchunker/providers/dashscope_provider.py` | DashScope LLM provider (OpenAI-compatible SDK) |
| `textchunker/providers/embeddings.py` | `DashScopeEmbeddingProvider`, `SentenceTransformerProvider` |
| `textchunker/experiments/recursive_ablation.py` | Grid search for recursive chunker params |
| `textchunker/experiments/strategy_comparison.py` | Cross-strategy head-to-head comparison |
| `tests/test_fixed_chunker.py` | FixedChunker tests |
| `tests/test_recursive_chunker.py` | RecursiveChunker tests |
| `tests/test_structure_chunker.py` | StructureChunker tests |
| `tests/test_llm_chunker.py` | LLMChunker tests (mocked providers) |
| `tests/test_readers.py` | Reader tests (txt, html, pdf) |
| `tests/test_config.py` | Config loading + CLI merge tests |
| `tests/test_settings.py` | Settings from env/yaml tests |
| `tests/test_logging_setup.py` | Logging setup tests |
| `tests/test_stats.py` | StatsCollector + decorator tests |
| `tests/test_utils.py` | Utils function tests |
| `tests/test_registry.py` | Registry tests |
| `tests/test_providers.py` | Provider tests (mocked API) |
| `scripts/install.sh` | Install with all extras |
| `scripts/test.sh` | Run pytest with coverage |
| `scripts/lint.sh` | Run flake8 + mypy |
| `scripts/format.sh` | Run isort + black |
| `scripts/benchmark.sh` | Benchmark all strategies |
| `scripts/experiment.sh` | Run ablation experiments |
| `scripts/ci.sh` | Full CI pipeline |
| `scripts/docker-build.sh` | Build Docker image |
| `Makefile` | All build targets |
| `Dockerfile` | Multi-stage container image |
| `docker-compose.yml` | Container orchestration |
| `.dockerignore` | Docker exclusions |
| `.claude/claude.md` | Project guide for Claude Code |
| `docs/strategy-guide.md` | Moved from Doc.md |

### Modified Files

| File | Changes |
|------|---------|
| `textchunker/__init__.py` | Expand `__all__` |
| `textchunker/types.py` | No changes (already clean) |
| `textchunker/utils.py` | Add `whitespace_sentences_with_offsets()` |
| `textchunker/config.py` | Add new CLI args (`--stats`, `--log-level`, etc.), integrate Settings |
| `textchunker/cli.py` | Integrate logging setup, stats, DashScope provider |
| `textchunker/chunkers/fixed.py` | Implement token mode properly |
| `textchunker/chunkers/semantic.py` | Fix offset tracking, add DashScope embedding option |
| `textchunker/chunkers/recursive.py` | Fix offset tracking with cumulative cursor |
| `textchunker/chunkers/structure.py` | Fix offset tracking in sub-splits |
| `textchunker/chunkers/llm_based.py` | Add DashScope provider option, improve error handling |
| `textchunker/chunkers/__init__.py` | Add `__all__` |
| `textchunker/providers/__init__.py` | Export new providers |
| `textchunker/providers/llm.py` | Improve JSON parsing, add logging |
| `textchunker/experiments/__init__.py` | Export new experiments |
| `textchunker/experiments/semantic_ablation.py` | Add Rich progress bar, CSV export |
| `tests/conftest.py` | Expand with shared fixtures |
| `setup.cfg` | Add new deps, bump version |
| `requirements.txt` | Add new deps |
| `configs/default.yaml` | Add settings + logging sections, switch to DashScope |
| `configs/semantic.yaml` | Add embedding_model |
| `configs/llm_based.yaml` | Switch to DashScope/qwen-max |
| `README.md` | Complete rewrite |

---

### Task 1: Exceptions Module

**Files:**
- Create: `textchunker/exceptions.py`
- Test: `tests/test_exceptions.py` (inline — no separate test file needed, tested via usage)

- [ ] **Step 1: Create exceptions module**

Create `textchunker/exceptions.py`:

```python
from __future__ import annotations


class ChunkerError(Exception):
    """Base exception for all text-chunker errors."""
    pass


class ConfigError(ChunkerError):
    """Raised when configuration is invalid or missing."""
    pass


class ProviderError(ChunkerError):
    """Raised when an LLM or embedding provider fails."""
    pass
```

- [ ] **Step 2: Verify import works**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -c "from textchunker.exceptions import ChunkerError, ConfigError, ProviderError; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add textchunker/exceptions.py
git commit -m "feat: add custom exception hierarchy (ChunkerError, ConfigError, ProviderError)"
```

---

### Task 2: Settings Class

**Files:**
- Create: `textchunker/settings.py`
- Create: `tests/test_settings.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_settings.py`:

```python
import os
import pytest
from textchunker.settings import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-123")
    s = Settings.from_env()
    assert s.api_key == "test-key-123"
    assert s.llm_model == "qwen-max"
    assert s.embedding_model == "text-embedding-v3"
    assert s.api_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_settings_from_env_missing_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    s = Settings.from_env()
    assert s.api_key == ""


def test_settings_from_yaml():
    yaml_cfg = {
        "settings": {
            "llm_model": "qwen-turbo",
            "embedding_model": "text-embedding-v2",
            "api_base_url": "https://custom.endpoint/v1",
        },
        "logging": {
            "level": "DEBUG",
            "json": True,
            "log_dir": "custom_logs/",
            "rotation": "5 MB",
            "retention": 3,
        },
    }
    s = Settings.from_env_and_yaml(yaml_cfg)
    assert s.llm_model == "qwen-turbo"
    assert s.embedding_model == "text-embedding-v2"
    assert s.log_level == "DEBUG"
    assert s.log_json is True
    assert s.log_dir == "custom_logs/"
    assert s.log_rotation == "5 MB"
    assert s.log_retention == 3


def test_settings_defaults():
    s = Settings.from_env_and_yaml({})
    assert s.llm_model == "qwen-max"
    assert s.log_level == "INFO"
    assert s.log_json is False
    assert s.stats_enabled is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'textchunker.settings'`

- [ ] **Step 3: Write Settings implementation**

Create `textchunker/settings.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Settings:
    """Centralized configuration for API, logging, and stats."""

    # API
    api_key: str = ""
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Models
    llm_model: str = "qwen-max"
    embedding_model: str = "text-embedding-v3"

    # Logging
    log_level: str = "INFO"
    log_json: bool = False
    log_dir: str = "logs/"
    log_rotation: str = "10 MB"
    log_retention: int = 5

    # Stats
    stats_enabled: bool = False
    stats_dir: str = "stats/"

    @classmethod
    def from_env(cls) -> Settings:
        """Create Settings from environment variables only."""
        return cls(api_key=os.environ.get("DASHSCOPE_API_KEY", ""))

    @classmethod
    def from_env_and_yaml(cls, yaml_cfg: Dict[str, Any]) -> Settings:
        """Create Settings from environment + YAML config sections."""
        settings_section = yaml_cfg.get("settings", {})
        logging_section = yaml_cfg.get("logging", {})

        return cls(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            api_base_url=settings_section.get(
                "api_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            llm_model=settings_section.get("llm_model", "qwen-max"),
            embedding_model=settings_section.get("embedding_model", "text-embedding-v3"),
            log_level=logging_section.get("level", "INFO"),
            log_json=bool(logging_section.get("json", False)),
            log_dir=logging_section.get("log_dir", "logs/"),
            log_rotation=logging_section.get("rotation", "10 MB"),
            log_retention=int(logging_section.get("retention", 5)),
            stats_enabled=bool(settings_section.get("stats_enabled", False)),
            stats_dir=settings_section.get("stats_dir", "stats/"),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_settings.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add textchunker/settings.py tests/test_settings.py
git commit -m "feat: add Settings dataclass with DashScope API defaults and YAML/env loading"
```

---

### Task 3: Advanced Loguru Logging System

**Files:**
- Create: `textchunker/logging.py`
- Create: `tests/test_logging_setup.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_logging_setup.py`:

```python
import json
import os
from pathlib import Path

from loguru import logger

from textchunker.logging import setup_logging
from textchunker.settings import Settings


def test_setup_logging_configures_console(capsys):
    settings = Settings(log_level="DEBUG", log_json=False, log_dir="logs/")
    setup_logging(settings)
    logger.info("test message")
    # loguru writes to stderr by default
    # Verify no crash — configuration is the key test
    assert True


def test_setup_logging_json_file(tmp_path):
    log_dir = str(tmp_path / "logs")
    settings = Settings(
        log_level="DEBUG",
        log_json=True,
        log_dir=log_dir,
        log_rotation="1 MB",
        log_retention=2,
    )
    setup_logging(settings)
    logger.info("json test message")
    logger.complete()

    log_files = list(Path(log_dir).glob("*.log"))
    assert len(log_files) >= 1

    with open(log_files[0], "r") as f:
        lines = f.readlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert "text" in record
    assert "json test message" in record["text"]


def test_setup_logging_level_filtering(tmp_path):
    log_dir = str(tmp_path / "logs")
    settings = Settings(log_level="WARNING", log_json=True, log_dir=log_dir)
    setup_logging(settings)
    logger.debug("should be filtered")
    logger.warning("should appear")
    logger.complete()

    log_files = list(Path(log_dir).glob("*.log"))
    if log_files:
        with open(log_files[0], "r") as f:
            content = f.read()
        assert "should be filtered" not in content
        assert "should appear" in content


def test_setup_logging_creates_log_dir(tmp_path):
    log_dir = str(tmp_path / "nested" / "logs")
    settings = Settings(log_level="INFO", log_json=False, log_dir=log_dir)
    setup_logging(settings)
    assert Path(log_dir).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_logging_setup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'textchunker.logging'`

- [ ] **Step 3: Write logging implementation**

Create `textchunker/logging.py`:

```python
from __future__ import annotations

import os
import sys
from pathlib import Path

from loguru import logger

from .settings import Settings

_CONSOLE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level:<8}</level> | "
    "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)


def setup_logging(settings: Settings) -> None:
    """Configure loguru with console + optional JSON file sinks.

    Removes all existing handlers first to allow re-configuration.
    """
    logger.remove()

    # Console sink — human-readable, colored
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=settings.log_level.upper(),
        colorize=True,
    )

    # File sink — JSON structured logs with rotation
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "textchunker.log"

    logger.add(
        str(log_file),
        format="{message}",
        level=settings.log_level.upper(),
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        serialize=settings.log_json,
        encoding="utf-8",
        enqueue=True,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_logging_setup.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add textchunker/logging.py tests/test_logging_setup.py
git commit -m "feat: add advanced loguru logging with console + JSON file sinks, rotation, and level filtering"
```

---

### Task 4: Statistics System

**Files:**
- Create: `textchunker/stats.py`
- Create: `tests/test_stats.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stats.py`:

```python
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
    assert table is not None  # Rich Table object
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'textchunker.stats'`

- [ ] **Step 3: Write stats implementation**

Create `textchunker/stats.py`:

```python
from __future__ import annotations

import json
import time
import functools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from rich.table import Table

from .exceptions import ChunkerError


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_stats.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add textchunker/stats.py tests/test_stats.py
git commit -m "feat: add StatsCollector singleton with decorators, comparison table, and run persistence"
```

---

### Task 5: Fix `utils.py` — Add Offset-Aware Sentence Splitting

**Files:**
- Modify: `textchunker/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_utils.py`:

```python
import pytest

from textchunker import utils as _utils


def test_count_tokens_returns_positive():
    assert _utils.count_tokens("hello world") > 0


def test_count_tokens_empty():
    assert _utils.count_tokens("") == 0


def test_whitespace_sentences_english():
    text = "First sentence. Second sentence. Third."
    sents = _utils.whitespace_sentences(text)
    assert len(sents) >= 2
    assert "First" in sents[0]


def test_whitespace_sentences_chinese():
    text = "第一句话。 第二句话！ 第三句话？"
    sents = _utils.whitespace_sentences(text)
    assert len(sents) >= 2


def test_whitespace_sentences_with_offsets():
    text = "First sentence. Second sentence. Third sentence."
    results = _utils.whitespace_sentences_with_offsets(text)
    for sent, start, end in results:
        assert text[start:end] == sent
        assert len(sent) > 0


def test_whitespace_sentences_with_offsets_no_gaps():
    text = "Hello world. How are you? I am fine."
    results = _utils.whitespace_sentences_with_offsets(text)
    # Verify offsets cover the text without contradiction
    for sent, start, end in results:
        assert start >= 0
        assert end <= len(text)
        assert start < end


def test_whitespace_sentences_with_offsets_single():
    text = "Just one sentence"
    results = _utils.whitespace_sentences_with_offsets(text)
    assert len(results) == 1
    assert results[0][0] == text.strip()


def test_clamp():
    assert _utils.clamp(5, 0, 10) == 5
    assert _utils.clamp(-1, 0, 10) == 0
    assert _utils.clamp(15, 0, 10) == 10


def test_sliding_windows():
    seq = ["a", "b", "c", "d"]
    result = list(_utils.sliding_windows(seq, 2))
    assert len(result) == 3
    assert result[0] == (0, ["a", "b"])
    assert result[2] == (2, ["c", "d"])


def test_sliding_windows_larger_than_seq():
    seq = ["a", "b"]
    result = list(_utils.sliding_windows(seq, 5))
    assert len(result) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_utils.py -v`
Expected: FAIL on `test_whitespace_sentences_with_offsets` — `AttributeError: module has no attribute 'whitespace_sentences_with_offsets'`

- [ ] **Step 3: Add `whitespace_sentences_with_offsets` to utils.py**

Add after the existing `whitespace_sentences` function in `textchunker/utils.py`:

```python
def whitespace_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """Sentence splitting that returns (sentence, start, end) tuples.

    Uses the same regex as whitespace_sentences but tracks character offsets,
    eliminating the need for text.find() which fails on duplicate sentences.
    """
    pat = r'(?<=[。！？!?；;])\s+|(?<=\.)\s+'
    stripped = text.strip()
    if not stripped:
        return []
    parts = re.split(pat, stripped)
    offset = text.index(stripped[0]) if stripped else 0
    results: List[Tuple[str, int, int]] = []
    cursor = offset
    for part in parts:
        if not part:
            continue
        start = text.find(part, cursor)
        if start == -1:
            start = cursor
        end = start + len(part)
        results.append((part, start, end))
        cursor = end
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_utils.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
git add textchunker/utils.py tests/test_utils.py
git commit -m "feat: add whitespace_sentences_with_offsets for safe offset tracking"
```

---

### Task 6: Fix FixedChunker — Implement Token Mode

**Files:**
- Modify: `textchunker/chunkers/fixed.py`
- Create: `tests/test_fixed_chunker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_fixed_chunker.py`:

```python
import pytest

from textchunker.types import StrategyConfig
from textchunker.chunkers.fixed import FixedChunker


def _cfg(**overrides):
    common = {"chunk_size": 50, "chunk_overlap": 10}
    fixed = {"use_tokens": False}
    common.update(overrides.get("common", {}))
    fixed.update(overrides.get("fixed", {}))
    return StrategyConfig(
        name="fixed", common=common, fixed=fixed,
        semantic={}, recursive={}, structure={}, llm={},
    )


def test_fixed_chunker_basic():
    text = "a" * 120
    chunker = FixedChunker(_cfg())
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) <= 50
        assert c.start >= 0
        assert c.end <= len(text)
        assert text[c.start:c.end] == c.text


def test_fixed_chunker_overlap():
    text = "a" * 100
    chunker = FixedChunker(_cfg(common={"chunk_size": 50, "chunk_overlap": 20}))
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    # Second chunk should start 20 chars before end of first
    assert chunks[1].start == chunks[0].end - 20


def test_fixed_chunker_empty_text():
    chunker = FixedChunker(_cfg())
    chunks = chunker.chunk("")
    assert chunks == []


def test_fixed_chunker_text_smaller_than_size():
    text = "short"
    chunker = FixedChunker(_cfg(common={"chunk_size": 100, "chunk_overlap": 10}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0].text == "short"


def test_fixed_chunker_max_chunks():
    text = "a" * 500
    chunker = FixedChunker(_cfg(common={"chunk_size": 50, "chunk_overlap": 0, "max_chunks": 3}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 3


def test_fixed_chunker_token_mode():
    """Token mode should produce chunks where each chunk's token count <= chunk_size."""
    from textchunker.utils import count_tokens, _enc
    if _enc is None:
        pytest.skip("tiktoken not available")
    text = "Hello world. " * 100  # ~300 tokens
    chunker = FixedChunker(_cfg(
        common={"chunk_size": 50, "chunk_overlap": 10},
        fixed={"use_tokens": True},
    ))
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2
    for c in chunks:
        assert count_tokens(c.text) <= 55  # small tolerance for boundary
        assert c.start >= 0
        assert c.end <= len(text)


def test_fixed_chunker_token_mode_fallback():
    """When tiktoken unavailable, token mode falls back to character mode."""
    import textchunker.utils as utils_mod
    original_enc = utils_mod._enc
    utils_mod._enc = None
    try:
        text = "a" * 120
        chunker = FixedChunker(_cfg(
            common={"chunk_size": 50, "chunk_overlap": 10},
            fixed={"use_tokens": True},
        ))
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2
    finally:
        utils_mod._enc = original_enc
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_fixed_chunker.py -v`
Expected: `test_fixed_chunker_token_mode` FAILS — token mode doesn't produce correct chunks (the `pass` block)

- [ ] **Step 3: Rewrite FixedChunker**

Replace the entire content of `textchunker/chunkers/fixed.py`:

```python
from __future__ import annotations

from typing import List

from loguru import logger

from ..exceptions import ChunkerError
from ..types import Chunk
from ..utils import count_tokens
from ..registry import register
from .base import BaseChunker


try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    _enc = None


@register("fixed")
class FixedChunker(BaseChunker):
    """Fixed-size sliding window chunker (character or token based)."""

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        f = self.cfg.fixed
        size = int(c.get("chunk_size", 512))
        overlap = int(c.get("chunk_overlap", 80))
        use_tokens = bool(f.get("use_tokens", True))
        max_chunks = c.get("max_chunks")

        if not text:
            return []

        if use_tokens and _enc is not None:
            return self._chunk_by_tokens(text, size, overlap, max_chunks)

        if use_tokens and _enc is None:
            logger.warning("tiktoken unavailable, falling back to character-based chunking")

        return self._chunk_by_chars(text, size, overlap, max_chunks)

    def _chunk_by_chars(self, text: str, size: int, overlap: int, max_chunks: int | None) -> List[Chunk]:
        chunks: List[Chunk] = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + size, n)
            piece = text[start:end]
            chunks.append(Chunk(
                id=len(chunks), text=piece, start=start, end=end,
                meta={"strategy": "fixed", "mode": "char"},
            ))
            if max_chunks and len(chunks) >= max_chunks:
                break
            if end == n:
                break
            start = end - overlap if overlap < (end - start) else start + 1
        logger.info("Fixed chunking (char): {} chunks from {} chars", len(chunks), n)
        return chunks

    def _chunk_by_tokens(self, text: str, size: int, overlap: int, max_chunks: int | None) -> List[Chunk]:
        tokens = _enc.encode(text)
        chunks: List[Chunk] = []
        tok_start = 0
        n = len(tokens)
        while tok_start < n:
            tok_end = min(tok_start + size, n)
            chunk_tokens = tokens[tok_start:tok_end]
            piece = _enc.decode(chunk_tokens)
            # Map token positions back to character positions
            char_start = len(_enc.decode(tokens[:tok_start]))
            char_end = len(_enc.decode(tokens[:tok_end]))
            chunks.append(Chunk(
                id=len(chunks), text=piece, start=char_start, end=char_end,
                meta={"strategy": "fixed", "mode": "token"},
            ))
            if max_chunks and len(chunks) >= max_chunks:
                break
            if tok_end == n:
                break
            tok_start = tok_end - overlap if overlap < (tok_end - tok_start) else tok_start + 1
        logger.info("Fixed chunking (token): {} chunks from {} tokens", len(chunks), n)
        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_fixed_chunker.py -v`
Expected: All tests PASS (token mode test may skip if tiktoken unavailable)

- [ ] **Step 5: Commit**

```bash
git add textchunker/chunkers/fixed.py tests/test_fixed_chunker.py
git commit -m "fix: implement proper token-based chunking in FixedChunker with tiktoken encode/decode"
```

---

### Task 7: Fix SemanticChunker — Use Offset Map

**Files:**
- Modify: `textchunker/chunkers/semantic.py`
- Modify: existing `tests/test_semantic_chunker.py`

- [ ] **Step 1: Write additional failing test**

Add to `tests/test_semantic_chunker.py`:

```python
def test_semantic_chunker_handles_duplicate_sentences():
    """Duplicate sentences should have correct, non-overlapping offsets."""
    text = "Hello world. Hello world. Goodbye world."
    chunker = SemanticChunker(_strategy_cfg())
    chunks = chunker.chunk(text)
    # All chunks must have valid start/end
    for c in chunks:
        assert c.start >= 0
        assert c.end <= len(text)
        assert c.start < c.end
    # No two chunks should have the exact same start
    starts = [c.start for c in chunks]
    assert len(starts) == len(set(starts))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_semantic_chunker.py::test_semantic_chunker_handles_duplicate_sentences -v`
Expected: May FAIL due to duplicate sentence offset bug

- [ ] **Step 3: Update SemanticChunker to use `whitespace_sentences_with_offsets`**

Replace the content of `textchunker/chunkers/semantic.py`:

```python
from __future__ import annotations

from typing import List

import numpy as np  # type: ignore
from loguru import logger
from sentence_transformers import SentenceTransformer  # type: ignore

from ..exceptions import ChunkerError
from ..types import Chunk
from ..utils import whitespace_sentences_with_offsets, count_tokens
from ..registry import register
from .base import BaseChunker


@register("semantic")
class SemanticChunker(BaseChunker):
    """Semantic chunker: merges sentences by embedding similarity."""

    def __init__(self, cfg):
        super().__init__(cfg)
        model_name = self.cfg.semantic.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loading sentence-transformers model: {}", model_name)
        self.model = SentenceTransformer(model_name)

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        s = self.cfg.semantic
        size = int(c.get("chunk_size", 600))
        overlap = int(c.get("chunk_overlap", 100))
        min_sim = float(s.get("min_similarity", 0.62))
        window = int(s.get("sentence_window", 1))
        max_chunks = c.get("max_chunks")

        sent_data = whitespace_sentences_with_offsets(text)
        if not sent_data:
            return [Chunk(id=0, text=text, start=0, end=len(text), meta={"strategy": "semantic"})]

        sents = [sd[0] for sd in sent_data]
        offsets = [(sd[1], sd[2]) for sd in sent_data]

        embs = self.model.encode(sents, normalize_embeddings=True)
        chunks: List[Chunk] = []
        buf_indices: List[int] = []

        def flush() -> None:
            if not buf_indices:
                return
            chunk_start = offsets[buf_indices[0]][0]
            chunk_end = offsets[buf_indices[-1]][1]
            piece = text[chunk_start:chunk_end].strip()
            if piece:
                chunks.append(Chunk(
                    id=len(chunks), text=piece, start=chunk_start, end=chunk_end,
                    meta={"strategy": "semantic"},
                ))
            buf_indices.clear()

        for i in range(len(sents)):
            if buf_indices and i >= window:
                v = embs[i]
                ctx = embs[max(0, i - window):i].mean(axis=0)
                sim = float(np.dot(v, ctx))
                if sim < min_sim:
                    flush()
                    if max_chunks and len(chunks) >= max_chunks:
                        return chunks

            buf_indices.append(i)

            buf_text = text[offsets[buf_indices[0]][0]:offsets[i][1]]
            if count_tokens(buf_text) >= size:
                flush()
                if max_chunks and len(chunks) >= max_chunks:
                    return chunks

                if overlap > 0:
                    back_target = offsets[i][1] - overlap
                    j = i
                    while j >= 0 and offsets[j][0] > back_target:
                        j -= 1
                    if j + 1 <= i:
                        buf_indices.append(i)
                    # else: no overlap possible
                continue

            if max_chunks and len(chunks) >= max_chunks:
                break

        if buf_indices and (not max_chunks or len(chunks) < max_chunks):
            flush()

        logger.info("Semantic chunking: {} chunks from {} sentences", len(chunks), len(sents))
        return chunks
```

- [ ] **Step 4: Run all semantic tests**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_semantic_chunker.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add textchunker/chunkers/semantic.py tests/test_semantic_chunker.py
git commit -m "fix: SemanticChunker uses offset map instead of text.find(), fixing duplicate sentence bug"
```

---

### Task 8: Fix RecursiveChunker — Cumulative Offset Tracking

**Files:**
- Modify: `textchunker/chunkers/recursive.py`
- Create: `tests/test_recursive_chunker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_recursive_chunker.py`:

```python
import pytest

from textchunker.types import StrategyConfig
from textchunker.chunkers.recursive import RecursiveChunker


def _cfg(**overrides):
    common = {"chunk_size": 50, "chunk_overlap": 0}
    recursive = {"separators": ["\n\n", "\n", " ", ""]}
    common.update(overrides.get("common", {}))
    recursive.update(overrides.get("recursive", {}))
    return StrategyConfig(
        name="recursive", common=common, recursive=recursive,
        semantic={}, fixed={}, structure={}, llm={},
    )


def test_recursive_chunker_basic():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 100, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    for c in chunks:
        assert text[c.start:c.end] == c.text or c.text.startswith(text[c.start:c.start + 10])


def test_recursive_chunker_offset_correctness():
    text = "AAA.\n\nBBB.\n\nCCC."
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 10, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    for c in chunks:
        assert c.start >= 0
        assert c.end <= len(text)
        assert text[c.start:c.end] == c.text


def test_recursive_chunker_with_overlap():
    text = "Word " * 50  # 250 chars
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 50, "chunk_overlap": 10}))
    chunks = chunker.chunk(text)
    assert len(chunks) >= 2


def test_recursive_chunker_duplicate_content():
    text = "Same line.\n\nSame line.\n\nSame line."
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 20, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    starts = [c.start for c in chunks]
    # Each chunk should have a unique start position
    assert len(starts) == len(set(starts))


def test_recursive_chunker_max_chunks():
    text = "Word " * 100
    chunker = RecursiveChunker(_cfg(common={"chunk_size": 20, "chunk_overlap": 0, "max_chunks": 3}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 3


def test_recursive_chunker_empty_text():
    chunker = RecursiveChunker(_cfg())
    chunks = chunker.chunk("")
    assert len(chunks) <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_recursive_chunker.py -v`
Expected: `test_recursive_chunker_duplicate_content` likely FAILS — `text.find()` returns wrong offset for duplicates

- [ ] **Step 3: Fix RecursiveChunker offset tracking**

Replace `textchunker/chunkers/recursive.py`:

```python
from __future__ import annotations

from typing import List

from loguru import logger

from ..exceptions import ChunkerError
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..utils import count_tokens


def _split_by_separators(text: str, seps: List[str], size: int) -> List[str]:
    """Recursively split text by separator priority until chunks fit within size."""
    if count_tokens(text) <= size or not seps:
        return [text]
    sep = seps[0]
    parts = text.split(sep) if sep else list(text)
    chunks: List[str] = []
    buf = ""
    for i, p in enumerate(parts):
        piece = buf + (p + sep if i < len(parts) - 1 else p)
        if count_tokens(piece) <= size:
            buf = piece
        else:
            if buf:
                chunks.append(buf)
            if count_tokens(p) > size:
                chunks.extend(_split_by_separators(p, seps[1:], size))
                buf = ""
            else:
                buf = p + (sep if i < len(parts) - 1 else "")
    if buf:
        chunks.append(buf)
    out: List[str] = []
    for ch in chunks:
        if count_tokens(ch) > size and len(seps) > 1:
            out.extend(_split_by_separators(ch, seps[1:], size))
        else:
            out.append(ch)
    return out


def _build_offset_map(text: str, parts: List[str]) -> List[int]:
    """Build start offsets for each part using cumulative cursor.

    This is safe for duplicate parts because the cursor only moves forward.
    """
    offsets: List[int] = []
    cursor = 0
    for part in parts:
        idx = text.find(part, cursor)
        if idx == -1:
            idx = cursor
        offsets.append(idx)
        cursor = idx + len(part)
    return offsets


@register("recursive")
class RecursiveChunker(BaseChunker):
    """Recursive chunker: splits by paragraph > line > word > character priority."""

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        r = self.cfg.recursive
        size = int(c.get("chunk_size", 512))
        overlap = int(c.get("chunk_overlap", 80))
        seps = list(r.get("separators", ["\n\n", "\n", " ", ""]))
        max_chunks = c.get("max_chunks")

        if not text.strip():
            return []

        parts = _split_by_separators(text, seps, size)
        offsets = _build_offset_map(text, parts)

        chunks: List[Chunk] = []
        for i, part in enumerate(parts):
            start = offsets[i]
            end = start + len(part)
            chunks.append(Chunk(
                id=len(chunks), text=part, start=start, end=end,
                meta={"strategy": "recursive"},
            ))
            if max_chunks and len(chunks) >= max_chunks:
                break

        # Apply overlap: prepend tail of previous chunk
        if overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev, cur = chunks[i - 1], chunks[i]
                head_start = max(cur.start - overlap, prev.start)
                head = text[head_start:cur.start]
                cur.text = head + cur.text
                cur.start = head_start

        logger.info("Recursive chunking: {} chunks, size={}, overlap={}", len(chunks), size, overlap)
        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_recursive_chunker.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add textchunker/chunkers/recursive.py tests/test_recursive_chunker.py
git commit -m "fix: RecursiveChunker uses cumulative offset map, fixing duplicate content bug"
```

---

### Task 9: Fix StructureChunker — Offset Tracking

**Files:**
- Modify: `textchunker/chunkers/structure.py`
- Create: `tests/test_structure_chunker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_structure_chunker.py`:

```python
import sys
import types
import pytest

# Stub bs4 if not installed
if "bs4" not in sys.modules:
    stub = types.ModuleType("bs4")
    stub.BeautifulSoup = None
    sys.modules["bs4"] = stub

from textchunker.types import StrategyConfig
from textchunker.chunkers.structure import StructureChunker


def _cfg(**overrides):
    common = {"chunk_size": 100, "chunk_overlap": 0}
    structure = {"prefer": "md", "sub_split": True}
    common.update(overrides.get("common", {}))
    structure.update(overrides.get("structure", {}))
    return StrategyConfig(
        name="structure", common=common, structure=structure,
        semantic={}, recursive={}, fixed={}, llm={},
    )


def test_structure_chunker_markdown_sections():
    text = "# Title 1\nContent for section one.\n\n# Title 2\nContent for section two."
    chunker = StructureChunker(_cfg(common={"chunk_size": 500, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 2
    assert "Title 1" in chunks[0].meta.get("title", "")
    assert "Title 2" in chunks[1].meta.get("title", "")


def test_structure_chunker_no_headers():
    text = "Just plain text without any headers at all."
    chunker = StructureChunker(_cfg())
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert chunks[0].meta["title"] == "Document"


def test_structure_chunker_sub_split():
    long_section = "# Big Section\n" + "Word " * 200
    chunker = StructureChunker(_cfg(common={"chunk_size": 50, "chunk_overlap": 0}))
    chunks = chunker.chunk(long_section)
    assert len(chunks) > 1
    for c in chunks:
        assert c.meta["title"] == "Big Section"


def test_structure_chunker_offset_correctness():
    text = "# A\nShort.\n\n# B\nAlso short."
    chunker = StructureChunker(_cfg(common={"chunk_size": 500, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    for c in chunks:
        assert c.start >= 0
        assert c.end <= len(text)


def test_structure_chunker_max_chunks():
    text = "# S1\nA.\n\n# S2\nB.\n\n# S3\nC.\n\n# S4\nD."
    chunker = StructureChunker(_cfg(common={"chunk_size": 500, "chunk_overlap": 0, "max_chunks": 2}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_structure_chunker.py -v`
Expected: May fail on offset correctness if sub-split has duplicate text

- [ ] **Step 3: Fix StructureChunker offset tracking**

Replace `textchunker/chunkers/structure.py`:

```python
from __future__ import annotations

import re
from typing import List, Tuple

from loguru import logger

from ..exceptions import ChunkerError
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..utils import count_tokens
from .recursive import _split_by_separators, _build_offset_map


@register("structure")
class StructureChunker(BaseChunker):
    """Structure-based chunker: splits by document headings (Markdown/HTML)."""

    def _md_sections(self, text: str) -> List[Tuple[str, int, int]]:
        lines = text.splitlines(keepends=True)
        heads: List[int] = []
        for i, ln in enumerate(lines):
            if re.match(r"^\s*#{1,6}\s+", ln):
                heads.append(i)
        heads.append(len(lines))
        spans: List[Tuple[str, int, int]] = []
        for i in range(len(heads) - 1):
            beg_line = heads[i]
            end_line = heads[i + 1]
            beg = sum(len(l) for l in lines[:beg_line])
            end = sum(len(l) for l in lines[:end_line])
            title = re.sub(r"^#+\s+", "", lines[beg_line]).strip()
            spans.append((title, beg, end))
        if not spans:
            spans = [("Document", 0, len(text))]
        return spans

    def _html_sections(self, text: str) -> List[Tuple[str, int, int]]:
        return [("HTML", 0, len(text))]

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        st = self.cfg.structure
        size = int(c.get("chunk_size", 800))
        overlap = int(c.get("chunk_overlap", 100))
        prefer = st.get("prefer", "auto")
        sub_split = bool(st.get("sub_split", True))
        max_chunks = c.get("max_chunks")

        if prefer in {"md", "auto"}:
            spans = self._md_sections(text)
        elif prefer == "html":
            spans = self._html_sections(text)
        else:
            spans = self._md_sections(text)

        chunks: List[Chunk] = []
        for title, beg, end in spans:
            seg = text[beg:end]
            if count_tokens(seg) <= size or not sub_split:
                chunks.append(Chunk(
                    id=len(chunks), text=seg, start=beg, end=end,
                    meta={"strategy": "structure", "title": title},
                ))
            else:
                parts = _split_by_separators(seg, ["\n\n", "\n", " ", ""], size)
                offsets = _build_offset_map(seg, parts)
                for j, p in enumerate(parts):
                    s = beg + offsets[j]
                    e = s + len(p)
                    chunks.append(Chunk(
                        id=len(chunks), text=p, start=s, end=e,
                        meta={"strategy": "structure", "title": title},
                    ))
            if max_chunks and len(chunks) >= max_chunks:
                break

        # Apply overlap
        if overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev, cur = chunks[i - 1], chunks[i]
                head_start = max(cur.start - overlap, prev.start)
                head = text[head_start:cur.start]
                cur.text = head + cur.text
                cur.start = head_start

        logger.info("Structure chunking: {} chunks from {} sections", len(chunks), len(spans))
        return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_structure_chunker.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add textchunker/chunkers/structure.py tests/test_structure_chunker.py
git commit -m "fix: StructureChunker uses _build_offset_map for safe sub-split offset tracking"
```

---

### Task 10: DashScope Provider + LLM Provider Improvements

**Files:**
- Create: `textchunker/providers/dashscope_provider.py`
- Modify: `textchunker/providers/llm.py`
- Modify: `textchunker/providers/__init__.py`
- Modify: `textchunker/chunkers/llm_based.py`
- Create: `tests/test_providers.py`
- Create: `tests/test_llm_chunker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_providers.py`:

```python
import json
import pytest
from unittest.mock import MagicMock, patch

from textchunker.providers.dashscope_provider import DashScopeProvider
from textchunker.providers.llm import HFProvider, BaseLLMProvider


def test_dashscope_provider_parses_response():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps([
        {"title": "Intro", "start": 0, "end": 50},
        {"title": "Body", "start": 50, "end": 200},
    ])

    with patch("textchunker.providers.dashscope_provider.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        provider = DashScopeProvider.__new__(DashScopeProvider)
        provider.model = "qwen-max"
        provider.client = mock_client

        spans = provider.propose_spans("test text", "system prompt", 8000)
        assert len(spans) == 2
        assert spans[0] == (0, 50, "Intro")
        assert spans[1] == (50, 200, "Body")


def test_dashscope_provider_handles_invalid_json():
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "not valid json at all"

    with patch("textchunker.providers.dashscope_provider.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        provider = DashScopeProvider.__new__(DashScopeProvider)
        provider.model = "qwen-max"
        provider.client = mock_client

        spans = provider.propose_spans("text", "prompt", 1000)
        assert spans == []


def test_base_provider_is_abstract():
    provider = BaseLLMProvider()
    with pytest.raises(NotImplementedError):
        provider.propose_spans("text", "prompt", 100)
```

Create `tests/test_llm_chunker.py`:

```python
import pytest
from unittest.mock import MagicMock, patch

from textchunker.types import StrategyConfig
from textchunker.chunkers.llm_based import LLMChunker


def _cfg(**overrides):
    llm = {
        "provider": "dashscope",
        "llm_model": "qwen-max",
        "max_chars_per_call": 8000,
        "system_prompt": "Segment the text.",
    }
    llm.update(overrides)
    return StrategyConfig(
        name="llm", llm=llm, common={},
        semantic={}, recursive={}, fixed={}, structure={},
    )


def test_llm_chunker_with_mock_provider():
    cfg = _cfg()
    with patch("textchunker.chunkers.llm_based.DashScopeProvider") as MockProvider:
        mock_instance = MagicMock()
        mock_instance.propose_spans.return_value = [
            (0, 10, "Part 1"),
            (10, 20, "Part 2"),
        ]
        MockProvider.return_value = mock_instance

        chunker = LLMChunker(cfg)
        chunks = chunker.chunk("Hello World Testing Text Foo Bar")
        assert len(chunks) == 2
        assert chunks[0].meta["strategy"] == "llm"


def test_llm_chunker_fallback_on_empty_spans():
    cfg = _cfg()
    with patch("textchunker.chunkers.llm_based.DashScopeProvider") as MockProvider:
        mock_instance = MagicMock()
        mock_instance.propose_spans.return_value = []
        MockProvider.return_value = mock_instance

        chunker = LLMChunker(cfg)
        text = "Entire document as one chunk."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text


def test_llm_chunker_max_chunks():
    cfg = _cfg()
    cfg.common["max_chunks"] = 1
    with patch("textchunker.chunkers.llm_based.DashScopeProvider") as MockProvider:
        mock_instance = MagicMock()
        mock_instance.propose_spans.return_value = [
            (0, 5, "A"), (5, 10, "B"), (10, 15, "C"),
        ]
        MockProvider.return_value = mock_instance

        chunker = LLMChunker(cfg)
        chunks = chunker.chunk("Hello World Testing")
        assert len(chunks) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_providers.py tests/test_llm_chunker.py -v`
Expected: FAIL — modules don't exist yet

- [ ] **Step 3: Create DashScope provider**

Create `textchunker/providers/dashscope_provider.py`:

```python
from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Tuple

from loguru import logger

from ..exceptions import ProviderError
from .llm import BaseLLMProvider


class DashScopeProvider(BaseLLMProvider):
    """DashScope LLM provider using OpenAI-compatible SDK."""

    def __init__(self, model: str = "qwen-max", api_key: str = "", api_base_url: str = "") -> None:
        from openai import OpenAI

        self.model = model
        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base = api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not key:
            logger.warning("DASHSCOPE_API_KEY not set — DashScope provider will fail on API calls")

        self.client = OpenAI(api_key=key, base_url=base)

    def propose_spans(
        self, text: str, system_prompt: str, max_chars: int
    ) -> List[Tuple[int, int, Optional[str]]]:
        prompt = text[:max_chars]
        msg = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"请基于如下文本生成 JSON 切分：\n\n{prompt}\n\n"
                f'返回格式示例：[{{"title": "节名或null", "start": 0, "end": 100}}, ...]'
            )},
        ]
        try:
            logger.debug("DashScope API call: model={}, prompt_len={}", self.model, len(prompt))
            resp = self.client.chat.completions.create(
                model=self.model, messages=msg, temperature=0,
            )
            content = resp.choices[0].message.content or "[]"
        except Exception as e:
            logger.warning("DashScope API call failed: {}", e)
            return []

        return self._parse_spans(content)

    @staticmethod
    def _parse_spans(content: str) -> List[Tuple[int, int, Optional[str]]]:
        """Parse JSON spans from LLM response with robust extraction."""
        # Try direct parse first
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [(int(d.get("start", 0)), int(d.get("end", 0)), d.get("title")) for d in data]
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting JSON array with regex
        match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', content, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [(int(d.get("start", 0)), int(d.get("end", 0)), d.get("title")) for d in data]
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning("Failed to parse spans from LLM response: {}", content[:200])
        return []
```

- [ ] **Step 4: Update HFProvider in `providers/llm.py`**

Replace `textchunker/providers/llm.py`:

```python
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..exceptions import ProviderError


class BaseLLMProvider:
    """Base class for LLM providers."""

    def propose_spans(
        self, text: str, system_prompt: str, max_chars: int
    ) -> List[Tuple[int, int, Optional[str]]]:
        raise NotImplementedError


class HFProvider(BaseLLMProvider):
    """HuggingFace transformers text generation provider."""

    def __init__(self, model: str) -> None:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline  # type: ignore

        logger.info("Loading HF model: {}", model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)
        self.pipe = pipeline("text-generation", model=self._model, tokenizer=self.tokenizer)

    def propose_spans(
        self, text: str, system_prompt: str, max_chars: int
    ) -> List[Tuple[int, int, Optional[str]]]:
        prompt = system_prompt.strip() + "\n\n" + text[:max_chars]
        try:
            out = self.pipe(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        except Exception as e:
            logger.warning("HF inference failed: {}", e)
            return []

        # Robust JSON extraction
        match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', out, re.DOTALL)
        if not match:
            logger.warning("No JSON array found in HF response")
            return []

        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from HF response")
            return []

        spans: List[Tuple[int, int, Optional[str]]] = []
        for item in data:
            spans.append((int(item.get("start", 0)), int(item.get("end", 0)), item.get("title")))
        return spans
```

- [ ] **Step 5: Update `providers/__init__.py`**

Replace `textchunker/providers/__init__.py`:

```python
from .llm import BaseLLMProvider, HFProvider
from .dashscope_provider import DashScopeProvider

__all__ = ["BaseLLMProvider", "DashScopeProvider", "HFProvider"]
```

- [ ] **Step 6: Update LLMChunker to use DashScope**

Replace `textchunker/chunkers/llm_based.py`:

```python
from __future__ import annotations

from typing import List

from loguru import logger

from ..exceptions import ChunkerError, ProviderError
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..providers import DashScopeProvider, HFProvider


@register("llm")
class LLMChunker(BaseChunker):
    """LLM-based chunker: model returns [start, end, title] span proposals."""

    def __init__(self, cfg):
        super().__init__(cfg)
        p = cfg.llm.get("provider", "dashscope").lower()
        if p in ("dashscope", "openai"):
            model = cfg.llm.get("llm_model", cfg.llm.get("openai_model", "qwen-max"))
            api_key = cfg.llm.get("api_key", "")
            api_base = cfg.llm.get("api_base_url", "")
            self.provider = DashScopeProvider(model=model, api_key=api_key, api_base_url=api_base)
        elif p == "hf":
            self.provider = HFProvider(cfg.llm.get("hf_model", "Qwen/Qwen2.5-7B-Instruct"))
        else:
            raise ConfigError(f"Unknown provider: {p}")

    def chunk(self, text: str) -> List[Chunk]:
        l = self.cfg.llm
        max_chars = int(l.get("max_chars_per_call", 8000))
        sys_prompt = l.get("system_prompt", "Segment the document into meaningful spans. Return JSON spans.")
        max_chunks = self.cfg.common.get("max_chunks")

        spans = self.provider.propose_spans(text, system_prompt=sys_prompt, max_chars=max_chars)
        if not spans:
            logger.warning("Provider returned no spans, falling back to single chunk")
            return [Chunk(id=0, text=text, start=0, end=len(text), meta={"strategy": "llm"})]

        chunks: List[Chunk] = []
        for s, e, title in spans:
            s = max(0, min(s, len(text)))
            e = max(s, min(e, len(text)))
            piece = text[s:e]
            chunks.append(Chunk(
                id=len(chunks), text=piece, start=s, end=e,
                meta={"strategy": "llm", "title": title},
            ))
            if max_chunks and len(chunks) >= max_chunks:
                break

        logger.info("LLM chunking: {} chunks from provider spans", len(chunks))
        return chunks
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_providers.py tests/test_llm_chunker.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add textchunker/providers/ textchunker/chunkers/llm_based.py tests/test_providers.py tests/test_llm_chunker.py
git commit -m "feat: add DashScopeProvider with qwen-max, improve JSON parsing, refactor LLMChunker"
```

---

### Task 11: Embedding Providers

**Files:**
- Create: `textchunker/providers/embeddings.py`

- [ ] **Step 1: Create embedding providers**

Create `textchunker/providers/embeddings.py`:

```python
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from loguru import logger


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def encode(self, sentences: List[str]) -> np.ndarray:
        raise NotImplementedError


class DashScopeEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding via DashScope API (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model: str = "text-embedding-v3",
        api_key: str = "",
        api_base_url: str = "",
    ) -> None:
        from openai import OpenAI

        self.model = model
        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base = api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=key, base_url=base)

    def encode(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.array([])
        resp = self.client.embeddings.create(model=self.model, input=sentences)
        vecs = [item.embedding for item in resp.data]
        arr = np.array(vecs, dtype=float)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Embedding via local sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("Loading sentence-transformers model: {}", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(sentences, normalize_embeddings=True)
```

- [ ] **Step 2: Update `providers/__init__.py`**

Add to `textchunker/providers/__init__.py`:

```python
from .llm import BaseLLMProvider, HFProvider
from .dashscope_provider import DashScopeProvider
from .embeddings import BaseEmbeddingProvider, DashScopeEmbeddingProvider, SentenceTransformerProvider

__all__ = [
    "BaseLLMProvider", "DashScopeProvider", "HFProvider",
    "BaseEmbeddingProvider", "DashScopeEmbeddingProvider", "SentenceTransformerProvider",
]
```

- [ ] **Step 3: Verify import**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -c "from textchunker.providers import DashScopeEmbeddingProvider, SentenceTransformerProvider; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add textchunker/providers/
git commit -m "feat: add DashScopeEmbeddingProvider (text-embedding-v3) and SentenceTransformerProvider"
```

---

### Task 12: Update Config, CLI & Init Files

**Files:**
- Modify: `textchunker/config.py`
- Modify: `textchunker/cli.py`
- Modify: `textchunker/__init__.py`
- Modify: `textchunker/chunkers/__init__.py`
- Create: `tests/test_config.py`
- Create: `tests/test_registry.py`
- Create: `tests/test_readers.py`

- [ ] **Step 1: Write config tests**

Create `tests/test_config.py`:

```python
import pytest
from pathlib import Path

from textchunker.config import load_yaml, to_project_config, build_argparser, merge_cli


def test_load_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("strategy:\n  name: fixed\n  common:\n    chunk_size: 100\n")
    data = load_yaml(str(yaml_file))
    assert data["strategy"]["name"] == "fixed"
    assert data["strategy"]["common"]["chunk_size"] == 100


def test_to_project_config():
    d = {
        "project": {"name": "test"},
        "io": {"input_path": "in.txt"},
        "strategy": {
            "name": "recursive",
            "common": {"chunk_size": 256},
        },
    }
    cfg = to_project_config(d)
    assert cfg.strategy.name == "recursive"
    assert cfg.strategy.common["chunk_size"] == 256


def test_to_project_config_defaults():
    cfg = to_project_config({})
    assert cfg.strategy.name == "recursive"
    assert cfg.project == {}


def test_merge_cli_strategy_override():
    cfg = to_project_config({"strategy": {"name": "recursive"}})
    args = build_argparser().parse_args(["--strategy", "semantic"])
    merged = merge_cli(cfg, args)
    assert merged.strategy.name == "semantic"


def test_merge_cli_preserves_unset():
    cfg = to_project_config({"strategy": {"name": "fixed"}})
    args = build_argparser().parse_args([])
    merged = merge_cli(cfg, args)
    assert merged.strategy.name == "fixed"
```

Create `tests/test_registry.py`:

```python
import pytest
from textchunker.registry import register, get, available, _REGISTRY


def test_register_and_get():
    @register("test_strategy_xyz")
    class TestChunker:
        pass
    assert get("test_strategy_xyz") is TestChunker
    del _REGISTRY["test_strategy_xyz"]


def test_get_unknown():
    with pytest.raises(KeyError, match="Unknown chunker"):
        get("nonexistent_strategy_abc")


def test_available():
    result = available()
    assert isinstance(result, dict)
    # Should have at least the 5 built-in strategies
    assert "fixed" in result or "recursive" in result


def test_register_case_insensitive():
    @register("CamelCase_Test")
    class CamelChunker:
        pass
    assert get("camelcase_test") is CamelChunker
    del _REGISTRY["camelcase_test"]
```

Create `tests/test_readers.py`:

```python
import pytest
from pathlib import Path

from textchunker.readers import load_inputs


def test_read_text_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world content", encoding="utf-8")
    docs = list(load_inputs(str(f)))
    assert len(docs) == 1
    assert docs[0].text == "Hello world content"
    assert docs[0].path == str(f)


def test_read_markdown_file(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("# Title\nContent here", encoding="utf-8")
    docs = list(load_inputs(str(f)))
    assert len(docs) == 1
    assert "Title" in docs[0].text


def test_read_directory(tmp_path):
    (tmp_path / "a.txt").write_text("File A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("File B", encoding="utf-8")
    docs = list(load_inputs(str(tmp_path)))
    assert len(docs) == 2
    texts = {d.text for d in docs}
    assert "File A" in texts
    assert "File B" in texts


def test_read_unknown_extension(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("col1,col2\na,b", encoding="utf-8")
    docs = list(load_inputs(str(f)))
    assert len(docs) == 1
    assert "col1" in docs[0].text
```

- [ ] **Step 2: Update config.py with new CLI args**

Add these arguments to `build_argparser()` in `textchunker/config.py`, after the existing `--hf-model` argument:

```python
    # Logging
    p.add_argument("--log-level", type=str, default=None,
                   help="Log level: TRACE/DEBUG/INFO/WARNING/ERROR")
    # Stats
    p.add_argument("--stats", action="store_true", help="Enable statistics collection")
    p.add_argument("--stats-dir", type=str, default=None, help="Directory for stats output")
    # DashScope
    p.add_argument("--llm-model", type=str, default=None, help="LLM model name (default: qwen-max)")
    p.add_argument("--embedding-model", type=str, default=None, help="Embedding model (default: text-embedding-v3)")
    # Chunk params
    p.add_argument("--chunk-size", type=int, default=None, help="Override chunk_size")
    p.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk_overlap")
```

Add to `merge_cli()` in `textchunker/config.py`, before the `return cfg` line:

```python
    if getattr(args, 'chunk_size', None) is not None:
        cfg.strategy.common["chunk_size"] = args.chunk_size
    if getattr(args, 'chunk_overlap', None) is not None:
        cfg.strategy.common["chunk_overlap"] = args.chunk_overlap
    if getattr(args, 'llm_model', None):
        cfg.strategy.llm["llm_model"] = args.llm_model
    if getattr(args, 'embedding_model', None):
        cfg.strategy.semantic["embedding_model"] = args.embedding_model
```

- [ ] **Step 3: Update cli.py**

Replace `textchunker/cli.py`:

```python
from __future__ import annotations

import os
import time

from loguru import logger

from .config import load_yaml, to_project_config, build_argparser, merge_cli
from .settings import Settings
from .logging import setup_logging
from .stats import StatsCollector
from .readers import load_inputs
from .factory import create_chunker
from .export import save_jsonl, save_txt_dir
from .visualization import show_chunks
from .utils import count_tokens
from .chunkers import *  # noqa: F401  # trigger registration


def main() -> None:
    """CLI entry point: read -> chunk -> visualize -> export."""
    parser = build_argparser()
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    cfg = to_project_config(raw_cfg)
    cfg = merge_cli(cfg, args)

    # Initialize settings and logging
    settings = Settings.from_env_and_yaml(raw_cfg)
    if getattr(args, 'log_level', None):
        settings.log_level = args.log_level
    setup_logging(settings)

    # Initialize stats
    stats = StatsCollector.instance()
    stats.reset()
    stats_enabled = getattr(args, 'stats', False) or settings.stats_enabled

    input_path = cfg.io.get("input_path")
    output_path = cfg.io.get("output_path")
    visualize = bool(cfg.project.get("visualize", False))
    export_format = cfg.project.get("export_format", "jsonl")

    logger.info("Strategy = {}", cfg.strategy.name)
    chunker = create_chunker(cfg.strategy)

    for doc in load_inputs(input_path):
        logger.info("Processing: {}", doc.path)
        t0 = time.perf_counter()
        chunks = chunker.chunk(doc.text)
        elapsed = time.perf_counter() - t0

        if stats_enabled:
            doc_tokens = count_tokens(doc.text)
            stats.record_document(
                char_count=len(doc.text), token_count=doc_tokens,
                chunk_count=len(chunks), processing_time=elapsed,
            )
            stats.record_chunks(
                chunk_sizes=[len(c.text) for c in chunks],
                chunk_tokens=[count_tokens(c.text) for c in chunks],
                max_size=int(cfg.strategy.common.get("chunk_size", 512)),
            )

        if visualize:
            show_chunks(chunks, title=os.path.basename(doc.path))

        if output_path:
            if export_format in {"jsonl", "both"}:
                save_jsonl(output_path, chunks, source=doc.path)
                logger.info("Written JSONL to {}", output_path)
            if export_format in {"txt_dir", "both"}:
                outdir = os.path.splitext(output_path)[0] + "_parts"
                save_txt_dir(outdir, chunks, prefix=os.path.basename(doc.path))
                logger.info("Written TXT parts to {}", outdir)

    if stats_enabled:
        from rich.console import Console
        summary = stats.summary()
        console = Console()
        console.print("\n[bold]Pipeline Statistics[/bold]")
        for k, v in summary.items():
            if isinstance(v, float):
                console.print(f"  {k}: {v:.3f}")
            else:
                console.print(f"  {k}: {v}")

        stats_dir = getattr(args, 'stats_dir', None) or settings.stats_dir
        stats.save_run(
            os.path.join(stats_dir, f"{cfg.strategy.name}_run.json"),
            metadata={"strategy": cfg.strategy.name},
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update `__init__.py` files**

Replace `textchunker/__init__.py`:

```python
__all__ = [
    "cli", "factory", "registry", "types", "config",
    "settings", "logging", "stats", "exceptions",
    "readers", "export", "visualization", "utils",
]
```

Replace `textchunker/chunkers/__init__.py`:

```python
from . import fixed, semantic, recursive, structure, llm_based  # noqa: F401

__all__ = ["fixed", "semantic", "recursive", "structure", "llm_based"]
```

- [ ] **Step 5: Run all tests**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/test_config.py tests/test_registry.py tests/test_readers.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add textchunker/config.py textchunker/cli.py textchunker/__init__.py textchunker/chunkers/__init__.py tests/test_config.py tests/test_registry.py tests/test_readers.py
git commit -m "feat: integrate Settings, logging, and stats into CLI pipeline; add config/registry/reader tests"
```

---

### Task 13: Update YAML Configs

**Files:**
- Modify: `configs/default.yaml`
- Modify: `configs/semantic.yaml`
- Modify: `configs/llm_based.yaml`

- [ ] **Step 1: Update default.yaml**

Replace `configs/default.yaml`:

```yaml
# Global defaults
project:
  name: text-chunker-demo
  seed: 42
  visualize: true
  export_format: jsonl

settings:
  api_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  llm_model: "qwen-max"
  embedding_model: "text-embedding-v3"

logging:
  level: INFO
  json: false
  log_dir: logs/
  rotation: "10 MB"
  retention: 5

io:
  input_path: ./sample.txt
  output_path: ./out/chunks.jsonl

strategy:
  name: recursive
  common:
    chunk_size: 512
    chunk_overlap: 80
    max_chunks: null
  recursive:
    separators: ["\n\n", "\n", " ", ""]
  fixed:
    use_tokens: true
  semantic:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    min_similarity: 0.62
    sentence_window: 1
  structure:
    prefer: md
    sub_split: true
  llm:
    provider: dashscope
    llm_model: qwen-max
    hf_model: Qwen/Qwen2.5-7B-Instruct
    max_chars_per_call: 8000
    system_prompt: |
      你是一名文本结构工程师。请将输入文本按主题/标题/时间线合理分块，
      产出 JSON 列表：[{ "title": str|null, "start": int, "end": int }]
```

- [ ] **Step 2: Update semantic.yaml**

Replace `configs/semantic.yaml`:

```yaml
strategy:
  name: semantic
  common:
    chunk_size: 600
    chunk_overlap: 100
  semantic:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    embedding_model: text-embedding-v3
    min_similarity: 0.65
```

- [ ] **Step 3: Update llm_based.yaml**

Replace `configs/llm_based.yaml`:

```yaml
strategy:
  name: llm
  llm:
    provider: dashscope
    llm_model: qwen-max
    max_chars_per_call: 6000
```

- [ ] **Step 4: Commit**

```bash
git add configs/
git commit -m "feat: update YAML configs for DashScope API with qwen-max and text-embedding-v3"
```

---

### Task 14: Update setup.cfg & requirements.txt

**Files:**
- Modify: `setup.cfg`
- Modify: `requirements.txt`

- [ ] **Step 1: Update setup.cfg**

Replace `setup.cfg`:

```ini
[metadata]
name = textchunker
version = 0.2.0
description = Pluggable text chunking factory for RAG with DashScope/LLM support
author = Your Name
license = MIT
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/1998x-stack/text-chunker
classifiers =
    Programming Language :: Python :: 3
    License :: OSI Approved :: MIT License
    Topic :: Text Processing :: General

[options]
packages = find:
python_requires = >=3.9
install_requires =
    pyyaml>=6.0.1
    loguru>=0.7.2
    rich>=13.7.1

[options.extras_require]
semantic =
    sentence-transformers>=3.0.1
    numpy>=1.26.0
token =
    tiktoken>=0.7.0
structure =
    beautifulsoup4>=4.12.3
    pypdf>=5.0.0
    markdown-it-py>=3.0.0
llm =
    openai>=1.46.0
    transformers>=4.44.2
    torch>=2.2.0
experiments =
    datasets>=2.20.0
dev =
    pytest>=8.0.0
    pytest-cov>=5.0.0
    flake8>=7.0.0
    mypy>=1.8.0
    isort>=5.13.0
    black>=24.0.0
all =
    %(semantic)s
    %(token)s
    %(structure)s
    %(llm)s
    %(experiments)s

[options.entry_points]
console_scripts =
    textchunker = textchunker.cli:main

[flake8]
max-line-length = 100
extend-ignore = E203,W503

[mypy]
python_version = 3.9
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true
```

- [ ] **Step 2: Update requirements.txt**

Replace `requirements.txt`:

```
# Core
pyyaml>=6.0.1
loguru>=0.7.2
rich>=13.7.1

# Tokenization
tiktoken>=0.7.0

# Semantic chunking
sentence-transformers>=3.0.1
numpy>=1.26.0

# Structure-based parsing
beautifulsoup4>=4.12.3
pypdf>=5.0.0
markdown-it-py>=3.0.0

# LLM providers (DashScope-compatible)
openai>=1.46.0
transformers>=4.44.2
torch>=2.2.0

# Experiment utilities
datasets>=2.20.0

# Dev tools
pytest>=8.0.0
pytest-cov>=5.0.0
flake8>=7.0.0
mypy>=1.8.0
isort>=5.13.0
black>=24.0.0
```

- [ ] **Step 3: Commit**

```bash
git add setup.cfg requirements.txt
git commit -m "chore: bump to v0.2.0, add dev deps, DashScope-compatible setup"
```

---

### Task 15: Enhanced Experiments

**Files:**
- Modify: `textchunker/experiments/semantic_ablation.py` (add progress bar + CSV)
- Create: `textchunker/experiments/recursive_ablation.py`
- Create: `textchunker/experiments/strategy_comparison.py`
- Modify: `textchunker/experiments/__init__.py`

- [ ] **Step 1: Add progress bar to semantic_ablation.py**

In `textchunker/experiments/semantic_ablation.py`, add `from rich.progress import Progress` to imports, then change the loop in `run_semantic_ablation` from:

```python
    results: List[AblationMetrics] = []
    for scenario in scenarios:
```

To:

```python
    results: List[AblationMetrics] = []
    with Progress() as progress:
        task = progress.add_task("Running scenarios...", total=len(scenarios))
        for scenario in scenarios:
```

And after `results.append(metrics)`, add:

```python
            progress.advance(task)
```

Also add CSV export after the JSON export block:

```python
    if args.save_json:
        # ... existing JSON save code ...

        # Also save CSV
        csv_path = path.with_suffix(".csv")
        import csv
        with csv_path.open("w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=list(results[0].to_dict().keys()))
            writer.writeheader()
            for r in results:
                flat = r.to_dict()
                flat.update(flat.pop("scenario"))
                writer.writerow(flat)
        logger.info("Saved CSV to %s", csv_path)
```

- [ ] **Step 2: Create recursive_ablation.py**

Create `textchunker/experiments/recursive_ablation.py`:

```python
from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ..chunkers import *  # noqa: F401,F403
from ..config import load_yaml, to_project_config
from ..factory import create_chunker
from ..utils import count_tokens
from .semantic_ablation import (
    AblationMetrics,
    Scenario,
    load_hf_texts,
    summarize_metrics,
)


def evaluate_recursive_scenario(
    base_chunker,
    scenario: Scenario,
    texts: List[str],
) -> AblationMetrics:
    original_common = copy.deepcopy(base_chunker.cfg.common)
    try:
        base_chunker.cfg.common.update({
            "chunk_size": scenario.chunk_size,
            "chunk_overlap": scenario.chunk_overlap,
        })

        doc_token_counts: List[int] = []
        doc_char_counts: List[int] = []
        chunk_token_counts: List[int] = []
        chunk_char_counts: List[int] = []
        chunk_counts: List[int] = []
        boundary_hits = 0

        for text in texts:
            doc_token_counts.append(count_tokens(text))
            doc_char_counts.append(len(text))
            chunks = base_chunker.chunk(text)
            chunk_counts.append(len(chunks))
            for ch in chunks:
                ctoks = count_tokens(ch.text)
                chunk_token_counts.append(ctoks)
                chunk_char_counts.append(len(ch.text))
                if ctoks >= scenario.chunk_size:
                    boundary_hits += 1

        return summarize_metrics(
            scenario, doc_token_counts, doc_char_counts,
            chunk_token_counts, chunk_char_counts, chunk_counts, boundary_hits,
        )
    finally:
        base_chunker.cfg.common = original_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive chunking ablation study")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-docs", type=int, default=64)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 800])
    parser.add_argument("--chunk-overlaps", type=int, nargs="+", default=[0, 50, 100])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def run_recursive_ablation(args: argparse.Namespace | None = None) -> List[AblationMetrics]:
    if args is None:
        args = parse_args()

    cfg = to_project_config(load_yaml(args.config)).strategy
    cfg.name = "recursive"

    texts = load_hf_texts(
        dataset=args.dataset, subset=args.subset, split=args.split,
        text_field=args.text_field, max_docs=args.max_docs,
        min_doc_chars=args.min_doc_chars, seed=args.seed, cache_dir=args.cache_dir,
    )

    chunker = create_chunker(cfg)

    scenarios = [
        Scenario(cs, ov, 0.0, 0)
        for cs, ov in itertools.product(args.chunk_sizes, args.chunk_overlaps)
    ]

    logger.info("Running {} recursive scenarios", len(scenarios))
    results: List[AblationMetrics] = []

    with Progress() as progress:
        task = progress.add_task("Running recursive ablation...", total=len(scenarios))
        for scenario in scenarios:
            metrics = evaluate_recursive_scenario(chunker, scenario, texts)
            results.append(metrics)
            progress.advance(task)

    console = Console()
    table = Table(title="Recursive Ablation Results", show_header=True, header_style="bold magenta")
    table.add_column("chunk_size", justify="right")
    table.add_column("overlap", justify="right")
    table.add_column("chunks/doc", justify="right")
    table.add_column("avg_chunk_tok", justify="right")
    table.add_column("p95_tok", justify="right")
    table.add_column("redundancy", justify="right")
    table.add_column("coverage", justify="right")
    for r in results:
        table.add_row(
            str(r.scenario.chunk_size), str(r.scenario.chunk_overlap),
            f"{r.avg_chunks_per_doc:.2f}", f"{r.avg_chunk_tokens:.1f}",
            f"{r.p95_chunk_tokens:.1f}", f"{r.redundancy_ratio*100:.1f}%",
            f"{r.coverage_ratio:.2f}x",
        )
    console.print(table)

    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=["chunk_size", "chunk_overlap", "avg_chunks_per_doc", "avg_chunk_tokens", "p95_chunk_tokens", "redundancy_ratio", "coverage_ratio"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "chunk_size": r.scenario.chunk_size, "chunk_overlap": r.scenario.chunk_overlap,
                    "avg_chunks_per_doc": f"{r.avg_chunks_per_doc:.2f}",
                    "avg_chunk_tokens": f"{r.avg_chunk_tokens:.1f}",
                    "p95_chunk_tokens": f"{r.p95_chunk_tokens:.1f}",
                    "redundancy_ratio": f"{r.redundancy_ratio:.4f}",
                    "coverage_ratio": f"{r.coverage_ratio:.4f}",
                })
        logger.info("Saved results to {} and {}", path, csv_path)

    return results


def main() -> None:
    run_recursive_ablation()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create strategy_comparison.py**

Create `textchunker/experiments/strategy_comparison.py`:

```python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from rich.console import Console
from rich.table import Table

from ..chunkers import *  # noqa: F401,F403
from ..config import load_yaml, to_project_config
from ..factory import create_chunker
from ..utils import count_tokens
from .semantic_ablation import load_hf_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-strategy comparison")
    parser.add_argument("--strategies", type=str, nargs="+", default=["fixed", "recursive"])
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-docs", type=int, default=32)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def run_strategy_comparison(args: argparse.Namespace | None = None) -> List[Dict[str, Any]]:
    if args is None:
        args = parse_args()

    base_cfg = to_project_config(load_yaml(args.config))

    texts = load_hf_texts(
        dataset=args.dataset, subset=args.subset, split=args.split,
        text_field=args.text_field, max_docs=args.max_docs,
        min_doc_chars=args.min_doc_chars, seed=args.seed, cache_dir=args.cache_dir,
    )

    results: List[Dict[str, Any]] = []

    for strategy_name in args.strategies:
        logger.info("Evaluating strategy: {}", strategy_name)
        cfg = base_cfg.strategy
        cfg.name = strategy_name

        try:
            chunker = create_chunker(cfg)
        except Exception as e:
            logger.warning("Failed to create chunker for {}: {}", strategy_name, e)
            continue

        total_chunks = 0
        total_chars = 0
        total_tokens = 0
        t0 = time.perf_counter()

        for text in texts:
            chunks = chunker.chunk(text)
            total_chunks += len(chunks)
            for c in chunks:
                total_chars += len(c.text)
                total_tokens += count_tokens(c.text)

        elapsed = time.perf_counter() - t0
        n_docs = len(texts)
        doc_chars = sum(len(t) for t in texts)
        doc_tokens = sum(count_tokens(t) for t in texts)

        results.append({
            "strategy": strategy_name,
            "total_chunks": total_chunks,
            "chunks_per_doc": total_chunks / n_docs if n_docs else 0,
            "avg_chunk_chars": total_chars / total_chunks if total_chunks else 0,
            "avg_chunk_tokens": total_tokens / total_chunks if total_chunks else 0,
            "redundancy_ratio": max(0, (total_tokens - doc_tokens) / doc_tokens) if doc_tokens else 0,
            "coverage_ratio": total_chars / doc_chars if doc_chars else 0,
            "time_seconds": elapsed,
        })

    # Display
    console = Console()
    table = Table(title="Strategy Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Strategy")
    table.add_column("Chunks", justify="right")
    table.add_column("Chunks/Doc", justify="right")
    table.add_column("Avg Chars", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Redundancy", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Time (s)", justify="right")

    for r in results:
        table.add_row(
            r["strategy"], str(r["total_chunks"]),
            f"{r['chunks_per_doc']:.1f}", f"{r['avg_chunk_chars']:.0f}",
            f"{r['avg_chunk_tokens']:.0f}", f"{r['redundancy_ratio']*100:.1f}%",
            f"{r['coverage_ratio']:.2f}x", f"{r['time_seconds']:.2f}",
        )
    console.print(table)

    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved comparison to {}", path)

    return results


def main() -> None:
    run_strategy_comparison()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update experiments/__init__.py**

Replace `textchunker/experiments/__init__.py`:

```python
"""Experiment utilities for benchmarking chunking strategies."""

from .semantic_ablation import Scenario, AblationMetrics, run_semantic_ablation  # noqa: F401
from .recursive_ablation import run_recursive_ablation  # noqa: F401
from .strategy_comparison import run_strategy_comparison  # noqa: F401
```

- [ ] **Step 5: Verify imports**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -c "from textchunker.experiments import run_recursive_ablation, run_strategy_comparison; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add textchunker/experiments/
git commit -m "feat: add recursive ablation, strategy comparison experiments, Rich progress + CSV export"
```

---

### Task 16: Bash Scripts & Makefile

**Files:**
- Create: `scripts/install.sh`
- Create: `scripts/test.sh`
- Create: `scripts/lint.sh`
- Create: `scripts/format.sh`
- Create: `scripts/benchmark.sh`
- Create: `scripts/experiment.sh`
- Create: `scripts/ci.sh`
- Create: `scripts/docker-build.sh`
- Create: `Makefile`

- [ ] **Step 1: Create all scripts**

Create `scripts/install.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."
pip install -e ".[all,dev]"
echo "Installation complete."
```

Create `scripts/test.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."
python -m pytest tests/ -v --cov=textchunker --cov-report=term-missing --cov-fail-under=60
```

Create `scripts/lint.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."
echo "=== flake8 ==="
python -m flake8 textchunker/ tests/ --max-line-length=100 --extend-ignore=E203,W503
echo "=== mypy ==="
python -m mypy textchunker/ --ignore-missing-imports --no-error-summary || true
echo "=== isort check ==="
python -m isort --check-only --diff textchunker/ tests/ || true
echo "Lint complete."
```

Create `scripts/format.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."
python -m isort textchunker/ tests/
python -m black textchunker/ tests/ --line-length 100
echo "Formatting complete."
```

Create `scripts/benchmark.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."

echo "=== Benchmarking all strategies ==="
for strategy in fixed recursive structure; do
    echo "--- $strategy ---"
    if [ -f sample.txt ]; then
        python -m textchunker.cli --config configs/default.yaml --strategy "$strategy" --input sample.txt --stats 2>&1 | tail -20
    else
        echo "No sample.txt found. Create one to run benchmarks."
    fi
done
echo "Benchmark complete."
```

Create `scripts/experiment.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."

echo "=== Running Recursive Ablation ==="
python -m textchunker.experiments.recursive_ablation \
    --max-docs 32 --chunk-sizes 256 512 800 --chunk-overlaps 0 50 100 \
    --save-json results/recursive_ablation.json

echo ""
echo "=== Running Strategy Comparison ==="
python -m textchunker.experiments.strategy_comparison \
    --strategies fixed recursive --max-docs 32 \
    --save-json results/strategy_comparison.json

echo "Experiments complete."
```

Create `scripts/ci.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."

echo "=== CI Pipeline ==="
echo "Step 1/3: Lint"
bash scripts/lint.sh

echo ""
echo "Step 2/3: Test"
bash scripts/test.sh

echo ""
echo "Step 3/3: Benchmark (if sample.txt exists)"
bash scripts/benchmark.sh || true

echo ""
echo "CI Pipeline complete."
```

Create `scripts/docker-build.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
source ~/.zshrc
cd "$(dirname "$0")/.."

TAG="${1:-textchunker:latest}"
echo "Building Docker image: $TAG"
docker build -t "$TAG" .
echo "Build complete: $TAG"
```

- [ ] **Step 2: Make scripts executable**

Run: `chmod +x /Users/mx/Desktop/series/项目系列/text-chunker/scripts/*.sh`

- [ ] **Step 3: Create Makefile**

Create `Makefile`:

```makefile
.PHONY: install test lint format benchmark experiment ci docker-build docker-run clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package with all extras and dev dependencies
	bash scripts/install.sh

test: ## Run tests with coverage
	bash scripts/test.sh

lint: ## Run flake8 + mypy + isort check
	bash scripts/lint.sh

format: ## Format code with isort + black
	bash scripts/format.sh

benchmark: ## Benchmark all strategies on sample data
	bash scripts/benchmark.sh

experiment: ## Run ablation experiments
	bash scripts/experiment.sh

ci: ## Run full CI pipeline (lint + test + benchmark)
	bash scripts/ci.sh

docker-build: ## Build Docker image
	bash scripts/docker-build.sh

docker-run: ## Run chunker in Docker container
	docker-compose up

clean: ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info __pycache__ .pytest_cache .mypy_cache logs/ stats/ results/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
```

- [ ] **Step 4: Commit**

```bash
git add scripts/ Makefile
git commit -m "feat: add bash scripts (install/test/lint/format/benchmark/experiment/ci/docker) and Makefile"
```

---

### Task 17: Docker

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.dockerignore`

- [ ] **Step 1: Create Dockerfile**

Create `Dockerfile`:

```dockerfile
# Multi-stage build for text-chunker
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
RUN pip install --no-cache-dir -e .

ENTRYPOINT ["python", "-m", "textchunker.cli"]
CMD ["--help"]
```

- [ ] **Step 2: Create docker-compose.yml**

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  chunker:
    build: .
    environment:
      - DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY}
    volumes:
      - ./input:/app/input
      - ./output:/app/output
      - ./logs:/app/logs
      - ./stats:/app/stats
      - ./configs:/app/configs
    command: >
      --config configs/default.yaml
      --input input/
      --output output/chunks.jsonl
      --visualize
      --stats
```

- [ ] **Step 3: Create .dockerignore**

Create `.dockerignore`:

```
.git
__pycache__
*.pyc
*.egg-info
.pytest_cache
.mypy_cache
logs/
stats/
results/
.DS_Store
*.swp
.claude/
docs/superpowers/
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "feat: add Docker multi-stage build, docker-compose, and .dockerignore"
```

---

### Task 18: CLAUDE.md

**Files:**
- Create: `.claude/claude.md`

- [ ] **Step 1: Create .claude directory and claude.md**

Create `.claude/claude.md`:

```markdown
# Text-Chunker Project

## Architecture
- Factory + Registry pattern for pluggable chunking strategies
- 5 strategies: fixed, recursive, semantic, structure, llm
- DashScope API (DASHSCOPE_API_KEY) for LLM and embeddings
- Models: qwen-max (LLM), text-embedding-v3 (embeddings)
- Settings class centralizes API/model/logging/stats configuration

## Development
- Python >=3.9
- Install: `pip install -e ".[all,dev]"` or `make install`
- Test: `make test` or `pytest --cov=textchunker`
- Lint: `make lint` (flake8 + mypy + isort)
- Format: `make format` (isort + black)
- Always `source ~/.zshrc` before running commands

## Conventions
- Type hints on all public functions
- Loguru for logging (never print())
- snake_case everywhere
- Tests mock heavy deps (models, APIs)
- Custom exceptions: ChunkerError, ConfigError, ProviderError (in exceptions.py)

## Key Modules
- `settings.py` — Settings dataclass (API keys, model names, log config)
- `logging.py` — Loguru setup (console + JSON file sinks)
- `stats.py` — StatsCollector singleton + @track_time / @count_calls decorators
- `factory.py` + `registry.py` — Strategy instantiation via @register decorator
- `chunkers/` — Strategy implementations (fixed, recursive, semantic, structure, llm_based)
- `providers/` — DashScope + HF backends (LLM and embedding)
- `experiments/` — Ablation studies and strategy comparison

## API Keys
- DASHSCOPE_API_KEY must be set in environment (loaded via `source ~/.zshrc`)
- Never hardcode API keys in source files

## Testing
- Stub sentence_transformers with dummy model in tests
- Mock DashScope/OpenAI API calls — never make real API calls in tests
- Use tmp_path fixture for file I/O tests
- Target: >60% coverage
```

- [ ] **Step 2: Commit**

```bash
git add .claude/claude.md
git commit -m "docs: add CLAUDE.md project guide"
```

---

### Task 19: README.md Rewrite

**Files:**
- Rewrite: `README.md`
- Move: `Doc.md` → `docs/strategy-guide.md`

- [ ] **Step 1: Move Doc.md**

Run: `mv /Users/mx/Desktop/series/项目系列/text-chunker/Doc.md /Users/mx/Desktop/series/项目系列/text-chunker/docs/strategy-guide.md`

- [ ] **Step 2: Rewrite README.md**

Replace `README.md`:

````markdown
# Text-Chunker

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A pluggable, production-ready text chunking framework for RAG (Retrieval-Augmented Generation) applications. Supports 5 chunking strategies with YAML configuration, CLI interface, DashScope/LLM integration, and comprehensive experimentation tools.

## Features

- **5 Chunking Strategies**: Fixed-size, Recursive, Semantic, Structure-based, LLM-based
- **Factory + Registry Pattern**: Add new strategies with a single `@register` decorator
- **DashScope Integration**: `qwen-max` for LLM chunking, `text-embedding-v3` for semantic embeddings
- **YAML + CLI Configuration**: Flexible config with runtime overrides
- **Advanced Logging**: Loguru-based structured logging with JSON file sinks and rotation
- **Statistics System**: Pipeline metrics, strategy comparison, historical run tracking
- **Rich Visualization**: Terminal-based chunk preview with tables and statistics
- **Experiment Framework**: Grid-search ablation studies on HuggingFace datasets
- **Multi-format Input**: TXT, Markdown, HTML, PDF
- **Docker Support**: Multi-stage build with docker-compose

## Architecture

```
CLI (cli.py) → Config (YAML + CLI args) → Settings → Logging Setup
      ↓
  Reader (txt/md/html/pdf) → FileDoc
      ↓
  Factory → Registry Lookup → Chunker Instance
      ↓
  ┌─ FixedChunker (char/token sliding window)
  ├─ RecursiveChunker (separator-priority recursive split)
  ├─ SemanticChunker (embedding similarity breakpoints)
  ├─ StructureChunker (heading-based sections)
  └─ LLMChunker (DashScope/HF model-proposed spans)
      ↓
  Stats Collection → Visualization → Export (JSONL/TXT)
```

## Quick Start

```bash
# Install with all extras
pip install -e ".[all,dev]"

# Basic chunking
python -m textchunker.cli --config configs/default.yaml --input your_file.txt --visualize

# With statistics
python -m textchunker.cli --input doc.txt --strategy recursive --stats
```

## Strategy Guide

| Strategy | Best For | Key Params |
|----------|----------|------------|
| `fixed` | Uniform chunks, simple use cases | `chunk_size`, `chunk_overlap`, `use_tokens` |
| `recursive` | General-purpose (recommended default) | `chunk_size`, `chunk_overlap`, `separators` |
| `semantic` | Topic-aware splitting | `min_similarity`, `sentence_window`, `model_name` |
| `structure` | Documents with headings (MD/HTML) | `prefer` (md/html/auto), `sub_split` |
| `llm` | Complex documents needing understanding | `provider`, `llm_model`, `system_prompt` |

## Configuration

### YAML Config (`configs/default.yaml`)

```yaml
settings:
  api_base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  llm_model: "qwen-max"
  embedding_model: "text-embedding-v3"

logging:
  level: INFO
  json: false
  log_dir: logs/
  rotation: "10 MB"

strategy:
  name: recursive
  common:
    chunk_size: 512
    chunk_overlap: 80
```

### CLI Reference

```bash
python -m textchunker.cli \
    --config configs/default.yaml \   # YAML config file
    --input docs/ \                   # Input file or directory
    --output out/chunks.jsonl \       # Output path
    --strategy semantic \             # Override strategy
    --chunk-size 600 \                # Override chunk size
    --chunk-overlap 100 \             # Override overlap
    --visualize \                     # Show Rich preview
    --stats \                         # Enable statistics
    --log-level DEBUG \               # Log verbosity
    --max-chunks 50                   # Limit output chunks
```

## Python API

```python
from textchunker.config import load_yaml, to_project_config
from textchunker.factory import create_chunker

cfg = to_project_config(load_yaml("configs/default.yaml"))
chunker = create_chunker(cfg.strategy)
chunks = chunker.chunk("Your text here...")

for c in chunks:
    print(f"[{c.start}:{c.end}] {c.text[:50]}...")
```

## Experiments

### Semantic Ablation

```bash
python -m textchunker.experiments.semantic_ablation \
    --dataset wikitext --subset wikitext-2-raw-v1 --split validation \
    --chunk-sizes 400 600 800 --min-sims 0.58 0.62 0.66 \
    --save-json results/semantic_ablation.json
```

### Recursive Ablation

```bash
python -m textchunker.experiments.recursive_ablation \
    --chunk-sizes 256 512 800 --chunk-overlaps 0 50 100 \
    --save-json results/recursive_ablation.json
```

### Strategy Comparison

```bash
python -m textchunker.experiments.strategy_comparison \
    --strategies fixed recursive --max-docs 32 \
    --save-json results/comparison.json
```

## Development

```bash
make install      # Install with all extras
make test         # Run tests with coverage
make lint         # Run flake8 + mypy
make format       # Format with isort + black
make benchmark    # Benchmark strategies
make experiment   # Run ablation experiments
make ci           # Full CI pipeline
```

### Docker

```bash
make docker-build                           # Build image
docker-compose up                           # Run with docker-compose
docker run -e DASHSCOPE_API_KEY=$DASHSCOPE_API_KEY textchunker:latest \
    --config configs/default.yaml --input /app/input/doc.txt
```

### Adding a New Strategy

```python
# textchunker/chunkers/my_strategy.py
from textchunker.registry import register
from textchunker.chunkers.base import BaseChunker

@register("my_strategy")
class MyChunker(BaseChunker):
    def chunk(self, text: str) -> List[Chunk]:
        # Your implementation here
        ...
```

Then import it in `textchunker/chunkers/__init__.py`.

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `DASHSCOPE_API_KEY` | DashScope API authentication | For LLM/embedding strategies |

## Project Structure

```
textchunker/
├── cli.py                 # CLI entry point
├── config.py              # YAML/CLI configuration
├── settings.py            # Settings dataclass
├── logging.py             # Loguru setup
├── stats.py               # Statistics collection
├── exceptions.py          # Custom exceptions
├── factory.py             # Chunker factory
├── registry.py            # Strategy registry
├── types.py               # Data structures
├── utils.py               # Token counting, sentence splitting
├── readers.py             # Multi-format file reading
├── export.py              # JSONL/TXT export
├── visualization.py       # Rich terminal output
├── chunkers/              # Strategy implementations
│   ├── fixed.py
│   ├── recursive.py
│   ├── semantic.py
│   ├── structure.py
│   └── llm_based.py
├── providers/             # LLM & embedding backends
│   ├── dashscope_provider.py
│   ├── embeddings.py
│   └── llm.py
└── experiments/           # Ablation & comparison tools
    ├── semantic_ablation.py
    ├── recursive_ablation.py
    └── strategy_comparison.py
```

## License

MIT
````

- [ ] **Step 3: Commit**

```bash
git add README.md docs/strategy-guide.md
git rm --cached Doc.md 2>/dev/null || true
git commit -m "docs: rewrite README.md with full documentation, move Doc.md to docs/strategy-guide.md"
```

---

### Task 20: Update conftest.py & Run Full Test Suite

**Files:**
- Modify: `tests/conftest.py`

- [ ] **Step 1: Expand conftest.py**

Replace `tests/conftest.py`:

```python
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_text_short():
    return "First sentence. Second sentence. Third sentence."


@pytest.fixture
def sample_text_long():
    return "Paragraph one with content.\n\n" * 20


@pytest.fixture
def sample_text_chinese():
    return "第一段话。 第二段话！ 第三段话？ " * 10


@pytest.fixture
def default_strategy_config():
    from textchunker.types import StrategyConfig
    return StrategyConfig(
        name="recursive",
        common={"chunk_size": 100, "chunk_overlap": 20},
        recursive={"separators": ["\n\n", "\n", " ", ""]},
        semantic={}, fixed={}, structure={}, llm={},
    )
```

- [ ] **Step 2: Run full test suite**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Run with coverage**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/ -v --cov=textchunker --cov-report=term-missing`
Expected: Coverage report showing >=60% coverage

- [ ] **Step 4: Commit**

```bash
git add tests/conftest.py
git commit -m "test: expand conftest with shared fixtures, verify full test suite passes"
```

---

### Task 21: Final Integration Verification

- [ ] **Step 1: Run full test suite**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m pytest tests/ -v --cov=textchunker --cov-report=term-missing`
Expected: All tests PASS, coverage >=60%

- [ ] **Step 2: Run lint**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m flake8 textchunker/ tests/ --max-line-length=100 --extend-ignore=E203,W503`
Expected: No errors (or only minor warnings)

- [ ] **Step 3: Verify CLI works**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -m textchunker.cli --help`
Expected: Help output showing all arguments

- [ ] **Step 4: Verify imports**

Run: `source ~/.zshrc && cd /Users/mx/Desktop/series/项目系列/text-chunker && python -c "from textchunker.settings import Settings; from textchunker.logging import setup_logging; from textchunker.stats import StatsCollector; from textchunker.exceptions import ChunkerError; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 5: Final commit if any fixes needed**

```bash
git add -A
git commit -m "chore: final integration fixes and cleanup"
```
