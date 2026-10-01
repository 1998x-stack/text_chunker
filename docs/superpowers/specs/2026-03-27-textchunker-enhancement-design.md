# Text-Chunker Enhancement Design Spec

**Date**: 2026-03-27
**Approach**: Layered Enhancement (in-place, additive modules)
**Scope**: Full professional-grade enhancement across 6 areas

---

## 1. Bug Fixes & Code Quality

### 1.1 Critical Bug Fixes

**SemanticChunker offset tracking** (`chunkers/semantic.py`):
- Current: `text.find(sent, cursor)` fails on duplicate sentences
- Fix: Build offset map upfront during sentence splitting — `whitespace_sentences()` returns `List[Tuple[str, int, int]]` (sentence, start, end)
- Impact: Eliminates incorrect chunk.start/end values

**FixedChunker token mode** (`chunkers/fixed.py`):
- Current: Token path has `pass` — silently falls back to character mode
- Fix: Implement tiktoken encode → slice token IDs → decode for proper token-boundary chunking
- Fallback: If tiktoken unavailable, use character mode with warning log

**HFProvider JSON parsing** (`providers/llm.py`):
- Current: Naive `find("[")` / `rfind("]")` can grab wrong brackets
- Fix: Use `re.search(r'\[.*\]', out, re.DOTALL)` with JSON validation, reject non-list results

**RecursiveChunker offset tracking** (`chunkers/recursive.py`):
- Current: Same `text.find(part, cursor)` duplicate issue
- Fix: Track cumulative offset through split operations

**Silent error swallowing** (all providers):
- Current: Return `[]` silently on failure
- Fix: Log `logger.warning(...)` with error context, raise `ChunkerError` for non-recoverable failures

### 1.2 Code Style Improvements

- Add type hints to all public function signatures
- Custom exception hierarchy: `ChunkerError`, `ConfigError`, `ProviderError`
- `__all__` exports in every `__init__.py`
- Replace `Dict[str, Any]` config params with typed sub-dataclasses where feasible
- Consistent snake_case naming throughout
- Remove dead code and unused imports

---

## 2. Settings Class & DashScope Integration

### 2.1 New Module: `textchunker/settings.py`

```python
@dataclass
class Settings:
    # API
    api_key: str              # from DASHSCOPE_API_KEY env var
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
    def from_env_and_yaml(cls, yaml_cfg: dict) -> "Settings": ...

    @classmethod
    def from_env(cls) -> "Settings": ...
```

### 2.2 Provider Refactoring

- Rename/refactor `OpenAIProvider` → `DashScopeProvider`
  - Uses OpenAI-compatible SDK pointed at DashScope `api_base_url`
  - Reads `DASHSCOPE_API_KEY` from `Settings`
  - Default model: `qwen-max`
- Keep `HFProvider` for local model inference
- `SemanticChunker` gains option to use `text-embedding-v3` via DashScope API (OpenAI-compatible embeddings endpoint) alongside sentence-transformers fallback
- Add `BaseLLMProvider` → `DashScopeProvider`, `HFProvider`
- Add `BaseEmbeddingProvider` → `DashScopeEmbeddingProvider`, `SentenceTransformerProvider`

### 2.3 YAML Config Extension

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
  retention: 5
```

---

## 3. Advanced Loguru Logging System

### 3.1 New Module: `textchunker/logging.py`

**`setup_logging(settings: Settings)`** — single entry point:

- **Console sink**: Colored, `{time:HH:mm:ss} | {level:<8} | {module}:{function}:{line} | {message}` format
- **File sink**: JSON-structured, rotation by size (default 10MB), retention count (default 5)
- **Per-module levels**: Via `logger.bind()` and filter functions
- **Context binding**: `logger.bind(strategy="semantic", doc="file.txt")` for structured fields
- **Lazy formatting**: loguru's native `{}` syntax, no f-string evaluation when filtered

### 3.2 Integration Points

| Module | What gets logged |
|--------|-----------------|
| `cli.py` | Pipeline start/end, config loaded, files processed count |
| Each chunker | Chunk count produced, split decisions at DEBUG, timing at INFO |
| Providers | API call start/end, response sizes, errors with context |
| Experiments | Scenario progress, metric summaries |
| `stats.py` | Run summaries, comparison results |
| `readers.py` | File read start, format detected, size |

### 3.3 Log Levels Convention

| Level | Usage |
|-------|-------|
| TRACE | Sentence-level splits, individual similarity scores |
| DEBUG | Per-chunk creation, offset calculations |
| INFO | Pipeline milestones, file counts, run summaries |
| WARNING | Fallback behavior, missing optional deps, API retries |
| ERROR | Failed API calls, invalid config, file read failures |

---

## 4. Statistics System

### 4.1 New Module: `textchunker/stats.py`

**`StatsCollector` (singleton)**:
```python
class StatsCollector:
    # Per-run metrics
    chunk_counts: List[int]        # chunks per document
    chunk_sizes: List[int]         # character sizes
    chunk_tokens: List[int]        # token sizes
    processing_times: Dict[str, float]  # function → total seconds
    call_counts: Dict[str, int]    # function → invocation count

    # Computed metrics
    def summary() -> Dict           # mean, p50, p95, std, min, max
    def redundancy_ratio() -> float
    def coverage_ratio() -> float
    def boundary_rate(max_size) -> float

    # I/O
    def save_run(path, metadata)    # JSON with timestamp + strategy info
    def load_runs(dir) -> List      # historical runs
    def reset()                     # clear for next run
```

### 4.2 Decorators

```python
@track_time      # measures wall-clock time, stores in StatsCollector
@count_calls     # increments call counter in StatsCollector
```

Applied to: `chunk()`, `load_inputs()`, `propose_spans()`, `encode()` methods.

### 4.3 Strategy Comparison

```python
def compare_strategies(text: str, strategies: List[str], cfg: ProjectConfig) -> ComparisonReport:
    """Run multiple strategies on same text, return comparison table."""
```

Output: Rich table with columns: Strategy | Chunks | Avg Size | P95 | Redundancy | Coverage | Time

### 4.4 Historical Run Tracking

- `stats/` directory with `{timestamp}_{strategy}.json` files
- Each file: `{strategy, params, metrics, timestamp, input_summary}`
- CLI: `--stats` enables collection, `--stats-dir` sets output path
- `--compare-strategies fixed,recursive,semantic` runs head-to-head comparison

### 4.5 Rich Output

- Summary table after each run (auto-displayed when `--stats` enabled)
- ASCII histogram of chunk size distribution
- Comparison table for multi-strategy runs

---

## 5. Tests & Experiments

### 5.1 Test Files (target: comprehensive coverage)

| Test File | Scope |
|-----------|-------|
| `tests/test_fixed_chunker.py` | Char mode, token mode, overlap, empty input, single-char input |
| `tests/test_recursive_chunker.py` | Separator hierarchy, offset correctness, overlap, deep nesting |
| `tests/test_semantic_chunker.py` | Existing + offset map verification, similarity thresholds |
| `tests/test_structure_chunker.py` | Markdown sections, sub-splitting, no-headers fallback |
| `tests/test_llm_chunker.py` | Mock provider, span parsing, fallback to single chunk |
| `tests/test_readers.py` | TXT, MD, HTML, PDF reading, directory recursion, missing files |
| `tests/test_config.py` | YAML loading, CLI merge, missing fields, defaults |
| `tests/test_settings.py` | Settings from env, from YAML, validation |
| `tests/test_logging.py` | setup_logging, JSON mode, level filtering |
| `tests/test_stats.py` | StatsCollector, decorators, comparison, save/load |
| `tests/test_export.py` | Existing + txt_dir format, edge cases |
| `tests/test_utils.py` | Token counting, sentence splitting, sliding windows, clamp |
| `tests/test_registry.py` | Register, retrieve, duplicate name, available() |
| `tests/test_providers.py` | DashScope mock, HF mock, error handling |

### 5.2 `conftest.py` Fixtures

- `sample_text_short` / `sample_text_long` / `sample_text_chinese` — realistic test content
- `mock_embedding_model` — returns deterministic embeddings
- `mock_dashscope_response` — simulated API responses
- `default_config` / `semantic_config` / `llm_config` — pre-built configs
- `tmp_output_dir` — temp directory for export tests
- `stats_collector` — fresh StatsCollector per test

### 5.3 New Experiments

**`experiments/recursive_ablation.py`**:
- Grid search: chunk_size × chunk_overlap × separator_sets
- Same metrics framework as semantic_ablation

**`experiments/strategy_comparison.py`**:
- Runs all 5 strategies (or subset) on same dataset
- Outputs comparison table + JSON
- Supports HuggingFace datasets + local files

**Experiment Enhancements**:
- Rich Progress bars during ablation runs
- CSV export alongside JSON
- Better CLI with `--progress` flag

---

## 6. Bash Scripts, Makefile & Docker

### 6.1 Scripts (all with `source ~/.zshrc` prefix)

| Script | Purpose |
|--------|---------|
| `scripts/install.sh` | `pip install -e ".[semantic,structure,llm,token,experiments]"` + dev deps |
| `scripts/test.sh` | `pytest --cov=textchunker --cov-report=term-missing --cov-fail-under=80` |
| `scripts/lint.sh` | `flake8 textchunker/ tests/` + `mypy textchunker/` + `isort --check .` |
| `scripts/format.sh` | `isort .` + `black textchunker/ tests/` |
| `scripts/benchmark.sh` | Run all strategies on sample data, output timing comparison |
| `scripts/experiment.sh` | Run semantic + recursive ablation with sensible defaults |
| `scripts/ci.sh` | `lint.sh && test.sh && benchmark.sh` |
| `scripts/docker-build.sh` | Build Docker image with tag |

### 6.2 Makefile

Targets: `install`, `test`, `lint`, `format`, `benchmark`, `experiment`, `docker-build`, `docker-run`, `clean`, `help`

### 6.3 Docker

- `Dockerfile`: Multi-stage, Python 3.11-slim, installs all extras
- `docker-compose.yml`: Service with volume mounts for `input/`, `output/`, `logs/`, `stats/`
- `.dockerignore`: .git, __pycache__, logs/, stats/, *.egg-info

---

## 7. README & Documentation

### 7.1 README.md — Complete Rewrite

Structure:
1. Badges (Python version, license, tests, coverage)
2. One-line description + feature highlights
3. Architecture diagram (ASCII)
4. Quick Start (install → chunk in 3 commands)
5. Strategy comparison table (5 strategies with when-to-use)
6. Configuration reference (YAML schema)
7. CLI reference with examples
8. Python API usage
9. Statistics & comparison usage
10. Experiments guide
11. Scripts & Docker
12. Contributing (how to add a new chunker)

### 7.2 CLAUDE.md

- Project architecture overview
- Development workflow (install → code → test → lint)
- Coding conventions (type hints, snake_case, loguru usage)
- Test requirements (mock heavy deps, use fixtures)
- Key files and their roles
- DashScope API usage notes

### 7.3 Other Docs

- `Doc.md` → `docs/strategy-guide.md` (rename + keep as deep reference)
- `ChatGPT-文本分块策略分析.md` stays as research reference

---

## 8. CLAUDE.md Content

```
# Text-Chunker Project

## Architecture
- Factory + Registry pattern for pluggable chunking strategies
- 5 strategies: fixed, recursive, semantic, structure, llm
- DashScope API (DASHSCOPE_API_KEY) for LLM and embeddings
- Models: qwen-max (LLM), text-embedding-v3 (embeddings)

## Development
- Python >=3.9, install: pip install -e ".[semantic,structure,llm,token,experiments]"
- Test: pytest --cov=textchunker
- Lint: flake8 + mypy
- Always source ~/.zshrc before running commands

## Conventions
- Type hints on all public functions
- Loguru for logging (never print())
- snake_case everywhere
- Tests mock heavy deps (models, APIs)
- Custom exceptions: ChunkerError, ConfigError, ProviderError

## Key Modules
- settings.py: Settings dataclass (API keys, model names, log config)
- logging.py: Loguru setup (console + JSON file sinks)
- stats.py: StatsCollector singleton + decorators
- factory.py + registry.py: Strategy instantiation
- chunkers/: Strategy implementations
- providers/: DashScope + HF backends
```

---

## File Change Summary

### New Files
| File | Purpose |
|------|---------|
| `textchunker/settings.py` | Settings dataclass |
| `textchunker/logging.py` | Loguru configuration |
| `textchunker/stats.py` | Statistics collection + comparison |
| `textchunker/exceptions.py` | Custom exception hierarchy |
| `textchunker/providers/dashscope.py` | DashScope LLM provider |
| `textchunker/providers/embeddings.py` | Embedding providers (DashScope + ST) |
| `textchunker/experiments/recursive_ablation.py` | Recursive param grid search |
| `textchunker/experiments/strategy_comparison.py` | Cross-strategy comparison |
| `tests/test_fixed_chunker.py` | FixedChunker tests |
| `tests/test_recursive_chunker.py` | RecursiveChunker tests |
| `tests/test_structure_chunker.py` | StructureChunker tests |
| `tests/test_llm_chunker.py` | LLMChunker tests |
| `tests/test_readers.py` | Reader tests |
| `tests/test_config.py` | Config tests |
| `tests/test_settings.py` | Settings tests |
| `tests/test_logging.py` | Logging tests |
| `tests/test_stats.py` | Stats tests |
| `tests/test_utils.py` | Utils tests |
| `tests/test_registry.py` | Registry tests |
| `tests/test_providers.py` | Provider tests |
| `scripts/install.sh` | Install script |
| `scripts/test.sh` | Test runner |
| `scripts/lint.sh` | Linter runner |
| `scripts/format.sh` | Formatter |
| `scripts/benchmark.sh` | Benchmark runner |
| `scripts/experiment.sh` | Experiment runner |
| `scripts/ci.sh` | CI pipeline |
| `scripts/docker-build.sh` | Docker builder |
| `Makefile` | Build targets |
| `Dockerfile` | Container image |
| `docker-compose.yml` | Container orchestration |
| `.dockerignore` | Docker exclusions |
| `.claude/claude.md` | CLAUDE.md project guide |
| `docs/strategy-guide.md` | Moved from Doc.md |

### Modified Files
| File | Changes |
|------|---------|
| `textchunker/types.py` | Add exception classes import, refine dataclasses |
| `textchunker/config.py` | Integrate Settings, add new CLI args |
| `textchunker/utils.py` | Return offsets from sentence splitting |
| `textchunker/cli.py` | Integrate logging, stats, new providers |
| `textchunker/chunkers/fixed.py` | Implement token mode, fix overlap |
| `textchunker/chunkers/semantic.py` | Fix offset tracking, add embedding provider option |
| `textchunker/chunkers/recursive.py` | Fix offset tracking |
| `textchunker/chunkers/structure.py` | Fix overlap, improve section detection |
| `textchunker/chunkers/llm_based.py` | Use DashScopeProvider, improve error handling |
| `textchunker/providers/llm.py` | Refactor, add DashScope support |
| `textchunker/experiments/semantic_ablation.py` | Add progress bars, CSV export |
| `textchunker/__init__.py` | Expand exports |
| `textchunker/chunkers/__init__.py` | Add __all__ |
| `textchunker/providers/__init__.py` | Add new providers |
| `textchunker/experiments/__init__.py` | Add new experiments |
| `tests/conftest.py` | Expand fixtures |
| `setup.cfg` | Add new deps, update version |
| `requirements.txt` | Add new deps |
| `configs/default.yaml` | Add settings + logging sections |
| `configs/semantic.yaml` | Add embedding_model config |
| `configs/llm_based.yaml` | Switch to DashScope/qwen-max |
| `README.md` | Complete rewrite |
