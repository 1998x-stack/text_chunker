<p align="center">
  <h1 align="center">Text-Chunker</h1>
  <p align="center">
    <strong>A pluggable, production-ready text chunking framework for RAG applications</strong>
  </p>
  <p align="center">
    <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.9%2B-blue.svg?style=flat-square" alt="Python 3.9+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="MIT License"></a>
    <a href="https://github.com/1998x-stack/text_chunker/actions"><img src="https://img.shields.io/badge/CI-passing-brightgreen?style=flat-square" alt="CI"></a>
    <a href="https://github.com/1998x-stack/text_chunker"><img src="https://img.shields.io/badge/coverage-62%25-yellow?style=flat-square" alt="Coverage"></a>
    <a href="https://github.com/1998x-stack/text_chunker/pulls"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome"></a>
  </p>
</p>

---

> **5 strategies. 1 decorator to add your own. Zero lock-in.**
>
> Split any document — English, Chinese, mixed-language, code, math — into semantically meaningful chunks for RAG pipelines, search indexes, and LLM context windows.

## Why Text-Chunker?

| Pain Point | How Text-Chunker Solves It |
|---|---|
| Fixed-size splits break mid-sentence | **Recursive chunker** respects paragraph/sentence boundaries |
| Topic shifts ignored | **Semantic chunker** detects topic boundaries via embeddings |
| Structured docs lose hierarchy | **Structure chunker** splits on headings, preserving context |
| One strategy doesn't fit all docs | **Registry pattern** — swap strategies via config, not code |
| Hard to measure chunk quality | **Built-in metrics**: redundancy, coverage, boundary rate, token distribution |
| Experiment overhead | **Grid-search ablation** on HuggingFace datasets with CSV/JSON export |

## Strategies at a Glance

```
            Speed                          Quality
  Fast ◄─────────────────────────────────────► Best

  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  ┌─────────┐
  │  Fixed   │  │ Recursive │  │Structure │  │ Semantic  │  │   LLM   │
  │ O(n)     │  │ O(n)      │  │ O(n)     │  │ O(n·emb)  │  │ O(n·api)│
  │ char/tok │  │ separators│  │ headings │  │ embeddings│  │ qwen-max│
  └──────────┘  └───────────┘  └──────────┘  └───────────┘  └─────────┘
```

| Strategy | Best For | Speed | Cost |
|:---------|:---------|:-----:|:----:|
| **Fixed** | Uniform chunks, batch indexing | Instant | Free |
| **Recursive** | General-purpose (recommended) | Instant | Free |
| **Structure** | Markdown/HTML with headings | Instant | Free |
| **Semantic** | Topic-aware splitting | Fast | GPU/CPU |
| **LLM** | Complex documents needing deep understanding | Slow | API calls |

## Quick Start

```bash
# Install
pip install -e ".[all,dev]"

# Chunk a document
python -m textchunker.cli --input doc.txt --strategy recursive --visualize

# With statistics
python -m textchunker.cli --input doc.txt --strategy semantic --stats --chunk-size 600
```

**Output:**

```
╭───────────────────────── Chunks=8, TotalChars=9649 ──────────────────────────╮
│ ┏━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
│ ┃ ID ┃ Chars ┃ Span        ┃ Meta               ┃ Preview                  ┃ │
│ ┡━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩ │
│ │  0 │ 632   │ [0,632)     │ strategy:recursive │ 第一章：深度学习的数学基… │ │
│ │  1 │ 1990  │ [582,2572)  │ strategy:recursive │ Chapter 2: The Archite…  │ │
│ │  2 │ 711   │ [2522,3233) │ strategy:recursive │ 第三章：检索增强生成…     │ │
│ │ .. │ ...   │ ...         │ ...                │ ...                      │ │
│ └────┴───────┴─────────────┴────────────────────┴──────────────────────────┘ │
╰──────────────────────────────────────────────────────────────────────────────╯

Pipeline Statistics
  total_chunks: 8     avg_chunk_tokens: 463     redundancy_ratio: 0.044
  coverage_ratio: 1.038     boundary_rate: 1.000
```

## Python API

```python
from textchunker.config import load_yaml, to_project_config
from textchunker.factory import create_chunker

cfg = to_project_config(load_yaml("configs/default.yaml"))
chunker = create_chunker(cfg.strategy)
chunks = chunker.chunk("Your text here...")

for c in chunks:
    print(f"[{c.start}:{c.end}] {c.text[:80]}...")
```

### Add Your Own Strategy in 3 Lines

```python
from textchunker.registry import register
from textchunker.chunkers.base import BaseChunker

@register("my_strategy")
class MyChunker(BaseChunker):
    def chunk(self, text: str) -> list:
        ...  # your logic
```

Then add `from . import my_strategy` to `textchunker/chunkers/__init__.py`.

## Architecture

```
                    ┌──────────────────────────────┐
                    │          CLI / API            │
                    │   (YAML + CLI arg merging)    │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │     Settings + Logging        │
                    │  (Loguru, JSON sinks, rotate)  │
                    └──────────┬───────────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │         Reader (txt/md/html/pdf)     │
              └────────────────┬────────────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │    Factory + Registry  (@register)   │
              ├─────┬──────┬──────┬──────┬──────────┤
              │Fixed│Recur.│Seman.│Struc.│   LLM    │
              └─────┴──────┴──┬───┴──────┴────┬─────┘
                              │               │
                    ┌─────────▼──┐   ┌────────▼────────┐
                    │ Embeddings │   │  LLM Providers   │
                    │ (DashScope │   │  (DashScope/HF)  │
                    │  / ST)     │   │  qwen-max        │
                    └────────────┘   └─────────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │  Stats + Visualization + Export      │
              │  (metrics, Rich tables, JSONL/TXT)   │
              └─────────────────────────────────────┘
```

## Benchmark: Multilingual Stress Test

Tested on a **9,299-character multilingual document** (Chinese + English, math formulas, code blocks, Markdown tables, Unicode symbols):

| Strategy | Chunks | Avg Chars | Avg Tokens | Redundancy | Coverage | Time |
|:---------|:------:|:---------:|:----------:|:----------:|:--------:|:----:|
| **Fixed** | 8 | 1,281 | 488 | 9.9% | 110.2% | 3ms |
| **Recursive** | 8 | 1,206 | 464 | 4.4% | 103.8% | 14ms |
| **Structure** | 5 | 833 | 325 | 0.0% | 44.8% | 6ms |

> **Note:** Structure strategy shows lower coverage on this document because it targets Markdown headings (`#`), and many sections use Chinese-style headings without `#` markers. This is expected — structure chunking excels on well-formatted Markdown/HTML.

## Experiment Framework

Run grid-search ablation studies on HuggingFace datasets:

```bash
# Semantic ablation — sweep similarity thresholds × chunk sizes
python -m textchunker.experiments.semantic_ablation \
    --dataset wikitext --subset wikitext-2-raw-v1 --split validation \
    --chunk-sizes 400 600 800 --min-sims 0.58 0.62 0.66 \
    --save-json results/semantic_ablation.json

# Recursive ablation — sweep chunk sizes × overlaps
python -m textchunker.experiments.recursive_ablation \
    --chunk-sizes 256 512 800 --chunk-overlaps 0 50 100

# Cross-strategy comparison
python -m textchunker.experiments.strategy_comparison \
    --strategies fixed recursive semantic --max-docs 32
```

## Configuration

<details>
<summary><strong>YAML Config</strong> (click to expand)</summary>

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
  recursive:
    separators: ["\n\n", "\n", " ", ""]
  semantic:
    min_similarity: 0.62
    sentence_window: 3
```

</details>

<details>
<summary><strong>CLI Reference</strong> (click to expand)</summary>

```bash
python -m textchunker.cli \
    --config configs/default.yaml \   # YAML config file
    --input docs/ \                   # Input file or directory
    --output out/chunks.jsonl \       # Output path
    --strategy semantic \             # Override strategy
    --chunk-size 600 \                # Override chunk size
    --chunk-overlap 100 \             # Override overlap
    --visualize \                     # Show Rich preview table
    --stats \                         # Enable pipeline statistics
    --stats-dir stats/ \              # Stats output directory
    --log-level DEBUG \               # Log verbosity
    --llm-model qwen-max \            # LLM model name
    --embedding-model text-embedding-v3  # Embedding model
    --max-chunks 50                   # Limit output chunks
```

</details>

## Chunk Size Guide

| Use Case | Recommended Size | Why |
|:---------|:----------------:|:----|
| Question Answering | 200–400 tokens | Higher precision, less noise |
| Summarization | 800–1,200 tokens | Preserves broader context |
| General RAG | 400–600 tokens | Best balance (sweet spot) |
| Code Documentation | 300–500 tokens | Keeps functions intact |

## Development

```bash
make install      # Install with all extras
make test         # pytest with coverage (target: >60%)
make lint         # flake8 + mypy + isort check
make format       # Auto-format with isort + black
make benchmark    # Benchmark all strategies
make experiment   # Run ablation experiments
make ci           # Full CI pipeline (lint + test)
```

### Docker

```bash
make docker-build
docker-compose up
docker run -e DASHSCOPE_API_KEY=$DASHSCOPE_API_KEY textchunker:latest \
    --config configs/default.yaml --input /app/input/doc.txt
```

## Project Structure

```
textchunker/
├── cli.py                  # CLI entry point
├── config.py               # YAML/CLI configuration
├── settings.py             # Settings dataclass
├── log_config.py           # Loguru setup (console + JSON sinks)
├── stats.py                # StatsCollector + decorators
├── exceptions.py           # ChunkerError, ConfigError, ProviderError
├── factory.py              # Strategy factory
├── registry.py             # @register decorator + registry
├── types.py                # Chunk, FileDoc, StrategyConfig
├── utils.py                # Token counting, sentence splitting
├── readers.py              # Multi-format file reader
├── export.py               # JSONL/TXT export
├── visualization.py        # Rich terminal tables
├── chunkers/               # Strategy implementations
│   ├── fixed.py            #   char/token sliding window
│   ├── recursive.py        #   separator-priority recursive split
│   ├── semantic.py         #   embedding similarity breakpoints
│   ├── structure.py        #   heading-based sections
│   └── llm_based.py        #   LLM-proposed span boundaries
├── providers/              # LLM & embedding backends
│   ├── dashscope_provider.py  # DashScope (qwen-max)
│   ├── embeddings.py          # DashScope + SentenceTransformer
│   └── llm.py                 # Base + HuggingFace provider
└── experiments/            # Ablation & comparison tools
    ├── semantic_ablation.py
    ├── recursive_ablation.py
    └── strategy_comparison.py
```

## Environment Variables

| Variable | Purpose | Required |
|:---------|:--------|:--------:|
| `DASHSCOPE_API_KEY` | DashScope API authentication | For `semantic` / `llm` strategies |

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/amazing-feature`)
3. Add tests for new functionality
4. Run `make ci` to verify
5. Submit a Pull Request

## License

MIT
