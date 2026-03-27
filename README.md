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
CLI (cli.py) -> Config (YAML + CLI args) -> Settings -> Logging Setup
      |
  Reader (txt/md/html/pdf) -> FileDoc
      |
  Factory -> Registry Lookup -> Chunker Instance
      |
  +- FixedChunker (char/token sliding window)
  +- RecursiveChunker (separator-priority recursive split)
  +- SemanticChunker (embedding similarity breakpoints)
  +- StructureChunker (heading-based sections)
  +- LLMChunker (DashScope/HF model-proposed spans)
      |
  Stats Collection -> Visualization -> Export (JSONL/TXT)
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
├── log_config.py          # Loguru setup
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
