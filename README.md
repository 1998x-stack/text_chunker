# Text-Chunker

Source-traceable text chunking for RAG experiments and indexing. Supports five strategies and a strict **DocumentIR → ChunkingResult → independent evaluation** pipeline.

[中文文档](docs/README.zh-CN.md) · [Architecture](docs/architecture-v0.3.md) · [Documentation index](docs/INDEX.md) · [Changelog](CHANGELOG.md)

**Status:** V0.3 implemented. Markdown support is a lossless line-block parser, **not** a full AST. HTML and PDF preserve extracted-text provenance, but cannot yet map spans to original DOM nodes, PDF glyph coordinates or bounding boxes. Structure-aware Hybrid Chunking is planned for V0.4. No retrieval-quality improvement claims are made.

## Install

Requirements: **Python 3.10+**. The minimal install does not load the heavy embedding/LLM stacks.

```bash
python -m pip install -e .
# Character mode needs no tokenizer extras.
python -m pip install -e ".[token,structure,dev]"
# Optional, much heavier:
python -m pip install -e ".[semantic]"  # sentence-transformers
python -m pip install -e ".[llm]"       # providers and transformer stack
```

## CLI

```bash
python -m textchunker.cli --config configs/default.yaml \
    --input sample.txt --output out/chunks.jsonl \
    --strategy recursive --tokenizer char --chunk-size 128 --chunk-overlap 16

# Multiple documents, one JSONL file (output cleared once per run).
python -m textchunker.cli --input docs/ --output out/docs.jsonl \
    --strategy recursive --tokenizer tiktoken:cl100k_base
```

The CLI also accepts `--strategy fixed|recursive|structure|semantic|llm`, `--max-chunks`, `--stats`, `--visualize` and `--config`; run `python -m textchunker.cli --help` for the full list. The default YAML enables preview output; override it with your own configuration for automated runs.

**Token units:** `--tokenizer char` counts Unicode code points. `tiktoken:cl100k_base` and `hf:<model>` require their corresponding dependencies. `legacy` defaults to cl100k_base when installed, else character count. Fixed mode with `use_tokens: false` always uses character units.

## Python API

```python
from textchunker.parsers import parse_markdown
from textchunker.pipeline import run_chunking
from textchunker.tokenizers import CharacterTokenizer
from textchunker.types import StrategyConfig

document = parse_markdown("# Guide\nA paragraph.\n", "docs/guide.md")
config = StrategyConfig(
    name="recursive",
    common={"chunk_size": 128, "chunk_overlap": 16},
)
result = run_chunking(document, config, CharacterTokenizer())
assert result.report.ok and result.complete
for chunk in result.chunks:
    print(chunk.chunk_id, chunk.span, chunk.heading_path)
    print(chunk.source_locations)
```

An existing direct interface `create_chunker(config).chunk(text)` is preserved. New callers should use `run_chunking` for explicit tokenizer identity, stable chunk IDs, source maps, completeness and independent validation.

**Integrity:** a successful full-document result has ordered spans, no uncovered source characters, matching text and valid token budgets. An explicitly limited run (`max_chunks`) may return `result.complete == False` with an INCOMPLETE warning; partial results must not be treated as a complete document index.

## Data flow

```text
readers / parsers
    → DocumentIR(text, blocks, SourceMap)
    → chunker registry (fixed/recursive/structure/semantic/llm)
    → ChunkRecord + source locations
    → independent correctness evaluator
    → JSONL / TXT / reports
```

`DocumentIR.text` is the canonical Python Unicode string and `Span(start,end)` is half-open in **Unicode code points, not UTF-8 bytes**. Plain TXT/Markdown use exact source-character mapping; HTML/PDF extraction sets raw source spans to null. PDF retains 1-based page anchors, not page geometry.

## Code layout

```text
textchunker/
  core/          # DocumentIR, spans, contracts, validation and budgets
  parsers/       # text, Markdown line blocks, HTML extraction, PDF pages
  tokenizers/    # char, legacy, tiktoken, HF adapters
  chunkers/      # five algorithms
  pipeline/      # DocumentIR-first runner
  evaluation/    # independent integrity evaluation
  providers/     # optional embedding and LLM providers
  experiments/   # historical ablations (legacy metrics, see below)
  cli.py, export.py, readers.py, config.py, types.py
tests/
  unit/, property/, integration/ and legacy tests
docs/           # authoritative architecture and usage documentation
```

## Verification

```bash
python -m pip install -e ".[token,structure,dev]" numpy
bash scripts/ci.sh
# Or inspect the GitHub Actions workflow under .github/workflows/ci.yml
```

CI checks syntax, fatal lint errors and pytest with a 60% minimum coverage gate. It does **not** yet establish strict typing compliance, full style conformance, performance benchmarking or live provider correctness.

Old benchmark percentages over 100% called “coverage” actually measured output/input character length (including overlap). Real unique-span coverage is bounded by 100%; historical figures should not be used as new performance evidence.

## Documentation

See [documentation index](docs/INDEX.md), [CLI and configuration](docs/usage.md), [architecture](docs/architecture-v0.3.md), [development and CI](docs/development.md), and [legacy strategy research](docs/strategy-guide.md).

License: [MIT](LICENSE).
