# Text-Chunker

Source-traceable text chunking for RAG experiments and indexing. Supports six strategies and a strict **DocumentIR → ChunkingResult → independent evaluation** pipeline.

[中文文档](docs/README.zh-CN.md) · [Architecture](docs/architecture-v0.5-v0.6.md) · [Documentation index](docs/INDEX.md) · [Changelog](CHANGELOG.md)

**Status:** V0.4 structural splitting plus V0.5 model-reliability controls and V0.6 offline golden-answer retrieval evaluation are included in the consolidated PR. Docling native document objects can provide page and bounding-box provenance through an optional adapter. Live Docling conversion, production RAG quality improvements, PDF OCR accuracy, and real hosted-model reliability have **not** been validated in CI.

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

## Model reliability & reproducible evaluation (V0.5–V0.6)

- `SemanticChunker`: injectable embedding backend, finite model-call and batch budgets, vector validation/normalization, optional per-instance cache and explicit recursive fallback on failure.
- `LLMChunker`: strict JSON offset parser, bounded attempts/provider calls/prompt characters, timeout for compatible API calls, explicit `fallback`/`error` behavior; source coverage remains complete even after budget exhaustion.
- `parse_docling_file(path)`: opt-in Docling DocumentConverter adapter with page numbers and available actual bounding boxes; raw PDF character offsets are never fabricated. Docling is not required for the offline CI suite.
- `evaluation.benchmark`: human-reviewable answer spans, independent integrity checks, deterministic lexical-ranking baseline, Recall@k, MRR, binary nDCG, oracle evidence coverage, inspectable JSON report.
- Lazy optional LangChain and LlamaIndex export adapters and non-destructive contextual text builder.

```bash
python -m textchunker.evaluation.benchmark \
  --dataset benchmarks/datasets/golden_v1.json \
  --strategies fixed recursive hybrid --tokenizer char \
  --chunk-size 128 --chunk-overlap 16 --k 3 \
  --output out/retrieval_benchmark.json
```

This small synthetic/handwritten corpus verifies evaluation machinery, **not a measured production improvement**. See [evidence and metric definitions](docs/evaluation.md) and [architecture notes](docs/architecture-v0.5-v0.6.md).

## V0.4 Structure-aware Hybrid

The `hybrid` strategy consumes canonical DocumentIR blocks: section headings set hard boundaries, while fenced code and tables remain intact whenever they fit the tokenizer budget. Oversized elements can be split with an `ATOMIC_OVERFLOW` warning, or rejected by policy. It never invents original source bytes or rendered document layout positions.

```bash
python -m pip install -e ".[token,structure,dev]"
python -m textchunker.cli --config configs/hybrid.yaml \
    --input sample.txt --output out/hybrid.jsonl --strategy hybrid --tokenizer char
```

See [V0.4 architecture and acceptance rules](docs/architecture-v0.4.md).

## CLI

```bash
python -m textchunker.cli --config configs/default.yaml \
    --input sample.txt --output out/chunks.jsonl \
    --strategy recursive --tokenizer char --chunk-size 128 --chunk-overlap 16

# Multiple documents, one JSONL file (output cleared once per run).
python -m textchunker.cli --input docs/ --output out/docs.jsonl \
    --strategy recursive --tokenizer tiktoken:cl100k_base
```

The CLI also accepts `--strategy fixed|recursive|structure|hybrid|semantic|llm`, `--max-chunks`, `--stats`, `--visualize` and `--config`; run `python -m textchunker.cli --help` for the full list. The default YAML enables preview output; override it with your own configuration for automated runs.

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
    → chunker registry (fixed/recursive/structure/hybrid/semantic/llm)
    → ChunkRecord + source locations
    → independent correctness evaluator
    → JSONL / TXT / reports
```

`DocumentIR.text` is the canonical Python Unicode string and `Span(start,end)` is half-open in **Unicode code points, not UTF-8 bytes**. Plain TXT/Markdown use exact source-character mapping; HTML/PDF extraction sets raw source spans to null. HTML DOM block IDs are not raw character offsets. PDF retains 1-based page anchors, not page geometry.

## Code layout

```text
textchunker/
  core/          # DocumentIR, spans, contracts, validation and budgets
  parsers/       # text, MarkdownIt, HTML DOM, PDF pages, native Docling
  tokenizers/    # char, legacy, tiktoken, HF adapters
  chunkers/      # six algorithms
  pipeline/      # DocumentIR-first runner
  evaluation/    # correctness, structure, retrieval, benchmark, reporting
  providers/     # optional embedding and LLM providers
  adapters/      # optional LangChain and LlamaIndex exports
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

See [documentation index](docs/INDEX.md), [CLI and configuration](docs/usage.md), [V0.4 architecture](docs/architecture-v0.4.md), [development and CI](docs/development.md), and [legacy strategy research](docs/strategy-guide.md).

License: [MIT](LICENSE).
