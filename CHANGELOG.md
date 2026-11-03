# Change log

The changelog is limited to changes present in code. Historical benchmark results are not treated as acceptance evidence.

## 0.4.0 — Structure-aware Hybrid chunking

- MarkdownIt CommonMark/GFM table token-backed structural blocks, with exact original Unicode source offsets, fenced-code and nested-list awareness.
- DOM-aware HTML extraction into typed blocks and per-block provenance identifiers; raw DOM character offsets remain intentionally unknown.
- New `hybrid` strategy: strict heading boundaries, budget-first block packing, code/table atomicity, bounded overlaps and deterministic over-budget fallback.
- Independent structural evaluation with `ATOMIC_SPLIT`, `ATOMIC_OVERFLOW` and `HEADING_CROSSING` diagnostics.
- Structure-preserving CLI ingestion without re-reading PDF pages, with old reader API retained.
- Offline regression, integration and CLI E2E coverage for Markdown/HTML/PDF provenance.

## 0.3.0 — DocumentIR and independent correctness verification

- Canonical `DocumentIR`, `DocumentBlock`, `SourceMap`, stable document/chunk IDs.
- New `ChunkingResult`, `ChunkRecord`, explicit complete flag and independent quality diagnostics.
- Tokenizer injection across five strategies; char, legacy, tiktoken and HF adapters.
- Text/Markdown exact character mapping; HTML extracted-text mapping and PDF page anchors with unknown raw offsets.
- CLI `--tokenizer` option and provenance-rich JSONL, retaining legacy keys.
- New unit/property/integration tests. Structural AST chunking and retrieval benchmarks remain pending.

## 0.2.1 — Chunk integrity baseline

- Fixed/recursive/structure splitting correctness and bounded overlap.
- Preserve Markdown preamble and all recursive separators.
- LLM window processing and untrusted span fallback.
- Multi-document JSONL append and union-of-spans unique coverage metric.
- Linux CI and regression tests.

## 0.2.0 — Legacy baseline

Original five strategies and experiment framework. Historical benchmark “coverage” included overlap output length and is not comparable to new unique-span coverage.
