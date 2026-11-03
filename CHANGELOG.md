# Change log

The changelog is limited to changes present in code. Historical benchmark results are not treated as acceptance evidence.

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
