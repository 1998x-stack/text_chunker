# v0.2.1 — Lossless chunking correctness baseline

Status: implementation proposal, PR checks required before merging.

## Compatibility strategy

The public package, CLI, YAML configuration and existing strategy names remain
unchanged. This release introduces internal shared primitives under
textchunker.core, but does not move legacy public imports.

## Contract

A chunk carries half-open Unicode-character offsets into the canonical text
provided to the chunker. The following must hold in full-document mode:

- chunk.text == source[chunk.start:chunk.end]
- spans are ordered and cover every character of the source
- overlaps may duplicate source coverage but do not create missing intervals
- measured chunk size does not exceed the configured budget
- invalid configurations fail before processing
- LLM-proposed spans are untrusted and validated before creating chunks

max_chunks explicitly enables a partial result, so total coverage is not
required in that mode. The existing list-returning API does not yet report a
structured 'complete' flag; that is scheduled for v0.3.

## Byte offsets versus source offsets

These offsets refer to Python Unicode code points, not UTF-8 bytes or PDF
coordinates. For parsed PDF and HTML input, offsets currently refer to the
extracted text view, not the original document layout. v0.4 will introduce
DocumentIR and explicit source maps.

## Tokenization

For compatibility, the legacy count_tokens helper uses cl100k_base when
available and character count when tiktoken is absent. The fixed token mode
never decodes arbitrary token-ID slices that could cut through a UTF-8 code
point. v0.3 will make tokenizer identity and units explicit.

## Metrics

coverage_ratio measures unique character coverage using the union of spans;
it is None when spans were never supplied to StatsCollector.
output_char_ratio preserves the old output/input character-length statistic,
which can exceed 1 when overlap is present.

## Known deferred work

- immutable DocumentIR with structural/source-map provenance
- strict run-scoped metrics instead of singleton
- tokenizer-specific model budgeting
- Markdown AST, HTML DOM and PDF structural parsing
- independent retrieval and golden benchmark harness
- consistent lightweight optional imports in experiment-only modules
- stable document and chunk hashes, atomic run-level output writer
- semantic/LLM provider retries, timeouts and deterministic caching

See the README for the legacy CLI and strategy options.
