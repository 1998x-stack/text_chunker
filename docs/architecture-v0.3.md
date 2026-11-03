# V0.3 — DocumentIR, Tokenizer and independently verified chunk results

This is a *stacked* change based on v0.2.1 PR #3. It adds new
DocumentIR-first APIs while leaving the old strategy names and direct
chunker.chunk(str) interface available.

## Data contract

Canonical text is a Python Unicode string. All Span offsets are half-open
code-point indexes into DocumentIR.text. **They are not UTF-8 byte offsets**.

SourceMap entries cover canonical text contiguously. Each maps to:
- exact original-character offsets for untouched text and Markdown, or
- extracted text with source_span=None (HTML and PDF).
- PDF extracted pages additionally retain page_number and block_id.

A PDF page number is not a PDF glyph coordinate, bounding box, or original
raw byte offset. HTML extraction likewise does not claim precise raw HTML
element offsets. Proper page geometry and Markdown AST parsing belong to V0.4.

DocumentBlock provides document-level structural hints. Markdown currently
offers line-level heading and fenced-code recognition, **not a complete AST**.

## New Python API

Use the high-level pipeline to get stable identities, bounded tokens, source
locations and an independent quality report:

    from textchunker.parsers import parse_markdown
    from textchunker.pipeline import run_chunking
    from textchunker.tokenizers import TiktokenTokenizer
    from textchunker.types import StrategyConfig

    doc = parse_markdown("# Topic\nText content.\n", "memory://example.md")
    cfg = StrategyConfig(
        name="recursive",
        common={"chunk_size": 256, "chunk_overlap": 32},
    )
    result = run_chunking(doc, cfg, tokenizer=TiktokenTokenizer("cl100k_base"))

    assert result.report.ok
    assert result.complete
    for record in result.chunks:
        print(record.chunk_id, record.span, record.source_locations)

Tokenizers: "legacy" (backwards-compatible tiktoken-or-character fallback),
"char" (strict code-point length), "tiktoken:<encoding>" or "hf:<model>".
Optional providers are lazily loaded when chosen.

The fixed strategy's explicit use_tokens=False always chooses character
units. All other strategies and fixed token mode consume the injected
tokenizer.count(text) for budget checks.

## CLI

    python -m textchunker.cli --input sample.txt --strategy recursive \
        --tokenizer tiktoken:cl100k_base --output out/chunks.jsonl

JSONL keeps the existing id/text/start/end/meta/source keys and adds stable
chunk_id, document_id, token_count, tokenizer_id, source_locations,
heading_path and a per-document complete marker. The output file is
truncated once per CLI run and appended across documents.

## Quality gates

Independent evaluation checks:
- input coverage (union of source spans, not output-length ratio);
- source text/offset equality;
- valid ordered and bounded spans;
- tokenizer-specific max size;
- explicit incomplete output when max_chunks truncates a document.

strict=True (default) aborts the pipeline if an error is detected. Partial
output requested via max_chunks reports incomplete=True? **No**:
result.complete=False with an INCOMPLETE warning. Never treat partial output
as a complete index.

## Known limitations

- Legacy chunker chunks are mutable; new ChunkRecord results are immutable
  dataclasses but metadata Mapping remains shallow.
- PDF page extraction in the CLI currently re-reads PDF pages after the
  legacy reader; V0.4 will eliminate duplicate work.
- Full Markdown/HTML AST and bounding-box provenance are intentionally deferred.
- Historical ablation metrics still use old output/input length ratios;
  these are not source coverage and must not be compared to new coverage.
- No retrieval-quality, embedding-provider or model-performance claims.
- External HF tokenizers may download models; never select them silently.
- Source URIs should be stable logical IDs when deterministic IDs across
  machines or environments matter.
