# CLI, configuration and output schema (V0.3)

## Basic CLI

```bash
python -m textchunker.cli --help
python -m textchunker.cli --config configs/default.yaml --input sample.txt \
  --output out/chunks.jsonl --strategy recursive --tokenizer char \
  --chunk-size 512 --chunk-overlap 80
```

`--strategy` accepts fixed, recursive, structure, semantic, llm. Use `--config` to set model-specific options and input/output defaults. `--max-chunks N` intentionally returns an incomplete prefix if the limit is reached. `--stats` records processing statistics, `--visualize` prints Rich chunk previews.

Do **not** insert inline comments after a backslash-continuation in a shell command.

## Minimal YAML

```yaml
project:
  export_format: jsonl
  visualize: false
io:
  input_path: ./sample.txt
  output_path: ./out/chunks.jsonl
strategy:
  name: recursive
  common:
    chunk_size: 512
    chunk_overlap: 80
    max_chunks: null
    tokenizer: tiktoken:cl100k_base
  recursive:
    separators: ["\n\n", "\n", " ", ""]
```

Tokenizer selection precedence: CLI `--tokenizer` → `strategy.common.tokenizer` → `legacy`. Fixed `use_tokens: false` overrides this and always uses character units.

## JSONL schema

Each output line includes `id, text, start, end, meta, source` (legacy compatibility), plus `chunk_id, document_id, token_count, heading_path, source_locations, complete, tokenizer_id`.

`source_locations` contains canonical spans and optional original source spans, page_number, block_id and mapping type:
- `mapping=exact` for character-preserving plain text/Markdown.
- `mapping=extracted` for HTML/PDF text where original character positions are **not** known.
- PDF page_number is 1-based; separators inserted between page texts are assigned to the preceding page's canonical interval.

An empty input produces no chunk lines; the document's empty status is not represented by a placeholder JSONL row. A future run-level manifest will address this.

## Compatibility and non-goals

The old `create_chunker(cfg).chunk(str)` API still works. V0.3 is not a retrieval evaluation product, HTML DOM span locator or full PDF layout parser. Historic benchmark “coverage” is not unique source coverage.

## V0.4 Hybrid and structural provenance

```bash
python -m textchunker.cli --config configs/hybrid.yaml --input doc.md \
  --output out/doc.jsonl --strategy hybrid --tokenizer tiktoken:cl100k_base
```

With HTML inputs, the CLI now passes raw HTML to the DOM parser and exports normalized canonical text plus per-block `dom:<index>:<tag>` anchors; **no raw HTML character spans** are invented. PDF page text is extracted once and page positions remain one-based. YAML `hybrid.oversized_element` chooses `split` (default, warning) or `error` (fail closed) for code/table over-budget elements.

With Markdown inputs, the Hybrid strategy requires the optional `markdown-it-py` package under `[structure]`. If not installed, it rejects the operation rather than claiming structural guarantees based on a weaker fallback.
