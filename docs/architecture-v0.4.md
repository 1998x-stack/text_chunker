# V0.4 — Structure-aware Hybrid Chunking

Status: V0.4 structural baseline in the consolidated V0.4–V0.6 PR. Native Docling adapter and independent retrieval evaluation are documented separately.

## Architecture

```text
TXT / Markdown / HTML / PDF-page text
    → parser: canonical DocumentIR + non-overlapping structural blocks
    → SourceMap: exact for TXT/Markdown; extracted-only for HTML/PDF
    → HybridChunker.chunk_document(DocumentIR)
       - hard heading/section boundaries
       - greedy block packing with explicit Tokenizer
       - table/code atomicity when within size budget
       - recursive fallback on oversized blocks or fail-closed policy
       - overlap allowed only within safe structural boundaries
    → independent correctness evaluator
       - source-spans cover canonical text exactly
       - source text equals mapped slice; no missing intervals
       - per-chunk token budget
    → independent structure evaluator
       - heading-crossing error
       - fitting table/code split error
       - oversized table/code split warning
    → ChunkingResult / provenance-rich JSONL
```

## DocumentIR + parser contract

- Span(start,end) is Python Unicode code point indexing into canonical DocumentIR.text (end excluded).
- Markdown uses MarkdownIt CommonMark tokens with GFM table tokens, retaining full original source text (including fences, blank lines and separators). Top-level blocks include headings, paragraphs, code, tables, lists, quotes and explicit unparsed whitespace intervals.
- A minimal line-block parser remains the backward-compatible fallback for non-Hybrid callers. Hybrid refuses Markdown if markdown-it-py is absent, because it cannot guarantee table/code atomicity.
- HTML uses BeautifulSoup and outermost semantic DOM blocks (headings, paragraphs, lists, blockquotes, pre and tables). It normalizes text and stores a stable per-document DOM block ID; source_span=None, so original HTML character offsets are **unknown**.
- PDF uses pypdf-extracted text and page anchors. No glyph coordinate, bounding box, page geometry, OCR or Docling layout is claimed. The CLI retains PDF pages after one read.
- HTML tables are canonicalized as row-separated, pipe-delimited text; the source text in ChunkingResult is exact **canonical text**, not the original HTML markup. More granular cell provenance is future work.

## API

```python
from textchunker.parsers import parse_markdown
from textchunker.pipeline import run_chunking
from textchunker.tokenizers import CharacterTokenizer
from textchunker.types import StrategyConfig

ir = parse_markdown("# Guide\n\n~~~py\nprint(1)\n~~~\n", "guide.md")
cfg = StrategyConfig(
    name="hybrid",
    common={"chunk_size": 100, "chunk_overlap": 12},
    hybrid={"oversized_element": "split"},
)
result = run_chunking(ir, cfg, CharacterTokenizer())
assert result.report.ok and result.complete
```

The legacy create_chunker(cfg).chunk(str) API remains for the original five strategies. A HybridChunker constructed directly also accepts a Markdown string, but structured DocumentIR is recommended.

## Atomicity and overflow policy

For each table/code block within budget, at least one output chunk must contain its whole canonical Span. The independent evaluator rejects fitting blocks split across chunks. Oversized blocks cannot satisfy both atomicity and max tokens, so:

- `hybrid.oversized_element: split`: recursively split at line/word/code point boundaries and report `ATOMIC_OVERFLOW` (warning) with metadata `atomic_overflow=True`.
- `hybrid.oversized_element: error`: refuse an oversized table/code block before returning incorrect results.
- No synthesized/repeated Markdown table headers, duplicate code fences or other invented source text are emitted; they would break offset integrity.
- Overlap never bridges a new heading, nor an intact code/table block. Only adjacent fragments of the same oversized atom may overlap under budget.

Under `max_chunks`, missing trailing input results in `result.complete=False`, as in V0.3.

## Evaluation and acceptance

Run Python 3.11 CI with extra dependencies; independent tests assert source coverage, heading isolation, table/code atomicity, overflow warning/fail-closed, HTML DOM block identifiers, PDF page anchors and multi-document CLI output. Tests involving live LLM/embeddings or an installed Docling parser are **not** part of this release.

Known limits: nested code fences inside quote/list wrappers are represented by the parent block; MarkdownIt gives stronger top-level block recognition, not a fully normalized inline AST. For deeply nested HTML tables/lists, the outer semantic block owns the children and provenance remains at that level. Further granular AST and live Docling conversion/complex layout tests need independent validation. An optional native DoclingDocument adapter now preserves provided provenance; see V0.5–V0.6 notes.
