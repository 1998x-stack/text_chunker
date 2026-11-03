"""Bridge from canonical DocumentIR to all five existing chunking strategies."""

from __future__ import annotations

from ..core.contracts import ChunkRecord, ChunkingResult
from ..core.models import DocumentIR, chunk_digest
from ..core.spans import Span
from ..evaluation import evaluate_correctness
from ..exceptions import ChunkerError
from ..factory import create_chunker
from ..tokenizers import CharacterTokenizer, LegacyTokenizer, Tokenizer
from ..types import StrategyConfig


def _heading_path(document: DocumentIR, position: int) -> tuple[str, ...]:
    headings: list[tuple[int, str]] = []
    for block in document.blocks:
        if block.span.start > position:
            break
        if block.kind == "heading" and block.heading_level is not None:
            while headings and headings[-1][0] >= block.heading_level:
                headings.pop()
            headings.append((block.heading_level, block.title or ""))
    return tuple(title for _, title in headings)


def run_chunking(
    document: DocumentIR,
    config: StrategyConfig,
    tokenizer: Tokenizer | None = None,
    strict: bool = True,
) -> ChunkingResult:
    """Run a registered strategy with provenance and independent QA.

    If strict, quality contract failures raise before a corrupt result
    can be exported. An explicit max_chunks permits a partial result,
    reported as complete=False and an INCOMPLETE warning.
    """
    if tokenizer is None:
        tokenizer = LegacyTokenizer()
    if config.name == "fixed" and not config.fixed.get("use_tokens", True):
        tokenizer = CharacterTokenizer()

    chunker = create_chunker(config, tokenizer=tokenizer)
    chunks = chunker.chunk(document.text)
    max_size = int(config.common.get("chunk_size", 512))
    report = evaluate_correctness(
        document.text, chunks, tokenizer.count, max_size,
        require_full_coverage=config.common.get("max_chunks") is None,
    )
    if strict and not report.ok:
        problems = ", ".join(
            d.code for d in report.diagnostics if d.severity == "error"
        )
        raise ChunkerError(f"Chunk integrity evaluation failed: {problems}")

    records = []
    for index, chunk in enumerate(chunks):
        valid_span = 0 <= chunk.start <= chunk.end <= len(document.text)
        span = Span(chunk.start, chunk.end) if valid_span else Span(0, 0)
        locations = document.source_map.resolve(span) if valid_span else ()
        records.append(ChunkRecord(
            chunk_id=chunk_digest(document.document_id, span, chunk.text),
            document_id=document.document_id, index=index,
            text=chunk.text, span=span,
            token_count=tokenizer.count(chunk.text),
            source_locations=locations,
            heading_path=_heading_path(document, chunk.start),
            metadata=dict(chunk.meta),
        ))

    return ChunkingResult(
        document_id=document.document_id, strategy=config.name,
        tokenizer_id=tokenizer.name, chunks=tuple(records), report=report,
    )
