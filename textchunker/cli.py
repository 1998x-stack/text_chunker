"""CLI with DocumentIR provenance and independent correctness gates."""

from __future__ import annotations

import os
import time

from loguru import logger

from .config import build_argparser, load_yaml, merge_cli, to_project_config
from .export import save_result_jsonl, save_txt_dir
from .log_config import setup_logging
from .parsers import parse_html, parse_extracted_text, parse_markdown, parse_pdf_pages, parse_text
from .parsers.docling_adapter import parse_docling_file
from .pipeline import run_chunking
from .readers import load_inputs
from .settings import Settings
from .stats import StatsCollector
from .tokenizers import create_tokenizer
from .types import Chunk
from .utils import count_tokens
from .visualization import show_chunks


def _parse_input(doc):
    ext = os.path.splitext(doc.path)[1].lower()
    if ext == ".md":
        return parse_markdown(doc.text, doc.path)
    if doc.meta.get("format") == "html":
        return (parse_html(doc.text, doc.path)
                if doc.meta.get("raw_html")
                else parse_extracted_text(doc.text, doc.path, "html"))
    if doc.meta.get("format") == "docling":
        return parse_docling_file(doc.meta["source_path"])
    if doc.meta.get("format") == "pdf":
        if "pages" in doc.meta:
            return parse_pdf_pages(doc.meta["pages"], doc.path)
        # Support external FileDoc callers that provide extracted PDF text.
        return parse_extracted_text(doc.text, doc.path, "pdf")
    return parse_text(doc.text, doc.path)


def main() -> None:
    args = build_argparser().parse_args()
    raw_cfg = load_yaml(args.config)
    cfg = merge_cli(to_project_config(raw_cfg), args)
    settings = Settings.from_env_and_yaml(raw_cfg)
    if args.log_level:
        settings.log_level = args.log_level
    setup_logging(settings)

    stats = StatsCollector.instance()
    stats.reset()
    stats_enabled = args.stats or settings.stats_enabled
    output_path = cfg.io.get("output_path")
    output_format = cfg.project.get("export_format", "jsonl")
    visualize = bool(cfg.project.get("visualize", False))
    tokenizer_name = args.tokenizer or cfg.strategy.common.get("tokenizer", "legacy")
    tokenizer = create_tokenizer(tokenizer_name)

    if output_path and output_format in {"jsonl", "both"}:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8"):
            pass

    for file_doc in load_inputs(cfg.io.get("input_path"),
                                preserve_structure=True,
                                pdf_engine=args.pdf_parser):
        document = _parse_input(file_doc)
        logger.info("Processing: {}", document.source_uri)
        started = time.perf_counter()
        result = run_chunking(document, cfg.strategy, tokenizer=tokenizer,
                              strict=True)
        elapsed = time.perf_counter() - started
        chunks = [
            Chunk(
                id=item.index, text=item.text, start=item.span.start,
                end=item.span.end, meta=dict(item.metadata),
            )
            for item in result.chunks
        ]
        if stats_enabled:
            stats.record_document(
                char_count=len(document.text),
                token_count=count_tokens(document.text),
                chunk_count=len(chunks), processing_time=elapsed,
            )
            stats.record_chunks(
                chunk_sizes=[len(chunk.text) for chunk in chunks],
                chunk_tokens=[count_tokens(chunk.text) for chunk in chunks],
                max_size=int(cfg.strategy.common.get("chunk_size", 512)),
            )
            stats.record_coverage(len(document.text), chunks)

        if visualize:
            show_chunks(chunks, title=os.path.basename(document.source_uri))
        if output_path:
            if output_format in {"jsonl", "both"}:
                save_result_jsonl(output_path, result, source=document.source_uri,
                                  mode="a")
            if output_format in {"txt_dir", "both"}:
                directory = os.path.splitext(output_path)[0] + "_parts"
                save_txt_dir(directory, chunks,
                             prefix=os.path.basename(document.source_uri))

    if stats_enabled:
        from rich.console import Console
        console = Console()
        console.print("\n[bold]Pipeline Statistics[/bold]")
        for key, value in stats.summary().items():
            console.print(f"  {key}: {value:.3f}" if isinstance(value, float)
                          else f"  {key}: {value}")
        destination = args.stats_dir or settings.stats_dir
        stats.save_run(
            os.path.join(destination, f"{cfg.strategy.name}_run.json"),
            metadata={"strategy": cfg.strategy.name,
                      "tokenizer": tokenizer.name},
        )


if __name__ == "__main__":
    main()
