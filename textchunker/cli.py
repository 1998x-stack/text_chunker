"""Compatible command-line entrypoint with per-document verification."""

from __future__ import annotations

import os
import time

from loguru import logger

from .config import build_argparser, load_yaml, merge_cli, to_project_config
from .core.validation import validate_chunks
from .export import save_jsonl, save_txt_dir
from .factory import create_chunker
from .log_config import setup_logging
from .readers import load_inputs
from .settings import Settings
from .stats import StatsCollector
from .utils import count_tokens
from .visualization import show_chunks
from .chunkers import *  # noqa: F401,F403 - trigger strategy registration


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
    input_path = cfg.io.get("input_path")
    output_path = cfg.io.get("output_path")
    visualize = bool(cfg.project.get("visualize", False))
    output_format = cfg.project.get("export_format", "jsonl")
    size = int(cfg.strategy.common.get("chunk_size", 512))
    max_chunks = cfg.strategy.common.get("max_chunks")

    chunker = create_chunker(cfg.strategy)
    if output_path and output_format in {"jsonl", "both"}:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Truncate once per run, never once per document.
        with open(output_path, "w", encoding="utf-8"):
            pass

    for doc in load_inputs(input_path):
        logger.info("Processing: {}", doc.path)
        started = time.perf_counter()
        chunks = chunker.chunk(doc.text)
        elapsed = time.perf_counter() - started

        # Preserve the current character-based fixed strategy's unit.
        fixed_char_mode = (
            cfg.strategy.name == "fixed"
            and (
                not cfg.strategy.fixed.get("use_tokens", True)
                or __import__("textchunker.utils", fromlist=["_enc"])._enc is None
            )
        )
        measure = len if fixed_char_mode else count_tokens
        validate_chunks(
            doc.text, chunks, count=measure, max_size=size,
            require_full_coverage=max_chunks is None,
        )

        if stats_enabled:
            stats.record_document(
                char_count=len(doc.text), token_count=count_tokens(doc.text),
                chunk_count=len(chunks), processing_time=elapsed,
            )
            stats.record_chunks(
                chunk_sizes=[len(chunk.text) for chunk in chunks],
                chunk_tokens=[count_tokens(chunk.text) for chunk in chunks],
                max_size=size,
            )
            stats.record_coverage(len(doc.text), chunks)

        if visualize:
            show_chunks(chunks, title=os.path.basename(doc.path))
        if output_path:
            if output_format in {"jsonl", "both"}:
                save_jsonl(output_path, chunks, source=doc.path, mode="a")
            if output_format in {"txt_dir", "both"}:
                directory = os.path.splitext(output_path)[0] + "_parts"
                save_txt_dir(directory, chunks, prefix=os.path.basename(doc.path))

    if stats_enabled:
        from rich.console import Console
        console = Console()
        console.print("\n[bold]Pipeline Statistics[/bold]")
        for key, value in stats.summary().items():
            console.print(f"  {key}: {value:.3f}" if isinstance(value, float)
                          else f"  {key}: {value}")
        target = args.stats_dir or settings.stats_dir
        stats.save_run(
            os.path.join(target, f"{cfg.strategy.name}_run.json"),
            metadata={"strategy": cfg.strategy.name},
        )


if __name__ == "__main__":
    main()
