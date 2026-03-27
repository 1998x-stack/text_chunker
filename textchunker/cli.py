from __future__ import annotations

import os
import time

from loguru import logger

from .config import load_yaml, to_project_config, build_argparser, merge_cli
from .settings import Settings
from .log_config import setup_logging
from .stats import StatsCollector
from .readers import load_inputs
from .factory import create_chunker
from .export import save_jsonl, save_txt_dir
from .visualization import show_chunks
from .utils import count_tokens
from .chunkers import *  # noqa: F401  # trigger registration


def main() -> None:
    """CLI entry point: read -> chunk -> visualize -> export."""
    parser = build_argparser()
    args = parser.parse_args()

    raw_cfg = load_yaml(args.config)
    cfg = to_project_config(raw_cfg)
    cfg = merge_cli(cfg, args)

    # Initialize settings and logging
    settings = Settings.from_env_and_yaml(raw_cfg)
    if getattr(args, 'log_level', None):
        settings.log_level = args.log_level
    setup_logging(settings)

    # Initialize stats
    stats = StatsCollector.instance()
    stats.reset()
    stats_enabled = getattr(args, 'stats', False) or settings.stats_enabled

    input_path = cfg.io.get("input_path")
    output_path = cfg.io.get("output_path")
    visualize = bool(cfg.project.get("visualize", False))
    export_format = cfg.project.get("export_format", "jsonl")

    logger.info("Strategy = {}", cfg.strategy.name)
    chunker = create_chunker(cfg.strategy)

    for doc in load_inputs(input_path):
        logger.info("Processing: {}", doc.path)
        t0 = time.perf_counter()
        chunks = chunker.chunk(doc.text)
        elapsed = time.perf_counter() - t0

        if stats_enabled:
            doc_tokens = count_tokens(doc.text)
            stats.record_document(
                char_count=len(doc.text), token_count=doc_tokens,
                chunk_count=len(chunks), processing_time=elapsed,
            )
            stats.record_chunks(
                chunk_sizes=[len(c.text) for c in chunks],
                chunk_tokens=[count_tokens(c.text) for c in chunks],
                max_size=int(cfg.strategy.common.get("chunk_size", 512)),
            )

        if visualize:
            show_chunks(chunks, title=os.path.basename(doc.path))

        if output_path:
            if export_format in {"jsonl", "both"}:
                save_jsonl(output_path, chunks, source=doc.path)
                logger.info("Written JSONL to {}", output_path)
            if export_format in {"txt_dir", "both"}:
                outdir = os.path.splitext(output_path)[0] + "_parts"
                save_txt_dir(outdir, chunks, prefix=os.path.basename(doc.path))
                logger.info("Written TXT parts to {}", outdir)

    if stats_enabled:
        from rich.console import Console
        summary = stats.summary()
        console = Console()
        console.print("\n[bold]Pipeline Statistics[/bold]")
        for k, v in summary.items():
            if isinstance(v, float):
                console.print(f"  {k}: {v:.3f}")
            else:
                console.print(f"  {k}: {v}")

        stats_dir = getattr(args, 'stats_dir', None) or settings.stats_dir
        stats.save_run(
            os.path.join(stats_dir, f"{cfg.strategy.name}_run.json"),
            metadata={"strategy": cfg.strategy.name},
        )


if __name__ == "__main__":
    main()
