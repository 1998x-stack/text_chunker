from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger
from rich.console import Console
from rich.table import Table

from ..chunkers import *  # noqa: F401,F403
from ..config import load_yaml, to_project_config
from ..factory import create_chunker
from ..utils import count_tokens
from .semantic_ablation import load_hf_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-strategy comparison")
    parser.add_argument("--strategies", type=str, nargs="+", default=["fixed", "recursive"])
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-docs", type=int, default=32)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def run_strategy_comparison(args: argparse.Namespace | None = None) -> List[Dict[str, Any]]:
    if args is None:
        args = parse_args()

    base_cfg = to_project_config(load_yaml(args.config))

    texts = load_hf_texts(
        dataset=args.dataset, subset=args.subset, split=args.split,
        text_field=args.text_field, max_docs=args.max_docs,
        min_doc_chars=args.min_doc_chars, seed=args.seed, cache_dir=args.cache_dir,
    )

    results: List[Dict[str, Any]] = []

    for strategy_name in args.strategies:
        logger.info("Evaluating strategy: {}", strategy_name)
        cfg = base_cfg.strategy
        cfg.name = strategy_name

        try:
            chunker = create_chunker(cfg)
        except Exception as e:
            logger.warning("Failed to create chunker for {}: {}", strategy_name, e)
            continue

        total_chunks = 0
        total_chars = 0
        total_tokens = 0
        t0 = time.perf_counter()

        for text in texts:
            chunks = chunker.chunk(text)
            total_chunks += len(chunks)
            for c in chunks:
                total_chars += len(c.text)
                total_tokens += count_tokens(c.text)

        elapsed = time.perf_counter() - t0
        n_docs = len(texts)
        doc_chars = sum(len(t) for t in texts)
        doc_tokens = sum(count_tokens(t) for t in texts)

        results.append({
            "strategy": strategy_name,
            "total_chunks": total_chunks,
            "chunks_per_doc": total_chunks / n_docs if n_docs else 0,
            "avg_chunk_chars": total_chars / total_chunks if total_chunks else 0,
            "avg_chunk_tokens": total_tokens / total_chunks if total_chunks else 0,
            "redundancy_ratio": max(0, (total_tokens - doc_tokens) / doc_tokens) if doc_tokens else 0,
            "coverage_ratio": total_chars / doc_chars if doc_chars else 0,
            "time_seconds": elapsed,
        })

    # Display
    console = Console()
    table = Table(title="Strategy Comparison", show_header=True, header_style="bold cyan")
    table.add_column("Strategy")
    table.add_column("Chunks", justify="right")
    table.add_column("Chunks/Doc", justify="right")
    table.add_column("Avg Chars", justify="right")
    table.add_column("Avg Tokens", justify="right")
    table.add_column("Redundancy", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Time (s)", justify="right")

    for r in results:
        table.add_row(
            r["strategy"], str(r["total_chunks"]),
            f"{r['chunks_per_doc']:.1f}", f"{r['avg_chunk_chars']:.0f}",
            f"{r['avg_chunk_tokens']:.0f}", f"{r['redundancy_ratio']*100:.1f}%",
            f"{r['coverage_ratio']:.2f}x", f"{r['time_seconds']:.2f}",
        )
    console.print(table)

    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("Saved comparison to {}", path)

    return results


def main() -> None:
    run_strategy_comparison()


if __name__ == "__main__":
    main()
