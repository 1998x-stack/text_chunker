from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from loguru import logger
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ..chunkers import *  # noqa: F401,F403
from ..config import load_yaml, to_project_config
from ..factory import create_chunker
from ..utils import count_tokens
from .semantic_ablation import (
    AblationMetrics,
    Scenario,
    load_hf_texts,
    summarize_metrics,
)


def evaluate_recursive_scenario(
    base_chunker,
    scenario: Scenario,
    texts: List[str],
) -> AblationMetrics:
    original_common = copy.deepcopy(base_chunker.cfg.common)
    try:
        base_chunker.cfg.common.update({
            "chunk_size": scenario.chunk_size,
            "chunk_overlap": scenario.chunk_overlap,
        })

        doc_token_counts: List[int] = []
        doc_char_counts: List[int] = []
        chunk_token_counts: List[int] = []
        chunk_char_counts: List[int] = []
        chunk_counts: List[int] = []
        boundary_hits = 0

        for text in texts:
            doc_token_counts.append(count_tokens(text))
            doc_char_counts.append(len(text))
            chunks = base_chunker.chunk(text)
            chunk_counts.append(len(chunks))
            for ch in chunks:
                ctoks = count_tokens(ch.text)
                chunk_token_counts.append(ctoks)
                chunk_char_counts.append(len(ch.text))
                if ctoks >= scenario.chunk_size:
                    boundary_hits += 1

        return summarize_metrics(
            scenario, doc_token_counts, doc_char_counts,
            chunk_token_counts, chunk_char_counts, chunk_counts, boundary_hits,
        )
    finally:
        base_chunker.cfg.common = original_common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursive chunking ablation study")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--max-docs", type=int, default=64)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 512, 800])
    parser.add_argument("--chunk-overlaps", type=int, nargs="+", default=[0, 50, 100])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def run_recursive_ablation(args: argparse.Namespace | None = None) -> List[AblationMetrics]:
    if args is None:
        args = parse_args()

    cfg = to_project_config(load_yaml(args.config)).strategy
    cfg.name = "recursive"

    texts = load_hf_texts(
        dataset=args.dataset, subset=args.subset, split=args.split,
        text_field=args.text_field, max_docs=args.max_docs,
        min_doc_chars=args.min_doc_chars, seed=args.seed, cache_dir=args.cache_dir,
    )

    chunker = create_chunker(cfg)

    scenarios = [
        Scenario(cs, ov, 0.0, 0)
        for cs, ov in itertools.product(args.chunk_sizes, args.chunk_overlaps)
    ]

    logger.info("Running {} recursive scenarios", len(scenarios))
    results: List[AblationMetrics] = []

    with Progress() as progress:
        task = progress.add_task("Running recursive ablation...", total=len(scenarios))
        for scenario in scenarios:
            metrics = evaluate_recursive_scenario(chunker, scenario, texts)
            results.append(metrics)
            progress.advance(task)

    console = Console()
    table = Table(title="Recursive Ablation Results", show_header=True, header_style="bold magenta")
    table.add_column("chunk_size", justify="right")
    table.add_column("overlap", justify="right")
    table.add_column("chunks/doc", justify="right")
    table.add_column("avg_chunk_tok", justify="right")
    table.add_column("p95_tok", justify="right")
    table.add_column("redundancy", justify="right")
    table.add_column("coverage", justify="right")
    for r in results:
        table.add_row(
            str(r.scenario.chunk_size), str(r.scenario.chunk_overlap),
            f"{r.avg_chunks_per_doc:.2f}", f"{r.avg_chunk_tokens:.1f}",
            f"{r.p95_chunk_tokens:.1f}", f"{r.redundancy_ratio*100:.1f}%",
            f"{r.coverage_ratio:.2f}x",
        )
    console.print(table)

    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=["chunk_size", "chunk_overlap", "avg_chunks_per_doc", "avg_chunk_tokens", "p95_chunk_tokens", "redundancy_ratio", "coverage_ratio"])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    "chunk_size": r.scenario.chunk_size, "chunk_overlap": r.scenario.chunk_overlap,
                    "avg_chunks_per_doc": f"{r.avg_chunks_per_doc:.2f}",
                    "avg_chunk_tokens": f"{r.avg_chunk_tokens:.1f}",
                    "p95_chunk_tokens": f"{r.p95_chunk_tokens:.1f}",
                    "redundancy_ratio": f"{r.redundancy_ratio:.4f}",
                    "coverage_ratio": f"{r.coverage_ratio:.4f}",
                })
        logger.info("Saved results to {} and {}", path, csv_path)

    return results


def main() -> None:
    run_recursive_ablation()


if __name__ == "__main__":
    main()
