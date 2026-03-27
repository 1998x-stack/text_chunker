from __future__ import annotations

import argparse
import copy
import itertools
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from loguru import logger

try:
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from ..chunkers import *  # noqa: F401,F403 - ensure chunkers register themselves
from ..config import load_yaml, to_project_config
from ..factory import create_chunker
from ..utils import count_tokens


@dataclass(frozen=True)
class Scenario:
    """组合一个语义分块实验的关键参数。"""

    chunk_size: int
    chunk_overlap: int
    min_similarity: float
    sentence_window: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "min_similarity": self.min_similarity,
            "sentence_window": self.sentence_window,
        }


@dataclass
class AblationMetrics:
    """针对一个实验设定的统计指标。"""

    scenario: Scenario
    total_docs: int
    total_chunks: int
    avg_chunks_per_doc: float
    avg_doc_tokens: float
    avg_chunk_tokens: float
    avg_chunk_chars: float
    p95_chunk_tokens: float
    std_chunk_tokens: float
    redundancy_ratio: float
    boundary_rate: float
    coverage_ratio: float

    def to_dict(self) -> Dict[str, float]:
        payload = asdict(self)
        payload["scenario"] = self.scenario.to_dict()
        return payload


def summarize_metrics(
    scenario: Scenario,
    doc_token_counts: Sequence[int],
    doc_char_counts: Sequence[int],
    chunk_token_counts: Sequence[int],
    chunk_char_counts: Sequence[int],
    chunk_counts: Sequence[int],
    boundary_hits: int
) -> AblationMetrics:
    """根据分块统计生成评估指标。"""

    total_docs = len(doc_token_counts)
    total_chunks = len(chunk_token_counts)

    if total_docs == 0:
        raise ValueError("No documents were provided for metric summarisation.")

    doc_tokens_sum = float(sum(doc_token_counts))
    doc_chars_sum = float(sum(doc_char_counts))
    chunk_tokens_sum = float(sum(chunk_token_counts))
    chunk_chars_sum = float(sum(chunk_char_counts))

    avg_chunks_per_doc = float(sum(chunk_counts)) / total_docs if total_docs else 0.0
    avg_doc_tokens = doc_tokens_sum / total_docs if doc_tokens_sum else 0.0

    if total_chunks:
        chunk_tokens_arr = np.array(chunk_token_counts, dtype=float)
        chunk_chars_arr = np.array(chunk_char_counts, dtype=float)
        avg_chunk_tokens = float(chunk_tokens_arr.mean())
        avg_chunk_chars = float(chunk_chars_arr.mean())
        p95_chunk_tokens = float(np.percentile(chunk_tokens_arr, 95))
        std_chunk_tokens = float(chunk_tokens_arr.std(ddof=0))
    else:
        avg_chunk_tokens = 0.0
        avg_chunk_chars = 0.0
        p95_chunk_tokens = 0.0
        std_chunk_tokens = 0.0

    redundancy_ratio = 0.0
    if doc_tokens_sum > 0:
        redundancy_ratio = max(0.0, (chunk_tokens_sum - doc_tokens_sum) / doc_tokens_sum)

    coverage_ratio = 0.0
    if doc_chars_sum > 0:
        coverage_ratio = chunk_chars_sum / doc_chars_sum

    boundary_rate = float(boundary_hits) / total_chunks if total_chunks else 0.0

    return AblationMetrics(
        scenario=scenario,
        total_docs=total_docs,
        total_chunks=total_chunks,
        avg_chunks_per_doc=avg_chunks_per_doc,
        avg_doc_tokens=avg_doc_tokens,
        avg_chunk_tokens=avg_chunk_tokens,
        avg_chunk_chars=avg_chunk_chars,
        p95_chunk_tokens=p95_chunk_tokens,
        std_chunk_tokens=std_chunk_tokens,
        redundancy_ratio=redundancy_ratio,
        boundary_rate=boundary_rate,
        coverage_ratio=coverage_ratio,
    )


def load_hf_texts(
    dataset: str,
    subset: str | None,
    split: str,
    text_field: str,
    max_docs: int | None,
    min_doc_chars: int,
    seed: int,
    cache_dir: str | None = None,
) -> List[str]:
    """从 Hugging Face Datasets 拉取文本样本。"""

    if load_dataset is None:
        raise ImportError("datasets>=2.20.0 is required for Hugging Face ablations. Install with `pip install -e .[experiments]`.")

    logger.info(
        "Loading dataset=%s subset=%s split=%s (max_docs=%s)",
        dataset,
        subset,
        split,
        max_docs,
    )
    kwargs: Dict[str, str | int] = {"split": split}
    if subset:
        kwargs["name"] = subset
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    ds = load_dataset(dataset, **kwargs)

    if max_docs is not None and max_docs < len(ds):
        ds = ds.shuffle(seed=seed).select(range(max_docs * 4))  # 倍数以提高过滤后的可用性

    texts: List[str] = []
    for record in ds:
        text = str(record.get(text_field, "") or "").strip()
        if len(text) < min_doc_chars:
            continue
        texts.append(text)
        if max_docs is not None and len(texts) >= max_docs:
            break

    if not texts:
        raise ValueError(
            "No documents satisfied the filtering criteria. Try lowering --min-doc-chars."
        )

    logger.info("Loaded %d documents after filtering", len(texts))
    return texts



def evaluate_scenario(
    base_chunker,
    scenario: Scenario,
    texts: Iterable[str],
) -> AblationMetrics:
    """在给定语料上评估单个分块场景。"""

    # 暂存原配置并覆盖
    original_common = copy.deepcopy(base_chunker.cfg.common)
    original_semantic = copy.deepcopy(base_chunker.cfg.semantic)
    try:
        base_chunker.cfg.common.update(
            {
                "chunk_size": scenario.chunk_size,
                "chunk_overlap": scenario.chunk_overlap,
            }
        )
        base_chunker.cfg.semantic.update(
            {
                "min_similarity": scenario.min_similarity,
                "sentence_window": scenario.sentence_window,
            }
        )

        doc_token_counts: List[int] = []
        doc_char_counts: List[int] = []
        chunk_token_counts: List[int] = []
        chunk_char_counts: List[int] = []
        chunk_counts: List[int] = []
        boundary_hits = 0

        for text in texts:
            tokens = count_tokens(text)
            doc_token_counts.append(tokens)
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
            scenario,
            doc_token_counts,
            doc_char_counts,
            chunk_token_counts,
            chunk_char_counts,
            chunk_counts,
            boundary_hits,
        )
    finally:
        base_chunker.cfg.common = original_common
        base_chunker.cfg.semantic = original_semantic


def render_table(results: Sequence[AblationMetrics]) -> Table:
    """将实验结果渲染为 Rich 表格。"""

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("chunk_size", justify="right")
    table.add_column("overlap", justify="right")
    table.add_column("min_sim")
    table.add_column("window", justify="right")
    table.add_column("chunks/doc", justify="right")
    table.add_column("avg_chunk_tok", justify="right")
    table.add_column("p95_tok", justify="right")
    table.add_column("redundancy", justify="right")
    table.add_column("boundary", justify="right")
    table.add_column("coverage", justify="right")

    for r in results:
        table.add_row(
            f"{r.scenario.chunk_size}",
            f"{r.scenario.chunk_overlap}",
            f"{r.scenario.min_similarity:.2f}",
            str(r.scenario.sentence_window),
            f"{r.avg_chunks_per_doc:.2f}",
            f"{r.avg_chunk_tokens:.1f}",
            f"{r.p95_chunk_tokens:.1f}",
            f"{r.redundancy_ratio*100:.1f}%",
            f"{r.boundary_rate*100:.1f}%",
            f"{r.coverage_ratio:.2f}x",
        )

    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic chunking ablation study")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--subset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--config", type=str, default="configs/semantic.yaml")
    parser.add_argument("--max-docs", type=int, default=64)
    parser.add_argument("--min-doc-chars", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[400, 600, 800])
    parser.add_argument("--chunk-overlaps", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--min-sims", type=float, nargs="+", default=[0.58, 0.62, 0.66])
    parser.add_argument("--sentence-windows", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    return parser.parse_args()


def run_semantic_ablation(args: argparse.Namespace | None = None) -> List[AblationMetrics]:
    """执行 Hugging Face 文本上的语义分块消融实验。"""

    if args is None:
        args = parse_args()

    cfg = to_project_config(load_yaml(args.config)).strategy

    texts = load_hf_texts(
        dataset=args.dataset,
        subset=args.subset,
        split=args.split,
        text_field=args.text_field,
        max_docs=args.max_docs,
        min_doc_chars=args.min_doc_chars,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )

    logger.info("Initialising semantic chunker for ablation study")
    chunker = create_chunker(cfg)

    scenarios = [
        Scenario(cs, ov, ms, win)
        for cs, ov, ms, win in itertools.product(
            args.chunk_sizes,
            args.chunk_overlaps,
            args.min_sims,
            args.sentence_windows,
        )
    ]

    logger.info("Running %d scenarios", len(scenarios))

    results: List[AblationMetrics] = []
    with Progress() as progress:
        task = progress.add_task("Running scenarios...", total=len(scenarios))
        for scenario in scenarios:
            logger.info(
                "Scenario chunk_size=%s overlap=%s min_sim=%.2f window=%s",
                scenario.chunk_size,
                scenario.chunk_overlap,
                scenario.min_similarity,
                scenario.sentence_window,
            )
            metrics = evaluate_scenario(chunker, scenario, texts)
            results.append(metrics)
            progress.advance(task)

    console = Console()
    console.print(render_table(results))

    if args.save_json:
        import csv
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([r.to_dict() for r in results], f, indent=2, ensure_ascii=False)
        logger.info("Saved metrics to %s", path)

        # Also save CSV
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=list(results[0].to_dict().keys()))
            writer.writeheader()
            for r in results:
                flat = r.to_dict()
                flat.update(flat.pop("scenario"))
                writer.writerow(flat)
        logger.info("Saved CSV to %s", csv_path)

    return results


def main() -> None:
    run_semantic_ablation()


if __name__ == "__main__":
    main()
