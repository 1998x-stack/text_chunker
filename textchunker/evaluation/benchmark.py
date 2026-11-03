"""Offline anchored retrieval evaluation; no network or model downloads.

python -m textchunker.evaluation.benchmark \
  --dataset benchmarks/datasets/golden_v1.json \
  --strategies fixed recursive hybrid --tokenizer char \
  --output out/retrieval_benchmark.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from ..core.spans import Span
from ..parsers import parse_document
from ..pipeline import run_chunking
from ..tokenizers import create_tokenizer
from ..types import StrategyConfig
from .report import write_report
from .retrieval import EvidenceAnchor, GoldenQuestion, evaluate_retrieval

_OFFLINE = frozenset({"fixed", "recursive", "structure", "hybrid"})


def load_golden_dataset(path: str | Path):
    """Reject duplicate/absent answer quotes instead of guessing offsets."""
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("schema_version") != 1:
        raise ValueError("Unsupported golden dataset schema_version")
    records = obj.get("documents", [])
    if not records:
        raise ValueError("Golden dataset requires documents")
    documents = {}
    for row in records:
        uri = row["source_uri"]
        if uri in documents or not uri:
            raise ValueError("Duplicate/empty source URI")
        documents[uri] = parse_document(
            row["text"], uri, row.get("format", "text")
        )
    questions = []
    for row in obj.get("queries", []):
        anchors = []
        for evidence in row["evidence"]:
            uri = evidence["source_uri"]
            if uri not in documents:
                raise ValueError(f"Unknown source: {uri}")
            content = documents[uri].text
            if "quote" in evidence:
                quote = evidence["quote"]
                if not quote or content.count(quote) != 1:
                    raise ValueError(
                        f"Evidence quote must be unique in {uri}: {quote!r}"
                    )
                start, stop = content.index(quote), content.index(quote) + len(quote)
            else:
                start, stop = evidence["start"], evidence["end"]
            if (type(start) is not int or type(stop) is not int
                    or not 0 <= start < stop <= len(content)):
                raise ValueError("Invalid golden evidence character span")
            anchors.append(EvidenceAnchor(uri, Span(start, stop)))
        questions.append(GoldenQuestion(
            row["id"], row["query"], tuple(anchors)
        ))
    if not questions:
        raise ValueError("Golden dataset requires queries")
    return list(documents.values()), tuple(questions)


def run_benchmark(
    dataset: str | Path,
    strategies: tuple[str, ...] = ("fixed", "recursive", "hybrid"),
    *, tokenizer: str = "char", chunk_size: int = 128,
    chunk_overlap: int = 16, k: int = 3,
):
    if not strategies or any(s not in _OFFLINE for s in strategies):
        raise ValueError("Only offline strategies are supported")
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Invalid shared chunk budget")
    documents, questions = load_golden_dataset(dataset)
    counter = create_tokenizer(tokenizer)
    comparisons = {}
    for strategy in strategies:
        cfg = StrategyConfig(
            name=strategy,
            common={"chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap, "max_chunks": None},
            fixed={"use_tokens": True},
            hybrid={"oversized_element": "split"},
        )
        started = perf_counter()
        results = [run_chunking(doc, cfg, counter, strict=True)
                   for doc in documents]
        seconds = perf_counter() - started
        ranking = evaluate_retrieval(results, questions, k=k)
        comparisons[strategy] = {
            "integrity_ok": all(r.report.ok and r.complete for r in results),
            "unique_coverage": [r.report.unique_coverage_ratio for r in results],
            "total_chunks": sum(len(r.chunks) for r in results),
            "elapsed_seconds": seconds,
            "retrieval": asdict(ranking),
        }
    return {
        "schema_version": 1, "dataset": str(dataset),
        "tokenizer": counter.name, "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap, "k": k,
        "strategies": comparisons, "scorer": "lexical_term_overlap_v1",
        "notes": "Deterministic offline lexical baseline, not an RAG quality claim.",
    }


def main():
    parser = argparse.ArgumentParser(description="Offline anchored RAG evaluation")
    parser.add_argument("--dataset", default="benchmarks/datasets/golden_v1.json")
    parser.add_argument("--strategies", nargs="+",
                        default=["fixed", "recursive", "hybrid"])
    parser.add_argument("--tokenizer", default="char")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=16)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", default="out/retrieval_benchmark.json")
    args = parser.parse_args()
    result = run_benchmark(
        args.dataset, tuple(args.strategies), tokenizer=args.tokenizer,
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, k=args.k,
    )
    write_report(args.output, result)
    print(f"Wrote offline report: {args.output}")


if __name__ == "__main__":
    main()
