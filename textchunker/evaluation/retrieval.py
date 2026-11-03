"""Offline retrieval metrics against canonical source evidence, not guesses."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from ..core.contracts import ChunkRecord, ChunkingResult
from ..core.spans import Span


@dataclass(frozen=True)
class EvidenceAnchor:
    source_uri: str
    span: Span

    def __post_init__(self):
        if not self.source_uri or self.span.start == self.span.end:
            raise ValueError("Anchored evidence must be nonempty")


@dataclass(frozen=True)
class GoldenQuestion:
    query_id: str
    query: str
    evidence: tuple[EvidenceAnchor, ...]

    def __post_init__(self):
        if not self.query_id or not self.query.strip() or not self.evidence:
            raise ValueError("Golden question needs id, query, and evidence")


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    source_uri: str
    span: Span
    score: float
    relevant: bool


@dataclass(frozen=True)
class QueryEvaluation:
    query_id: str
    hit: bool
    reciprocal_rank: float
    ndcg: float
    oracle_has_evidence: bool
    ranked: tuple[RankedChunk, ...]


@dataclass(frozen=True)
class RetrievalReport:
    total_queries: int
    k: int
    recall_at_k: float
    mrr_at_k: float
    ndcg_at_k: float
    oracle_recall: float
    questions: tuple[QueryEvaluation, ...]


def _terms(value: str) -> set[str]:
    """Simple reproducible lexical baseline for English and CJK text."""
    return set(re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", value.lower()))


def lexical_score(query: str, passage: str) -> float:
    q, t = _terms(query), _terms(passage)
    return len(q & t) / math.sqrt(len(q) * len(t)) if q and t else 0.0


def _source(chunk: ChunkRecord) -> str:
    return chunk.source_locations[0].source_uri if chunk.source_locations else ""


def _relevant(chunk: ChunkRecord, question: GoldenQuestion) -> bool:
    return any(
        anchor.source_uri == _source(chunk)
        and chunk.span.start <= anchor.span.start
        and chunk.span.end >= anchor.span.end
        for anchor in question.evidence
    )


def evaluate_retrieval(
    results: Sequence[ChunkingResult], questions: Sequence[GoldenQuestion],
    *, k: int = 5, scorer: Callable[[str, str], float] = lexical_score,
) -> RetrievalReport:
    """Query Recall@k, MRR@k, binary nDCG@k and oracle evidence recall.

    A relevant chunk *fully contains* at least one evidence span, anchored to
    the same source URI. Missing sources are dataset errors, not recall misses.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    ids = [q.query_id for q in questions]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate golden question IDs")
    chunks = [c for r in results for c in r.chunks]
    sources = {_source(c) for c in chunks}
    evaluations = []

    for question in questions:
        unknown = {e.source_uri for e in question.evidence} - sources
        if unknown:
            raise ValueError(
                f"Question {question.query_id}: unknown sources {sorted(unknown)}"
            )
        scored = []
        for chunk in chunks:
            value = float(scorer(question.query, chunk.text))
            if not math.isfinite(value):
                raise ValueError("Scorer returned non-finite value")
            scored.append((chunk, value, _relevant(chunk, question)))
        scored.sort(key=lambda item: (
            -item[1], _source(item[0]), item[0].span.start, item[0].chunk_id
        ))
        top = scored[:k]
        total_relevant = sum(1 for _, _, is_relevant in scored if is_relevant)
        ranks = [rank for rank, (_, _, relevant) in enumerate(top, 1)
                 if relevant]
        dcg = sum(1 / math.log2(rank + 1) for rank in ranks)
        ideal = sum(1 / math.log2(rank + 1)
                    for rank in range(1, min(k, total_relevant) + 1))
        evaluations.append(QueryEvaluation(
            query_id=question.query_id,
            hit=bool(ranks),
            reciprocal_rank=1 / ranks[0] if ranks else 0.0,
            ndcg=dcg / ideal if ideal else 0.0,
            oracle_has_evidence=total_relevant > 0,
            ranked=tuple(RankedChunk(
                chunk_id=chunk.chunk_id, source_uri=_source(chunk),
                span=chunk.span, score=score, relevant=relevant,
            ) for chunk, score, relevant in top),
        ))
    length = len(evaluations)
    if not length:
        raise ValueError("At least one golden question required")
    return RetrievalReport(
        total_queries=length, k=k,
        recall_at_k=sum(e.hit for e in evaluations) / length,
        mrr_at_k=sum(e.reciprocal_rank for e in evaluations) / length,
        ndcg_at_k=sum(e.ndcg for e in evaluations) / length,
        oracle_recall=sum(e.oracle_has_evidence for e in evaluations) / length,
        questions=tuple(evaluations),
    )
