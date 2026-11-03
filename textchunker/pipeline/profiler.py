"""Run-scoped timing wrapper; does not claim external retrieval latency."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from .runner import run_chunking


@dataclass(frozen=True)
class RunProfile:
    elapsed_seconds: float
    canonical_characters: int
    chunks: int
    integrity_ok: bool


def profile_chunking(document, config, tokenizer=None, **kwargs):
    started = perf_counter()
    result = run_chunking(document, config, tokenizer, **kwargs)
    elapsed = perf_counter() - started
    return result, RunProfile(
        elapsed_seconds=elapsed,
        canonical_characters=len(document.text),
        chunks=len(result.chunks),
        integrity_ok=result.report.ok,
    )
