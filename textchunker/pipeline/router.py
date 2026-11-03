"""Deterministic default strategy selection, no LLM inference."""
from __future__ import annotations


def recommend_strategy(document) -> str:
    """Prefer structures when parsed; use recursive for plain extracted text."""
    if document.format in {"markdown", "html", "docling"} and document.blocks:
        return "hybrid"
    return "recursive"
