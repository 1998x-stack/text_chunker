"""DocumentIR-first API; legacy chunker.chunk(str) remains available."""

from .runner import run_chunking

__all__ = ["run_chunking"]
