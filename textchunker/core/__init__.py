"""Shared, dependency-light correctness primitives for all chunkers."""

from .spans import Span
from .token_budget import largest_fitting_end, overlap_start
from .validation import validate_chunks, validate_parameters

__all__ = ["Span", "largest_fitting_end", "overlap_start",
           "validate_chunks", "validate_parameters"]
