"""Budget helpers that never split a Python Unicode code point.

These helpers operate on the caller's explicit length function. They work for
both token counts and character counts and verify every returned boundary.
"""

from __future__ import annotations

from typing import Callable

from ..exceptions import ChunkerError

Counter = Callable[[str], int]


def largest_fitting_end(text: str, start: int, budget: int,
                        count: Counter) -> int:
    """Find a fitting prefix of text[start:] without losing source characters.

    Token counts are not strictly monotonic for every tokenizer. The binary
    search is an efficient candidate search; its returned candidate is always
    verified. The algorithm guarantees a fitting result, not a globally
    maximal token-boundary choice.
    """
    if budget <= 0:
        raise ChunkerError("Token/character budget must be positive")
    if start < 0 or start >= len(text):
        raise ValueError("start must point at an existing character")

    lo, hi, best = start + 1, len(text), start
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(text[start:mid]) <= budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    if best == start:
        if count(text[start:start + 1]) > budget:
            raise ChunkerError(
                "A single Unicode character exceeds the configured budget"
            )
        best = start + 1

    while count(text[start:best]) > budget and best > start:
        best -= 1
    if best == start:
        raise ChunkerError("No nonempty prefix fits the configured budget")
    return best


def overlap_start(text: str, previous_start: int, current_start: int,
                  current_end: int, overlap_budget: int, full_budget: int,
                  count: Counter) -> int:
    """Choose a bounded overlap without growing the new chunk past budget.

    The current_start:current_end interval is never dropped. An overlap may
    be smaller than requested when the current content nearly fills budget.
    """
    if overlap_budget <= 0 or current_start <= previous_start:
        return current_start

    best = current_start
    candidate = current_start - 1
    while candidate >= previous_start:
        if count(text[candidate:current_start]) > overlap_budget:
            break
        if count(text[candidate:current_end]) > full_budget:
            break
        best = candidate
        candidate -= 1
    return best
