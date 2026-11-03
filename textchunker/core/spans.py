"""Half-open offsets into a canonical Python unicode string."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"Invalid half-open span [{self.start}, {self.end})")

    def extract(self, source: str) -> str:
        if self.end > len(source):
            raise ValueError("Span exceeds source length")
        return source[self.start:self.end]
