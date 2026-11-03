"""Deterministic batch-safe chunk serialization."""

from __future__ import annotations

import json
import os
from typing import List

from .types import Chunk


def save_jsonl(path: str, chunks: List[Chunk], source: str,
               mode: str = "w") -> None:
    """Write or explicitly append documents; never overwrite implicitly."""
    if mode not in {"w", "a"}:
        raise ValueError("mode must be 'w' or 'a'")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as stream:
        for chunk in chunks:
            record = {
                "id": chunk.id,
                "text": chunk.text,
                "start": chunk.start,
                "end": chunk.end,
                "meta": chunk.meta,
                "source": source,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_txt_dir(dirpath: str, chunks: List[Chunk],
                 prefix: str = "chunk") -> None:
    os.makedirs(dirpath, exist_ok=True)
    for chunk in chunks:
        path = os.path.join(dirpath, f"{prefix}-{chunk.id:04d}.txt")
        with open(path, "w", encoding="utf-8") as stream:
            stream.write(chunk.text)
