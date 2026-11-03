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


def save_result_jsonl(path: str, result, source: str,
                      mode: str = "w") -> None:
    """Write provenance-rich v0.3 records, retaining legacy JSONL keys."""
    if mode not in {"w", "a"}:
        raise ValueError("mode must be 'w' or 'a'")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, mode, encoding="utf-8") as stream:
        for chunk in result.chunks:
            record = {
                "id": chunk.index,
                "text": chunk.text,
                "start": chunk.span.start,
                "end": chunk.span.end,
                "meta": dict(chunk.metadata),
                "source": source,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "token_count": chunk.token_count,
                "heading_path": list(chunk.heading_path),
                "source_locations": [
                    {
                        "source_uri": location.source_uri,
                        "canonical_span": [
                            location.canonical_span.start,
                            location.canonical_span.end,
                        ],
                        "source_span": (
                            [location.source_span.start, location.source_span.end]
                            if location.source_span is not None else None
                        ),
                        "page_number": location.page_number,
                        "block_id": location.block_id,
                        "mapping": location.mapping,
                    }
                    for location in chunk.source_locations
                ],
                "complete": result.complete,
                "tokenizer_id": result.tokenizer_id,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
