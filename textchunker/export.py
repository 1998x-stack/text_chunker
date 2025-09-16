from __future__ import annotations
import json
import os
from typing import Iterable, List
from .types import Chunk, FileDoc


def save_jsonl(path: str, chunks: List[Chunk], source: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            rec = {"id": c.id, "text": c.text, "start": c.start, "end": c.end, "meta": c.meta, "source": source}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def save_txt_dir(dirpath: str, chunks: List[Chunk], prefix: str = "chunk") -> None:
    os.makedirs(dirpath, exist_ok=True)
    for c in chunks:
        fn = os.path.join(dirpath, f"{prefix}-{c.id:04d}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(c.text)
