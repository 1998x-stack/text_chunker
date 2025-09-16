from __future__ import annotations
from typing import List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from .types import Chunk

_console = Console()


def preview(text: str, limit: int = 80) -> str:
    s = text.replace("\n", " ")
    return (s[:limit] + "…") if len(s) > limit else s


def show_chunks(chunks: List[Chunk], title: str = "Chunk Preview") -> None:
    """Rich 可视化：展示分块统计与预览。"""
    total_chars = sum(len(c.text) for c in chunks)
    table = Table(title=title)
    table.add_column("ID", justify="right")
    table.add_column("Chars")
    table.add_column("Span")
    table.add_column("Meta")
    table.add_column("Preview")
    for c in chunks:
        meta = ", ".join(f"{k}:{v}" for k, v in c.meta.items() if v is not None) or "-"
        table.add_row(str(c.id), str(len(c.text)), f"[{c.start},{c.end})", meta, preview(c.text))
    _console.print(Panel.fit(table, title=f"Chunks={len(chunks)}, TotalChars={total_chars}"))
