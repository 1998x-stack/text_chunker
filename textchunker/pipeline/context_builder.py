"""Non-destructive contextual embedding/display text.

Never mutate ChunkRecord.text: the source-backed content must remain exact.
Context may exceed the chunk token budget and must be budgeted separately
when sent to a model.
"""
from __future__ import annotations


def contextualize_chunk(chunk, separator: str = " > ") -> str:
    heading = separator.join(chunk.heading_path)
    return (heading + "\n\n" + chunk.text) if heading else chunk.text
