from __future__ import annotations

from typing import List

import numpy as np  # type: ignore
from loguru import logger
from sentence_transformers import SentenceTransformer  # type: ignore

from ..exceptions import ChunkerError
from ..types import Chunk
from ..utils import whitespace_sentences_with_offsets, count_tokens
from ..registry import register
from .base import BaseChunker


@register("semantic")
class SemanticChunker(BaseChunker):
    """Semantic chunker: merges sentences by embedding similarity."""

    def __init__(self, cfg):
        super().__init__(cfg)
        model_name = self.cfg.semantic.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info("Loading sentence-transformers model: {}", model_name)
        self.model = SentenceTransformer(model_name)

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        s = self.cfg.semantic
        size = int(c.get("chunk_size", 600))
        overlap = int(c.get("chunk_overlap", 100))
        min_sim = float(s.get("min_similarity", 0.62))
        window = int(s.get("sentence_window", 1))
        max_chunks = c.get("max_chunks")

        sent_data = whitespace_sentences_with_offsets(text)
        if not sent_data:
            return [Chunk(id=0, text=text, start=0, end=len(text), meta={"strategy": "semantic"})]

        sents = [sd[0] for sd in sent_data]
        offsets = [(sd[1], sd[2]) for sd in sent_data]

        embs = self.model.encode(sents, normalize_embeddings=True)
        chunks: List[Chunk] = []
        buf_indices: List[int] = []

        def flush() -> None:
            if not buf_indices:
                return
            chunk_start = offsets[buf_indices[0]][0]
            chunk_end = offsets[buf_indices[-1]][1]
            piece = text[chunk_start:chunk_end].strip()
            if piece:
                chunks.append(Chunk(
                    id=len(chunks), text=piece, start=chunk_start, end=chunk_end,
                    meta={"strategy": "semantic"},
                ))
            buf_indices.clear()

        for i in range(len(sents)):
            if buf_indices and i >= window:
                v = embs[i]
                ctx = embs[max(0, i - window):i].mean(axis=0)
                sim = float(np.dot(v, ctx))
                if sim < min_sim:
                    flush()
                    if max_chunks and len(chunks) >= max_chunks:
                        return chunks

            buf_indices.append(i)

            buf_text = text[offsets[buf_indices[0]][0]:offsets[i][1]]
            if count_tokens(buf_text) >= size:
                flush()
                if max_chunks and len(chunks) >= max_chunks:
                    return chunks

                if overlap > 0:
                    back_target = offsets[i][1] - overlap
                    j = i
                    while j >= 0 and offsets[j][0] > back_target:
                        j -= 1
                    if j + 1 <= i:
                        buf_indices.append(i)
                continue

            if max_chunks and len(chunks) >= max_chunks:
                break

        if buf_indices and (not max_chunks or len(chunks) < max_chunks):
            flush()

        logger.info("Semantic chunking: {} chunks from {} sentences", len(chunks), len(sents))
        return chunks
