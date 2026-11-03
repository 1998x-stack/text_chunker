"""Embedding-based boundaries followed by strict deterministic budgeting."""

from __future__ import annotations

from typing import List

from loguru import logger

from ..core.validation import validate_parameters
from ..registry import register
from ..types import Chunk
from ..utils import count_tokens, whitespace_sentences_with_offsets
from .base import BaseChunker
from .recursive import _split_by_separators, add_bounded_overlap


@register("semantic")
class SemanticChunker(BaseChunker):
    def __init__(self, cfg):
        super().__init__(cfg)
        from sentence_transformers import SentenceTransformer

        model_name = cfg.semantic.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Loading sentence-transformers model: {}", model_name)
        self.model = SentenceTransformer(model_name)

    def chunk(self, text: str) -> List[Chunk]:
        if not text:
            return []
        import numpy as np

        common, options = self.cfg.common, self.cfg.semantic
        size = int(common.get("chunk_size", 600))
        overlap = int(common.get("chunk_overlap", 100))
        max_chunks = common.get("max_chunks")
        validate_parameters(size, overlap, max_chunks)
        min_similarity = float(options.get("min_similarity", 0.62))
        window = int(options.get("sentence_window", 1))
        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be in [-1, 1]")
        if window <= 0:
            raise ValueError("sentence_window must be positive")

        sentence_spans: List[tuple[int, int]] = []
        for sentence, start, _ in whitespace_sentences_with_offsets(text):
            # A single enormous sentence must also obey budget.
            pieces = (_split_by_separators(
                sentence, ["\n\n", "\n", " ", ""], size)
                if count_tokens(sentence) > size else [sentence])
            cursor = start
            for piece in pieces:
                sentence_spans.append((cursor, cursor + len(piece)))
                cursor += len(piece)

        sentences = [text[start:end].strip()
                     for start, end in sentence_spans]
        embs = self.model.encode(sentences, normalize_embeddings=True)
        chunks: List[Chunk] = []
        buf_start: int | None = None

        def flush(end: int) -> None:
            nonlocal buf_start
            if buf_start is None:
                return
            chunks.append(Chunk(
                id=len(chunks), text=text[buf_start:end],
                start=buf_start, end=end, meta={"strategy": "semantic"},
            ))
            buf_start = None

        for index, (start, end) in enumerate(sentence_spans):
            if buf_start is not None:
                ctx = embs[max(0, index - window):index].mean(axis=0)
                norm = float(np.linalg.norm(ctx))
                sim = float(np.dot(embs[index], ctx) / norm) if norm else 0.0
                if (sim < min_similarity
                        or count_tokens(text[buf_start:end]) > size):
                    flush(start)
            if buf_start is None:
                buf_start = start

        if buf_start is not None:
            flush(len(text))

        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        add_bounded_overlap(text, chunks, size, overlap)
        logger.info("Semantic chunking: {} chunks", len(chunks))
        return chunks
