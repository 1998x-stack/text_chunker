"""Deterministic, injectable semantic splitting with bounded embeddings."""
from __future__ import annotations

from collections import OrderedDict
from typing import List

from loguru import logger

from ..core.validation import validate_parameters
from ..exceptions import ConfigError, ProviderError
from ..registry import register
from ..types import Chunk
from ..utils import whitespace_sentences_with_offsets
from .base import BaseChunker
from .recursive import _split_by_separators, _split_spans, add_bounded_overlap


@register("semantic")
class SemanticChunker(BaseChunker):
    """Sentence boundaries + locally normalized embedding cosine similarities."""

    def __init__(self, cfg, embedding_provider=None):
        super().__init__(cfg)
        self._embedding_cache: OrderedDict[str, object] = OrderedDict()
        if embedding_provider is not None:
            self.model = embedding_provider
        else:
            from sentence_transformers import SentenceTransformer
            model_name = cfg.semantic.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Loading sentence-transformers model: {}", model_name)
            self.model = SentenceTransformer(model_name)

    def _encode(self, sentences, batch_size, cache_size, max_calls):
        import numpy as np

        # Each model instance owns its cache; no cross-model contamination.
        lookup = {}
        for value in sentences:
            if value in self._embedding_cache:
                lookup[value] = self._embedding_cache[value]
                self._embedding_cache.move_to_end(value)
        pending = list(dict.fromkeys(
            value for value in sentences if value not in lookup
        ))
        calls = 0
        dimension = None
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            if calls >= max_calls:
                raise ProviderError("Embedding request budget exceeded")
            calls += 1
            try:
                raw = self.model.encode(batch, normalize_embeddings=True)
                arr = np.asarray(raw, dtype=float)
            except Exception as exc:
                raise ProviderError(
                    f"Embedding backend failed: {type(exc).__name__}"
                ) from exc
            if (arr.ndim != 2 or arr.shape[0] != len(batch)
                    or arr.shape[1] == 0 or not np.isfinite(arr).all()):
                raise ProviderError("Embedding backend returned invalid shape/values")
            norms = np.linalg.norm(arr, axis=1)
            if not np.isfinite(norms).all() or (norms == 0).any():
                raise ProviderError("Embedding backend returned zero/invalid vectors")
            arr = arr / norms[:, None]
            if dimension is None and lookup:
                dimension = len(next(iter(lookup.values())))
            if dimension is None:
                dimension = arr.shape[1]
            if arr.shape[1] != dimension:
                raise ProviderError("Embedding dimensions changed between batches")
            for value, vector in zip(batch, arr):
                lookup[value] = vector
                if cache_size:
                    self._embedding_cache[value] = vector
                    self._embedding_cache.move_to_end(value)
                    while len(self._embedding_cache) > cache_size:
                        self._embedding_cache.popitem(last=False)

        result = [lookup[value] for value in sentences]
        if not result:
            return np.empty((0, 0))
        dimensions = {len(value) for value in result}
        if len(dimensions) != 1:
            raise ProviderError("Incompatible cached embedding dimensions")
        return np.stack(result)

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
        batch = int(options.get("embedding_batch_size", 64))
        max_calls = int(options.get("max_embedding_calls", 256))
        cache_size = int(options.get("embedding_cache_size", 0))
        on_error = options.get("on_embedding_error", "error")
        if (not -1 <= min_similarity <= 1 or window <= 0 or batch <= 0
                or max_calls <= 0 or cache_size < 0
                or on_error not in {"error", "recursive"}):
            raise ConfigError("Invalid semantic embedding/threshold configuration")

        sentence_spans = []
        for sentence, start, _ in whitespace_sentences_with_offsets(text):
            parts = (_split_by_separators(
                sentence, ["\n\n", "\n", " ", ""], size, self.count_tokens
            ) if self.count_tokens(sentence) > size else [sentence])
            cursor = start
            for piece in parts:
                sentence_spans.append((cursor, cursor + len(piece)))
                cursor += len(piece)

        values = [text[start:end].strip() or " "
                  for start, end in sentence_spans]
        try:
            embs = self._encode(values, batch, cache_size, max_calls)
        except ProviderError as exc:
            if on_error != "recursive":
                raise
            logger.warning("Embedding unavailable; deterministic fallback: {}", exc)
            spans = _split_spans(
                text, ["\n\n", "\n", " ", ""], size, self.count_tokens
            )
            chunks = [
                Chunk(
                    id=index, text=part.extract(text),
                    start=part.start, end=part.end,
                    meta={"strategy": "semantic", "fallback": "recursive"},
                )
                for index, part in enumerate(spans[:max_chunks])
            ] if max_chunks is not None else [
                Chunk(
                    id=index, text=part.extract(text),
                    start=part.start, end=part.end,
                    meta={"strategy": "semantic", "fallback": "recursive"},
                )
                for index, part in enumerate(spans)
            ]
            add_bounded_overlap(
                text, chunks, size, overlap, count=self.count_tokens
            )
            return chunks

        chunks = []
        beginning = None

        def flush(stop):
            nonlocal beginning
            if beginning is None:
                return
            chunks.append(Chunk(
                id=len(chunks), text=text[beginning:stop],
                start=beginning, end=stop, meta={"strategy": "semantic"},
            ))
            beginning = None

        for index, (start, end) in enumerate(sentence_spans):
            if beginning is not None:
                context = embs[max(0, index - window):index].mean(axis=0)
                norm = float(np.linalg.norm(context))
                similarity = float(np.dot(embs[index], context) / norm) if norm else 0
                if (similarity < min_similarity
                        or self.count_tokens(text[beginning:end]) > size):
                    flush(start)
            if beginning is None:
                beginning = start

        if beginning is not None:
            flush(len(text))
        if max_chunks is not None:
            chunks = chunks[:max_chunks]
        add_bounded_overlap(text, chunks, size, overlap, count=self.count_tokens)
        logger.info("Semantic splitting: {} chunks", len(chunks))
        return chunks
