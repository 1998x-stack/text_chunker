from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from loguru import logger


class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def encode(self, sentences: List[str]) -> np.ndarray:
        raise NotImplementedError


class DashScopeEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding via DashScope API (OpenAI-compatible endpoint)."""

    def __init__(
        self,
        model: str = "text-embedding-v3",
        api_key: str = "",
        api_base_url: str = "",
    ) -> None:
        from openai import OpenAI

        self.model = model
        key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        base = api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self.client = OpenAI(api_key=key, base_url=base)

    def encode(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.array([])
        resp = self.client.embeddings.create(model=self.model, input=sentences)
        vecs = [item.embedding for item in resp.data]
        arr = np.array(vecs, dtype=float)
        # Normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Embedding via local sentence-transformers model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("Loading sentence-transformers model: {}", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences: List[str]) -> np.ndarray:
        return self.model.encode(sentences, normalize_embeddings=True)
