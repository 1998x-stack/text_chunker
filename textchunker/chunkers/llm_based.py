"""LLM-proposed breakpoints, verified and repaired before publication."""

from __future__ import annotations

from typing import List

from loguru import logger

from ..core.validation import validate_parameters
from ..exceptions import ConfigError
from ..providers import DashScopeProvider, HFProvider
from ..registry import register
from ..types import Chunk
from ..utils import count_tokens
from .base import BaseChunker
from .recursive import _split_by_separators, add_bounded_overlap


def _proposals_are_complete(proposals, length: int) -> bool:
    if not proposals:
        return False
    cursor = 0
    for item in proposals:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            return False
        start, end, _ = item
        if (not isinstance(start, int) or not isinstance(end, int)
                or isinstance(start, bool) or isinstance(end, bool)
                or start != cursor or end <= start or end > length):
            return False
        cursor = end
    return cursor == length


@register("llm")
class LLMChunker(BaseChunker):
    """The model only suggests boundaries; the engine owns correctness."""

    def __init__(self, cfg):
        super().__init__(cfg)
        provider = cfg.llm.get("provider", "dashscope").lower()
        if provider in ("dashscope", "openai"):
            model = cfg.llm.get("llm_model",
                                cfg.llm.get("openai_model", "qwen-max"))
            self.provider = DashScopeProvider(
                model=model,
                api_key=cfg.llm.get("api_key", ""),
                api_base_url=cfg.llm.get("api_base_url", ""),
            )
        elif provider == "hf":
            self.provider = HFProvider(
                cfg.llm.get("hf_model", "Qwen/Qwen2.5-7B-Instruct")
            )
        else:
            raise ConfigError(f"Unknown provider: {provider}")

    def chunk(self, text: str) -> List[Chunk]:
        if not text:
            return []
        common = self.cfg.common
        size = int(common.get("chunk_size", 512))
        overlap = int(common.get("chunk_overlap", 80))
        max_chunks = common.get("max_chunks")
        validate_parameters(size, overlap, max_chunks)
        options = self.cfg.llm
        max_chars = int(options.get("max_chars_per_call", 8000))
        if max_chars <= 0:
            raise ConfigError("max_chars_per_call must be positive")
        system_prompt = options.get(
            "system_prompt", "Segment the document into meaningful spans. Return JSON spans."
        )

        chunks: List[Chunk] = []
        for base in range(0, len(text), max_chars):
            window = text[base:base + max_chars]
            try:
                proposals = self.provider.propose_spans(
                    window, system_prompt=system_prompt, max_chars=max_chars
                )
            except Exception as exc:
                logger.warning("LLM proposal failed: {}", exc)
                proposals = []

            valid = _proposals_are_complete(proposals, len(window))
            if not valid:
                logger.warning(
                    "LLM proposed missing/overlapping spans; using lossless fallback"
                )
                proposals = [(0, len(window), None)]

            for start, end, title in proposals:
                candidate = window[start:end]
                pieces = (_split_by_separators(
                    candidate, ["\n\n", "\n", " ", ""], size
                ) if count_tokens(candidate) > size else [candidate])
                cursor = base + start
                for piece in pieces:
                    chunks.append(Chunk(
                        id=len(chunks), text=piece,
                        start=cursor, end=cursor + len(piece),
                        meta={
                            "strategy": "llm", "title": title,
                            "fallback": not valid,
                        },
                    ))
                    cursor += len(piece)
                    if max_chunks is not None and len(chunks) >= max_chunks:
                        break
                if max_chunks is not None and len(chunks) >= max_chunks:
                    break
            if max_chunks is not None and len(chunks) >= max_chunks:
                break

        add_bounded_overlap(text, chunks, size, overlap)
        logger.info("LLM chunking: {} verified chunks", len(chunks))
        return chunks
