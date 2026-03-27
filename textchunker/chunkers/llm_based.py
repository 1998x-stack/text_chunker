from __future__ import annotations

from typing import List

from loguru import logger

from ..exceptions import ChunkerError, ConfigError
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..providers import DashScopeProvider, HFProvider


@register("llm")
class LLMChunker(BaseChunker):
    """LLM-based chunker: model returns [start, end, title] span proposals."""

    def __init__(self, cfg):
        super().__init__(cfg)
        p = cfg.llm.get("provider", "dashscope").lower()
        if p in ("dashscope", "openai"):
            model = cfg.llm.get("llm_model", cfg.llm.get("openai_model", "qwen-max"))
            api_key = cfg.llm.get("api_key", "")
            api_base = cfg.llm.get("api_base_url", "")
            self.provider = DashScopeProvider(model=model, api_key=api_key, api_base_url=api_base)
        elif p == "hf":
            self.provider = HFProvider(cfg.llm.get("hf_model", "Qwen/Qwen2.5-7B-Instruct"))
        else:
            raise ConfigError(f"Unknown provider: {p}")

    def chunk(self, text: str) -> List[Chunk]:
        l = self.cfg.llm
        max_chars = int(l.get("max_chars_per_call", 8000))
        sys_prompt = l.get("system_prompt", "Segment the document into meaningful spans. Return JSON spans.")
        max_chunks = self.cfg.common.get("max_chunks")

        spans = self.provider.propose_spans(text, system_prompt=sys_prompt, max_chars=max_chars)
        if not spans:
            logger.warning("Provider returned no spans, falling back to single chunk")
            return [Chunk(id=0, text=text, start=0, end=len(text), meta={"strategy": "llm"})]

        chunks: List[Chunk] = []
        for s, e, title in spans:
            s = max(0, min(s, len(text)))
            e = max(s, min(e, len(text)))
            piece = text[s:e]
            chunks.append(Chunk(
                id=len(chunks), text=piece, start=s, end=e,
                meta={"strategy": "llm", "title": title},
            ))
            if max_chunks and len(chunks) >= max_chunks:
                break

        logger.info("LLM chunking: {} chunks from provider spans", len(chunks))
        return chunks
