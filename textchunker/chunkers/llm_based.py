from __future__ import annotations
from typing import List
from loguru import logger
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..providers import OpenAIProvider, HFProvider


@register("llm")
class LLMChunker(BaseChunker):
    """基于 LLM 的智能分块：模型返回 [start,end,title] 列表。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        p = cfg.llm.get("provider", "openai").lower()
        if p == "openai":
            self.provider = OpenAIProvider(cfg.llm.get("openai_model", "gpt-4o-mini"))
        elif p == "hf":
            self.provider = HFProvider(cfg.llm.get("hf_model", "Qwen/Qwen2.5-7B-Instruct"))
        else:
            raise ValueError(f"Unknown provider: {p}")

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        l = self.cfg.llm
        max_chars = int(l.get("max_chars_per_call", 8000))
        sys_prompt = l.get("system_prompt", "Segment the document into meaningful spans. Return JSON spans.")
        max_chunks = c.get("max_chunks")

        spans = self.provider.propose_spans(text, system_prompt=sys_prompt, max_chars=max_chars)
        if not spans:
            # 回退：整体一个块
            return [Chunk(id=0, text=text, start=0, end=len(text), meta={"strategy": "llm"})]

        chunks: List[Chunk] = []
        for s, e, title in spans:
            s = max(0, min(s, len(text)))
            e = max(s, min(e, len(text)))
            piece = text[s:e]
            chunks.append(Chunk(id=len(chunks), text=piece, start=s, end=e,
                                meta={"strategy": "llm", "title": title}))
            if max_chunks and len(chunks) >= max_chunks:
                break
        return chunks
