from __future__ import annotations
from typing import Any, Dict
from .registry import get
from .types import StrategyConfig
from .chunkers.base import BaseChunker


def create_chunker(cfg: StrategyConfig, tokenizer=None,
                   embedding_provider=None, llm_provider=None) -> BaseChunker:
    """工厂：根据策略名创建分块器实例。"""
    cls = get(cfg.name)
    if cfg.name == "semantic" and embedding_provider is not None:
        chunker = cls(cfg, embedding_provider=embedding_provider)
    elif cfg.name == "llm" and llm_provider is not None:
        chunker = cls(cfg, llm_provider=llm_provider)
    else:
        chunker = cls(cfg)
    chunker.tokenizer = tokenizer
    return chunker
