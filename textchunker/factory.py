from __future__ import annotations
from typing import Any, Dict
from .registry import get
from .types import StrategyConfig
from .chunkers.base import BaseChunker


def create_chunker(cfg: StrategyConfig, tokenizer=None) -> BaseChunker:
    """工厂：根据策略名创建分块器实例。"""
    cls = get(cfg.name)
    chunker = cls(cfg)
    chunker.tokenizer = tokenizer
    return chunker
