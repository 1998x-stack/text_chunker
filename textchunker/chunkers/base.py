from __future__ import annotations
from typing import List
from abc import ABC, abstractmethod
from ..types import Chunk, StrategyConfig


class BaseChunker(ABC):
    """分块器基类。所有策略需实现 `chunk(text)`。"""

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def chunk(self, text: str) -> List[Chunk]:
        """将输入文本分割为 Chunk 列表。"""
        raise NotImplementedError
