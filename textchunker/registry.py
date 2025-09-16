from __future__ import annotations
from typing import Dict, Type
from .chunkers.base import BaseChunker

_REGISTRY: Dict[str, Type[BaseChunker]] = {}


def register(name: str):
    """装饰器：注册分块策略到全局工厂。"""
    def deco(cls: Type[BaseChunker]):
        _REGISTRY[name.lower()] = cls
        return cls
    return deco


def get(name: str) -> Type[BaseChunker]:
    if name.lower() not in _REGISTRY:
        raise KeyError(f"Unknown chunker '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name.lower()]


def available() -> Dict[str, Type[BaseChunker]]:
    return dict(_REGISTRY)
