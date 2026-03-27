from __future__ import annotations
from typing import Any, Dict, Type

_REGISTRY: Dict[str, Type[Any]] = {}


def register(name: str):
    """Decorator: register a chunking strategy in the global factory."""
    def deco(cls: Type[Any]):
        _REGISTRY[name.lower()] = cls
        return cls
    return deco


def get(name: str) -> Type[Any]:
    if name.lower() not in _REGISTRY:
        raise KeyError(f"Unknown chunker '{name}'. Registered: {list(_REGISTRY)}")
    return _REGISTRY[name.lower()]


def available() -> Dict[str, Type[Any]]:
    return dict(_REGISTRY)
