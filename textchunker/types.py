from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """表示一个文本分块的统一数据结构。

    Attributes:
        id: 分块编号（从 0 开始）。
        text: 分块文本内容。
        start: 在原始文本中的起始字符位置（包含）。
        end: 在原始文本中的结束字符位置（不包含）。
        meta: 额外元数据（如标题、层级、来源文件路径等）。
    """
    id: int
    text: str
    start: int
    end: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileDoc:
    """待处理的文档对象，包含路径与内容。"""
    path: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """策略通用与专项配置（从 YAML/CLI 合并而来）。"""
    name: str
    common: Dict[str, Any] = field(default_factory=dict)
    semantic: Dict[str, Any] = field(default_factory=dict)
    recursive: Dict[str, Any] = field(default_factory=dict)
    fixed: Dict[str, Any] = field(default_factory=dict)
    structure: Dict[str, Any] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProjectConfig:
    """工程配置。"""
    project: Dict[str, Any]
    io: Dict[str, Any]
    strategy: StrategyConfig
