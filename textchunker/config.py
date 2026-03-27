from __future__ import annotations
import argparse
import yaml
from dataclasses import asdict
from typing import Any, Dict
from .types import ProjectConfig, StrategyConfig


def load_yaml(path: str) -> Dict[str, Any]:
    """读取 YAML 配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_project_config(d: Dict[str, Any]) -> ProjectConfig:
    """将 dict 转为强类型配置。"""
    strat = d.get("strategy", {})
    sc = StrategyConfig(
        name=strat.get("name", "recursive"),
        common=strat.get("common", {}),
        semantic=strat.get("semantic", {}),
        recursive=strat.get("recursive", {}),
        fixed=strat.get("fixed", {}),
        structure=strat.get("structure", {}),
        llm=strat.get("llm", {}),
    )
    return ProjectConfig(
        project=d.get("project", {}),
        io=d.get("io", {}),
        strategy=sc,
    )


def build_argparser() -> argparse.ArgumentParser:
    """命令行参数定义。"""
    p = argparse.ArgumentParser(description="Pluggable Text Chunker (Factory)")
    p.add_argument("--config", type=str, default="configs/default.yaml",
                   help="YAML 配置文件路径")
    p.add_argument("--input", type=str, default=None, help="输入文件或目录，覆盖 YAML")
    p.add_argument("--output", type=str, default=None, help="输出路径，覆盖 YAML")
    p.add_argument("--strategy", type=str, default=None,
                   help="策略名：fixed/semantic/recursive/structure/llm")
    p.add_argument("--visualize", action="store_true",
                   help="可视化模式（rich）")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="限制返回的最大分块数")
    # LLM provider 及模型覆盖
    p.add_argument("--provider", type=str, default=None, help="dashscope/hf")
    p.add_argument("--openai-model", type=str, default=None)
    p.add_argument("--hf-model", type=str, default=None)
    # Logging
    p.add_argument("--log-level", type=str, default=None,
                   help="Log level: TRACE/DEBUG/INFO/WARNING/ERROR")
    # Stats
    p.add_argument("--stats", action="store_true", help="Enable statistics collection")
    p.add_argument("--stats-dir", type=str, default=None, help="Directory for stats output")
    # DashScope
    p.add_argument("--llm-model", type=str, default=None, help="LLM model name (default: qwen-max)")
    p.add_argument("--embedding-model", type=str, default=None, help="Embedding model (default: text-embedding-v3)")
    # Chunk params
    p.add_argument("--chunk-size", type=int, default=None, help="Override chunk_size")
    p.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk_overlap")
    return p


def merge_cli(cfg: ProjectConfig, args: argparse.Namespace) -> ProjectConfig:
    """合并 CLI 覆盖项到配置。"""
    if args.input:
        cfg.io["input_path"] = args.input
    if args.output:
        cfg.io["output_path"] = args.output
    if args.strategy:
        cfg.strategy.name = args.strategy
    if args.visualize:
        cfg.project["visualize"] = True
    if args.max_chunks is not None:
        cfg.strategy.common["max_chunks"] = args.max_chunks
    if args.provider:
        cfg.strategy.llm["provider"] = args.provider
    if args.openai_model:
        cfg.strategy.llm["openai_model"] = args.openai_model
    if args.hf_model:
        cfg.strategy.llm["hf_model"] = args.hf_model
    if getattr(args, 'chunk_size', None) is not None:
        cfg.strategy.common["chunk_size"] = args.chunk_size
    if getattr(args, 'chunk_overlap', None) is not None:
        cfg.strategy.common["chunk_overlap"] = args.chunk_overlap
    if getattr(args, 'llm_model', None):
        cfg.strategy.llm["llm_model"] = args.llm_model
    if getattr(args, 'embedding_model', None):
        cfg.strategy.semantic["embedding_model"] = args.embedding_model
    return cfg
