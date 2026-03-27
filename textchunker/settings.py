from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Settings:
    """Centralized configuration for API, logging, and stats."""

    # API
    api_key: str = ""
    api_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Models
    llm_model: str = "qwen-max"
    embedding_model: str = "text-embedding-v3"

    # Logging
    log_level: str = "INFO"
    log_json: bool = False
    log_dir: str = "logs/"
    log_rotation: str = "10 MB"
    log_retention: int = 5

    # Stats
    stats_enabled: bool = False
    stats_dir: str = "stats/"

    @classmethod
    def from_env(cls) -> Settings:
        """Create Settings from environment variables only."""
        return cls(api_key=os.environ.get("DASHSCOPE_API_KEY", ""))

    @classmethod
    def from_env_and_yaml(cls, yaml_cfg: Dict[str, Any]) -> Settings:
        """Create Settings from environment + YAML config sections."""
        settings_section = yaml_cfg.get("settings", {})
        logging_section = yaml_cfg.get("logging", {})

        return cls(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            api_base_url=settings_section.get(
                "api_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            llm_model=settings_section.get("llm_model", "qwen-max"),
            embedding_model=settings_section.get("embedding_model", "text-embedding-v3"),
            log_level=logging_section.get("level", "INFO"),
            log_json=bool(logging_section.get("json", False)),
            log_dir=logging_section.get("log_dir", "logs/"),
            log_rotation=logging_section.get("rotation", "10 MB"),
            log_retention=int(logging_section.get("retention", 5)),
            stats_enabled=bool(settings_section.get("stats_enabled", False)),
            stats_dir=settings_section.get("stats_dir", "stats/"),
        )
