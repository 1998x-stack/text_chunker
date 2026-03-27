import os
import pytest
from textchunker.settings import Settings


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key-123")
    s = Settings.from_env()
    assert s.api_key == "test-key-123"
    assert s.llm_model == "qwen-max"
    assert s.embedding_model == "text-embedding-v3"
    assert s.api_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"


def test_settings_from_env_missing_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    s = Settings.from_env()
    assert s.api_key == ""


def test_settings_from_yaml():
    yaml_cfg = {
        "settings": {
            "llm_model": "qwen-turbo",
            "embedding_model": "text-embedding-v2",
            "api_base_url": "https://custom.endpoint/v1",
        },
        "logging": {
            "level": "DEBUG",
            "json": True,
            "log_dir": "custom_logs/",
            "rotation": "5 MB",
            "retention": 3,
        },
    }
    s = Settings.from_env_and_yaml(yaml_cfg)
    assert s.llm_model == "qwen-turbo"
    assert s.embedding_model == "text-embedding-v2"
    assert s.log_level == "DEBUG"
    assert s.log_json is True
    assert s.log_dir == "custom_logs/"
    assert s.log_rotation == "5 MB"
    assert s.log_retention == 3


def test_settings_defaults():
    s = Settings.from_env_and_yaml({})
    assert s.llm_model == "qwen-max"
    assert s.log_level == "INFO"
    assert s.log_json is False
    assert s.stats_enabled is False
