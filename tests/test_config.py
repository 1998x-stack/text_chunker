import pytest
from pathlib import Path

from textchunker.config import load_yaml, to_project_config, build_argparser, merge_cli


def test_load_yaml(tmp_path):
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text("strategy:\n  name: fixed\n  common:\n    chunk_size: 100\n")
    data = load_yaml(str(yaml_file))
    assert data["strategy"]["name"] == "fixed"
    assert data["strategy"]["common"]["chunk_size"] == 100


def test_to_project_config():
    d = {
        "project": {"name": "test"},
        "io": {"input_path": "in.txt"},
        "strategy": {
            "name": "recursive",
            "common": {"chunk_size": 256},
        },
    }
    cfg = to_project_config(d)
    assert cfg.strategy.name == "recursive"
    assert cfg.strategy.common["chunk_size"] == 256


def test_to_project_config_defaults():
    cfg = to_project_config({})
    assert cfg.strategy.name == "recursive"
    assert cfg.project == {}


def test_merge_cli_strategy_override():
    cfg = to_project_config({"strategy": {"name": "recursive"}})
    args = build_argparser().parse_args(["--strategy", "semantic"])
    merged = merge_cli(cfg, args)
    assert merged.strategy.name == "semantic"


def test_merge_cli_preserves_unset():
    cfg = to_project_config({"strategy": {"name": "fixed"}})
    args = build_argparser().parse_args([])
    merged = merge_cli(cfg, args)
    assert merged.strategy.name == "fixed"
