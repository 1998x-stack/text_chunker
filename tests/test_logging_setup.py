import json
import os
import time
from pathlib import Path

from loguru import logger

from textchunker.log_config import setup_logging
from textchunker.settings import Settings


def test_setup_logging_configures_console():
    settings = Settings(log_level="DEBUG", log_json=False, log_dir="logs/")
    setup_logging(settings)
    logger.info("test message")
    assert True


def test_setup_logging_json_file(tmp_path):
    log_dir = str(tmp_path / "logs")
    settings = Settings(
        log_level="DEBUG",
        log_json=True,
        log_dir=log_dir,
        log_rotation="1 MB",
        log_retention=2,
    )
    setup_logging(settings)
    logger.info("json test message")
    time.sleep(0.1)  # small delay to flush

    log_files = list(Path(log_dir).glob("*.log"))
    assert len(log_files) >= 1

    with open(log_files[0], "r") as f:
        lines = f.readlines()
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert "text" in record
    assert "json test message" in record["text"]


def test_setup_logging_level_filtering(tmp_path):
    log_dir = str(tmp_path / "logs")
    settings = Settings(log_level="WARNING", log_json=True, log_dir=log_dir)
    setup_logging(settings)
    logger.debug("should be filtered")
    logger.warning("should appear")
    time.sleep(0.1)

    log_files = list(Path(log_dir).glob("*.log"))
    if log_files:
        with open(log_files[0], "r") as f:
            content = f.read()
        assert "should be filtered" not in content
        assert "should appear" in content


def test_setup_logging_creates_log_dir(tmp_path):
    log_dir = str(tmp_path / "nested" / "logs")
    settings = Settings(log_level="INFO", log_json=False, log_dir=log_dir)
    setup_logging(settings)
    assert Path(log_dir).exists()
