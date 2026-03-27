import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_text_short():
    return "First sentence. Second sentence. Third sentence."


@pytest.fixture
def sample_text_long():
    return "Paragraph one with content.\n\n" * 20


@pytest.fixture
def sample_text_chinese():
    return "第一段话。 第二段话！ 第三段话？ " * 10


@pytest.fixture
def default_strategy_config():
    from textchunker.types import StrategyConfig
    return StrategyConfig(
        name="recursive",
        common={"chunk_size": 100, "chunk_overlap": 20},
        recursive={"separators": ["\n\n", "\n", " ", ""]},
        semantic={}, fixed={}, structure={}, llm={},
    )
