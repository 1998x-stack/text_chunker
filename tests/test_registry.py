import pytest
from textchunker.registry import register, get, available, _REGISTRY


def test_register_and_get():
    @register("test_strategy_xyz")
    class TestChunker:
        pass
    assert get("test_strategy_xyz") is TestChunker
    del _REGISTRY["test_strategy_xyz"]


def test_get_unknown():
    with pytest.raises(KeyError, match="Unknown chunker"):
        get("nonexistent_strategy_abc")


def test_available():
    import textchunker.chunkers  # noqa: F401  # trigger registration
    result = available()
    assert isinstance(result, dict)
    assert "fixed" in result
    assert "recursive" in result


def test_register_case_insensitive():
    @register("CamelCase_Test")
    class CamelChunker:
        pass
    assert get("camelcase_test") is CamelChunker
    del _REGISTRY["camelcase_test"]
