import json
from pathlib import Path

from textchunker.export import save_jsonl
from textchunker.types import Chunk


def test_save_jsonl_supports_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    chunks = [
        Chunk(id=0, text="hello", start=0, end=5, meta={"strategy": "test"}),
        Chunk(id=1, text="world", start=6, end=11, meta={"strategy": "test"}),
    ]

    save_jsonl("chunks.jsonl", chunks, source="input.txt")

    data_path = Path("chunks.jsonl")
    assert data_path.exists()

    with data_path.open("r", encoding="utf-8") as fh:
        lines = [json.loads(line) for line in fh]

    assert lines[0]["source"] == "input.txt"
    assert lines[1]["text"] == "world"
