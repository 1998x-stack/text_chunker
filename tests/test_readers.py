import pytest
from pathlib import Path

from textchunker.readers import load_inputs


def test_read_text_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world content", encoding="utf-8")
    docs = list(load_inputs(str(f)))
    assert len(docs) == 1
    assert docs[0].text == "Hello world content"
    assert docs[0].path == str(f)


def test_read_markdown_file(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("# Title\nContent here", encoding="utf-8")
    docs = list(load_inputs(str(f)))
    assert len(docs) == 1
    assert "Title" in docs[0].text


def test_read_directory(tmp_path):
    (tmp_path / "a.txt").write_text("File A", encoding="utf-8")
    (tmp_path / "b.txt").write_text("File B", encoding="utf-8")
    docs = list(load_inputs(str(tmp_path)))
    assert len(docs) == 2
    texts = {d.text for d in docs}
    assert "File A" in texts
    assert "File B" in texts


def test_read_unknown_extension(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("col1,col2\na,b", encoding="utf-8")
    docs = list(load_inputs(str(f)))
    assert len(docs) == 1
    assert "col1" in docs[0].text
