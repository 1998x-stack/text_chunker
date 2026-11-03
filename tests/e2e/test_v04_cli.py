"""Offline end-to-end CLI covers actual Markdown AST and HTML DOM ingestion."""
import json
import sys

from textchunker.cli import main


def test_cli_hybrid_html_and_markdown_directory(tmp_path, monkeypatch):
    folder = tmp_path / "inputs"
    folder.mkdir()
    (folder / "readme.md").write_text(
        "# Markdown\nText content.\n\n~~~py\nprint('hi')\n~~~\n",
        encoding="utf-8",
    )
    (folder / "page.html").write_text(
        "<h1>HTML page</h1><p>Paragraph</p>"
        "<table><tr><td>a</td><td>b</td></tr></table>",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "project:\n  export_format: jsonl\n  visualize: false\n"
        "strategy:\n  name: hybrid\n  common:\n"
        "    chunk_size: 38\n    chunk_overlap: 2\n"
        "  hybrid:\n    oversized_element: split\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"
    monkeypatch.setattr(sys, "argv", [
        "textchunker", "--config", str(config), "--input", str(folder),
        "--output", str(output), "--tokenizer", "char",
    ])
    main()
    records = [json.loads(line) for line in
               output.read_text(encoding="utf-8").splitlines()]
    assert records
    assert len({item["source"] for item in records}) == 2
    assert all(item["complete"] is True for item in records)
    assert all(item["tokenizer_id"] == "character" for item in records)
    assert all(item["meta"]["strategy"] == "hybrid" for item in records)
    assert any(location["mapping"] == "extracted"
               for item in records for location in item["source_locations"])
    assert any(location["mapping"] == "exact"
               for item in records for location in item["source_locations"])


def test_pdf_reader_can_provide_page_text_in_single_pass(tmp_path, monkeypatch):
    from textchunker import readers
    from textchunker.cli import _parse_input

    source = tmp_path / "fake.pdf"
    source.write_bytes(b"%PDF-mock")
    calls = []

    def read_pages(path):
        calls.append(path)
        return ["One", "Two"]

    monkeypatch.setattr(readers, "_read_pdf_pages", read_pages)
    docs = list(readers.load_inputs(str(source), preserve_structure=True))
    assert len(docs) == 1
    assert len(calls) == 1
    ir = _parse_input(docs[0])
    assert ir.format == "pdf"
    assert ir.text == "One\n\nTwo"
    assert len(calls) == 1
