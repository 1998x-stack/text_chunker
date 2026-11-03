"""AST-backed Markdown and DOM-aware HTML document contracts."""
from textchunker.core.spans import Span
from textchunker.parsers import parse_document, parse_html, parse_markdown


def _assert_full_block_coverage(document):
    cursor = 0
    for block in document.blocks:
        assert block.span.start == cursor
        assert block.span.end > cursor
        cursor = block.span.end
    assert cursor == len(document.text)


def test_markdown_ast_preserves_all_unicode_and_blank_lines():
    source = (
        "前言🚀\n\n# 一级标题\n\nSome **bold** content.\n\n"
        "~~~python\n# This is code, not a heading\nprint('hello')\n~~~\n\n"
        "| Col | Value |\n| --- | --- |\n| 甲 | 🚀 |\n\n"
        "## 二级标题\n最后一段。\n"
    )
    doc = parse_markdown(source, "manual.md")
    assert doc.metadata["ast"] is True
    assert doc.text == source
    _assert_full_block_coverage(doc)
    kinds = [item.kind for item in doc.blocks]
    assert "code" in kinds
    assert "table" in kinds
    assert "heading" in kinds
    assert [item.title for item in doc.blocks if item.kind == "heading"] == [
        "一级标题", "二级标题"
    ]
    code = next(block for block in doc.blocks if block.kind == "code")
    assert "# This is code, not a heading" in code.span.extract(doc.text)
    assert not any(item.title == "This is code, not a heading"
                   for item in doc.blocks)


def test_markdown_setext_heading_and_list_parent():
    text = "Heading\n=======\n\n- First\n- Second\n\nEnd\n"
    doc = parse_markdown(text)
    _assert_full_block_coverage(doc)
    assert any(item.kind == "heading" and item.title == "Heading"
               and item.heading_level == 1 for item in doc.blocks)
    assert any(item.kind == "list" for item in doc.blocks)


def test_markdown_empty_and_whitespace_only():
    assert parse_markdown("").blocks == ()
    doc = parse_markdown("\n\n \n")
    _assert_full_block_coverage(doc)
    assert doc.source_map.resolve(Span(0, len(doc.text)))[0].mapping == "exact"


def test_html_dom_semantic_blocks_not_duplicated():
    html = (
        "<html><body><script>secret()</script><h1>Guide</h1>"
        "<div><p>First <b>paragraph</b></p></div>"
        "<pre><code>line 1\n#not heading</code></pre>"
        "<table><tr><th>Key</th><th>Value</th></tr>"
        "<tr><td>A</td><td>🚀</td></tr></table>"
        "<blockquote><p>Quoted <b>content</b></p></blockquote>"
        "</body></html>"
    )
    doc = parse_html(html, "guide.html")
    _assert_full_block_coverage(doc)
    assert "secret()" not in doc.text
    assert doc.text.count("First paragraph") == 1
    assert [item.kind for item in doc.blocks] == [
        "heading", "paragraph", "code", "table", "quote"
    ]
    assert "Key | Value\nA | 🚀" in doc.text
    assert doc.metadata["source_offsets"] == "unavailable"
    assert all(location.source_span is None for location in doc.source_map.resolve(
        Span(0, len(doc.text))))
    assert all(location.block_id for location in doc.source_map.resolve(
        Span(0, len(doc.text))))


def test_raw_html_parse_document_routes_through_dom():
    doc = parse_document("<h2>Title</h2><p>Text</p>", "a.html", "html")
    assert doc.format == "html"
    assert doc.blocks[0].heading_level == 2
    assert doc.blocks[0].title == "Title"


def test_html_without_semantic_tags_uses_fallback():
    doc = parse_html("<div>Loose <span>words</span></div>", "unknown.html")
    assert doc.text == "Loose words"
    assert len(doc.blocks) == 1
    assert doc.blocks[0].kind == "paragraph"
