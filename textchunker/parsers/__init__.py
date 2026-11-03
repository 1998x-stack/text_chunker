"""Adapters to DocumentIR. MarkdownIt blocks and HTML DOM are supported; PDF layout geometry is deferred."""

from .text import parse_text, parse_extracted_text
from .markdown import parse_markdown
from .html import parse_html
from .pdf import parse_pdf_pages


def parse_document(text: str, source_uri: str, format: str = "text"):
    if format in ("text", "txt"):
        return parse_text(text, source_uri)
    if format in ("markdown", "md"):
        return parse_markdown(text, source_uri)
    if format in ("html", "htm"):
        return parse_html(text, source_uri)
    raise ValueError(f"Unsupported raw document format: {format}")


__all__ = [
    "parse_document", "parse_text", "parse_extracted_text",
    "parse_markdown", "parse_html", "parse_pdf_pages",
]
