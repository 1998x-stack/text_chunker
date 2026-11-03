"""Multi-format readers with optional libraries imported only when needed."""

from __future__ import annotations

import os
from typing import Iterable

from .types import FileDoc


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="strict") as stream:
        return stream.read()


def _read_html(path: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(_read_text(path), "html.parser")
    return "\n".join(
        element.get_text(" ", strip=True)
        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li"]
        )
    )


def _read_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_inputs(path: str) -> Iterable[FileDoc]:
    """Yield deterministic document order for reproducible exports."""
    if not path:
        raise ValueError("Input path is required")

    if os.path.isdir(path):
        for root, directories, files in os.walk(path):
            directories.sort()
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                if ext in {".txt", ".md"}:
                    yield FileDoc(path=filepath, text=_read_text(filepath))
                elif ext in {".htm", ".html"}:
                    yield FileDoc(
                        path=filepath, text=_read_html(filepath),
                        meta={"format": "html"},
                    )
                elif ext == ".pdf":
                    yield FileDoc(
                        path=filepath, text=_read_pdf(filepath),
                        meta={"format": "pdf"},
                    )
        return

    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        yield FileDoc(path=path, text=_read_text(path))
    elif ext in {".htm", ".html"}:
        yield FileDoc(
            path=path, text=_read_html(path), meta={"format": "html"}
        )
    elif ext == ".pdf":
        yield FileDoc(path=path, text=_read_pdf(path), meta={"format": "pdf"})
    else:
        yield FileDoc(path=path, text=_read_text(path))
