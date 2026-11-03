"""Readers for legacy plain text and structure-preserving DocumentIR ingestion.

The legacy load_inputs(path) API still exposes extracted HTML and PDF strings.
Set preserve_structure=True (used by CLI) to retain raw HTML and PDF pages.
"""
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


def _read_pdf_pages(path: str) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


def _read_pdf(path: str) -> str:
    return "\n\n".join(_read_pdf_pages(path))


def _load_one(path: str, preserve_structure: bool = False,
              pdf_engine: str = "pypdf") -> FileDoc:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".txt", ".md"}:
        return FileDoc(path=path, text=_read_text(path))
    if ext in {".htm", ".html"}:
        return FileDoc(
            path=path,
            text=_read_text(path) if preserve_structure else _read_html(path),
            meta={"format": "html", "raw_html": preserve_structure},
        )
    if ext == ".pdf":
        if preserve_structure and pdf_engine == "docling":
            return FileDoc(path=path, text="",
                           meta={"format": "docling", "source_path": path})
        if preserve_structure:
            pages = _read_pdf_pages(path)
            return FileDoc(
                path=path, text="\n\n".join(pages),
                meta={"format": "pdf", "pages": pages},
            )
        return FileDoc(
            path=path, text=_read_pdf(path),
            meta={"format": "pdf"},
        )
    return FileDoc(path=path, text=_read_text(path))


def load_inputs(path: str, *, preserve_structure: bool = False,
                pdf_engine: str = "pypdf") -> Iterable[FileDoc]:
    """Yield deterministic document order; preserve old extracted API by default."""
    if not path:
        raise ValueError("Input path is required")
    if pdf_engine not in {"pypdf", "docling"}:
        raise ValueError("pdf_engine must be pypdf or docling")

    if os.path.isdir(path):
        for root, directories, files in os.walk(path):
            directories.sort()
            for filename in sorted(files):
                filepath = os.path.join(root, filename)
                if os.path.splitext(filename)[1].lower() in {
                    ".txt", ".md", ".htm", ".html", ".pdf"
                }:
                    yield _load_one(filepath, preserve_structure, pdf_engine)
        return
    yield _load_one(path, preserve_structure, pdf_engine)
