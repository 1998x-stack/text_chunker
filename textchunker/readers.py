from __future__ import annotations
import os
from typing import Iterable, List
from bs4 import BeautifulSoup  # type: ignore
from pypdf import PdfReader  # type: ignore
from .types import FileDoc


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_html(path: str) -> str:
    html = _read_text(path)
    soup = BeautifulSoup(html, "html.parser")
    # 将可见文本抽取并保持基本段落
    return "\n".join([t.get_text(" ", strip=True) for t in soup.find_all(["h1","h2","h3","h4","h5","h6","p","li"])])


def _read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n\n".join(pages)


def load_inputs(path: str) -> Iterable[FileDoc]:
    """支持目录/单文件的统一读取。"""
    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in files:
                fp = os.path.join(root, fn)
                ext = os.path.splitext(fn)[-1].lower()
                if ext in {".txt", ".md"}:
                    yield FileDoc(path=fp, text=_read_text(fp))
                elif ext in {".html", ".htm"}:
                    yield FileDoc(path=fp, text=_read_html(fp), meta={"format": "html"})
                elif ext == ".pdf":
                    yield FileDoc(path=fp, text=_read_pdf(fp), meta={"format": "pdf"})
    else:
        ext = os.path.splitext(path)[-1].lower()
        if ext in {".txt", ".md"}:
            yield FileDoc(path=path, text=_read_text(path))
        elif ext in {".html", ".htm"}:
            yield FileDoc(path=path, text=_read_html(path), meta={"format": "html"})
        elif ext == ".pdf":
            yield FileDoc(path=path, text=_read_pdf(path), meta={"format": "pdf"})
        else:
            # 默认尝试按文本
            yield FileDoc(path=path, text=_read_text(path))
