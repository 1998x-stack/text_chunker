from __future__ import annotations
import re
from typing import List, Tuple
from loguru import logger
from bs4 import BeautifulSoup  # type: ignore
from ..types import Chunk
from ..registry import register
from .base import BaseChunker
from ..utils import count_tokens
from .recursive import _split_by_separators


@register("structure")
class StructureChunker(BaseChunker):
    """基于文档结构：Markdown/HTML/PDF（文本抽出后按标题）"""

    def _md_sections(self, text: str) -> List[Tuple[str, int, int]]:
        # 依据 Markdown 标题分段：以 '#' 开头的行作为起点
        lines = text.splitlines(keepends=True)
        heads = []
        for i, ln in enumerate(lines):
            if re.match(r"^\s*#{1,6}\s+", ln):
                heads.append(i)
        heads.append(len(lines))
        spans: List[Tuple[str, int, int]] = []
        for i in range(len(heads) - 1):
            beg_line = heads[i]
            end_line = heads[i + 1]
            beg = sum(len(l) for l in lines[:beg_line])
            end = sum(len(l) for l in lines[:end_line])
            title = re.sub(r"^#+\s+", "", lines[beg_line]).strip()
            spans.append((title, beg, end))
        if not spans:  # 没检测到标题则整文做一个区块
            spans = [("Document", 0, len(text))]
        return spans

    def _html_sections(self, text: str) -> List[Tuple[str, int, int]]:
        # 已在 reader 中转成纯文本，此处退化为按双换行
        return [("HTML", 0, len(text))]

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        st = self.cfg.structure
        size = int(c.get("chunk_size", 800))
        overlap = int(c.get("chunk_overlap", 100))
        prefer = st.get("prefer", "auto")
        sub_split = bool(st.get("sub_split", True))
        max_chunks = c.get("max_chunks")

        if prefer in {"md", "auto"}:
            spans = self._md_sections(text)
        elif prefer == "html":
            spans = self._html_sections(text)
        else:  # pdf/auto
            spans = self._md_sections(text)  # 没有更好的结构时，按段落回退

        chunks: List[Chunk] = []
        for title, beg, end in spans:
            seg = text[beg:end]
            if count_tokens(seg) <= size or not sub_split:
                chunks.append(Chunk(id=len(chunks), text=seg, start=beg, end=end,
                                    meta={"strategy": "structure", "title": title}))
            else:
                parts = _split_by_separators(seg, ["\n\n", "\n", " ", ""], size)
                cursor = beg
                for p in parts:
                    s = text.find(p, cursor)
                    e = s + len(p)
                    chunks.append(Chunk(id=len(chunks), text=p, start=s, end=e,
                                        meta={"strategy": "structure", "title": title}))
                    cursor = e
            if max_chunks and len(chunks) >= max_chunks:
                break

        # 简单重叠拼接
        if overlap > 0 and chunks:
            for i in range(1, len(chunks)):
                prev, cur = chunks[i - 1], chunks[i]
                head = text[max(cur.start - overlap, prev.start):cur.start]
                cur.text = head + cur.text
                cur.start = cur.start - len(head)

        return chunks
