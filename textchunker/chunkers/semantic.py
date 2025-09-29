from __future__ import annotations
from typing import List
import numpy as np  # type: ignore
from loguru import logger
from sentence_transformers import SentenceTransformer  # type: ignore
from ..types import Chunk
from ..utils import whitespace_sentences, count_tokens
from ..registry import register
from .base import BaseChunker


@register("semantic")
class SemanticChunker(BaseChunker):
    """语义自适应分块：按句子级相似度断点进行合并。"""

    def __init__(self, cfg):
        super().__init__(cfg)
        model_name = self.cfg.semantic.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Loading sentence-transformers model: {model_name}")
        self.model = SentenceTransformer(model_name)

    def chunk(self, text: str) -> List[Chunk]:
        c = self.cfg.common
        s = self.cfg.semantic
        size = int(c.get("chunk_size", 600))
        overlap = int(c.get("chunk_overlap", 100))
        min_sim = float(s.get("min_similarity", 0.62))
        window = int(s.get("sentence_window", 1))
        max_chunks = c.get("max_chunks")

        sents = whitespace_sentences(text)
        if not sents:
            return [Chunk(id=0, text=text, start=0, end=len(text), meta={"strategy": "semantic"})]

        embs = self.model.encode(sents, normalize_embeddings=True)
        chunks: List[Chunk] = []
        buf: List[str] = []
        start_char = 0

        # 预先计算每句的起止字符位置
        offsets = []
        cursor = 0
        for s in sents:
            beg = text.find(s, cursor)
            end = beg + len(s)
            offsets.append((beg, end))
            cursor = end

        last_end = offsets[0][1]

        def flush(end_char: int, chunk_start: int) -> None:
            nonlocal buf
            piece = "".join(buf).strip()
            if piece:
                cid = len(chunks)
                chunks.append(Chunk(id=cid, text=piece, start=chunk_start, end=end_char,
                                    meta={"strategy": "semantic"}))
            buf = []

        for i, sent in enumerate(sents):
            sent_start, sent_end = offsets[i]

            if buf and i >= window:
                v = embs[i]
                ctx = embs[i - window:i].mean(axis=0)
                sim = float(np.dot(v, ctx))
                if sim < min_sim:
                    flush(sent_start, start_char)
                    if max_chunks and len(chunks) >= max_chunks:
                        return chunks
                    start_char = sent_start

            if not buf:
                start_char = sent_start

            buf.append(sent + " ")
            last_end = sent_end

            if count_tokens("".join(buf)) >= size:
                flush(last_end, start_char)
                if max_chunks and len(chunks) >= max_chunks:
                    break

                if overlap > 0 and i >= 0:
                    back_chars = max(0, last_end - overlap)
                    j = i
                    while j >= 0 and offsets[j][0] > back_chars:
                        j -= 1
                    if j + 1 <= i:
                        buf = [sents[j + 1] + " "]
                        start_char = offsets[j + 1][0]
                    else:
                        buf = []
                        start_char = last_end
                else:
                    buf = []
                    start_char = last_end
                continue

            if max_chunks and len(chunks) >= max_chunks:
                break

        if buf and (not max_chunks or len(chunks) < max_chunks):
            flush(last_end, start_char)

        return chunks
