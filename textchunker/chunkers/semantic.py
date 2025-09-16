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
        buf, start_char = [], 0

        # 预先计算每句的起止字符位置
        offsets = []
        cursor = 0
        for s in sents:
            beg = text.find(s, cursor)
            end = beg + len(s)
            offsets.append((beg, end))
            cursor = end

        def flush(end_char: int):
            cid = len(chunks)
            piece = "".join(buf).strip()
            if piece:
                chunks.append(Chunk(id=cid, text=piece, start=start_char, end=end_char,
                                    meta={"strategy": "semantic"}))

        for i, sent in enumerate(sents):
            buf.append(sent + " ")
            start_char = offsets[i][0] if not chunks and not buf[:-1] else start_char
            # 超长直接断
            if count_tokens("".join(buf)) >= size:
                end_char = offsets[i][1]
                flush(end_char)
                # 重叠回退若干句
                if overlap > 0 and i > 0:
                    back_chars = max(0, end_char - overlap)
                    # 重新起点：找到 back_chars 所在的句开始
                    j = i
                    while j >= 0 and offsets[j][0] > back_chars:
                        j -= 1
                    buf = [sents[j + 1] + " "] if j + 1 <= i else []
                    start_char = offsets[j + 1][0] if j + 1 <= i else end_char
                else:
                    buf, start_char = [], end_char
                if max_chunks and len(chunks) >= max_chunks:
                    break
                continue

            # 相似度断点检测（与前 window 句平均相似度）
            if i >= window:
                v = embs[i]
                ctx = embs[i - window:i].mean(axis=0)
                sim = float(np.dot(v, ctx))
                if sim < min_sim:
                    end_char = offsets[i][0]
                    flush(end_char)
                    buf, start_char = [sent + " "], offsets[i][0]

            # 收尾
            if i == len(sents) - 1 and buf:
                end_char = offsets[i][1]
                flush(end_char)

        return chunks
