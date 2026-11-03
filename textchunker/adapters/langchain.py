"""Convert verified chunk records to LangChain Documents."""
from __future__ import annotations


def to_langchain_documents(result):
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise ImportError(
            "Install langchain-core to export LangChain Documents"
        ) from exc

    return [
        Document(
            page_content=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "document_id": result.document_id,
                "source": (
                    chunk.source_locations[0].source_uri
                    if chunk.source_locations else ""
                ),
                "start": chunk.span.start,
                "end": chunk.span.end,
                "heading_path": list(chunk.heading_path),
                "tokenizer": result.tokenizer_id,
            },
        )
        for chunk in result.chunks
    ]
