"""Convert verified chunk records to LlamaIndex TextNodes."""
from __future__ import annotations


def to_llamaindex_nodes(result):
    try:
        from llama_index.core.schema import TextNode
    except ImportError as exc:
        raise ImportError(
            "Install llama-index-core to export LlamaIndex TextNodes"
        ) from exc

    return [
        TextNode(
            text=chunk.text,
            id_=chunk.chunk_id,
            start_char_idx=chunk.span.start,
            end_char_idx=chunk.span.end,
            metadata={
                "document_id": result.document_id,
                "source": (
                    chunk.source_locations[0].source_uri
                    if chunk.source_locations else ""
                ),
                "heading_path": list(chunk.heading_path),
                "tokenizer": result.tokenizer_id,
            },
        )
        for chunk in result.chunks
    ]
