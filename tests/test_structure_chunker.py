import sys
import types
import pytest

if "bs4" not in sys.modules:
    stub = types.ModuleType("bs4")
    stub.BeautifulSoup = None
    sys.modules["bs4"] = stub

if "sentence_transformers" not in sys.modules:
    st_stub = types.ModuleType("sentence_transformers")
    st_stub.SentenceTransformer = None
    sys.modules["sentence_transformers"] = st_stub

if "numpy" not in sys.modules:
    import numpy
    sys.modules["numpy"] = numpy

from textchunker.types import StrategyConfig
from textchunker.chunkers.structure import StructureChunker

def _cfg(**overrides):
    common = {"chunk_size": 100, "chunk_overlap": 0}
    structure = {"prefer": "md", "sub_split": True}
    common.update(overrides.get("common", {}))
    structure.update(overrides.get("structure", {}))
    return StrategyConfig(
        name="structure", common=common, structure=structure,
        semantic={}, recursive={}, fixed={}, llm={},
    )

def test_structure_chunker_markdown_sections():
    text = "# Title 1\nContent for section one.\n\n# Title 2\nContent for section two."
    chunker = StructureChunker(_cfg(common={"chunk_size": 500, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 2
    assert "Title 1" in chunks[0].meta.get("title", "")
    assert "Title 2" in chunks[1].meta.get("title", "")

def test_structure_chunker_no_headers():
    text = "Just plain text without any headers at all."
    chunker = StructureChunker(_cfg())
    chunks = chunker.chunk(text)
    assert len(chunks) >= 1
    assert chunks[0].meta["title"] == "Document"

def test_structure_chunker_sub_split():
    long_section = "# Big Section\n" + "Word " * 200
    chunker = StructureChunker(_cfg(common={"chunk_size": 50, "chunk_overlap": 0}))
    chunks = chunker.chunk(long_section)
    assert len(chunks) > 1
    for c in chunks:
        assert c.meta["title"] == "Big Section"

def test_structure_chunker_offset_correctness():
    text = "# A\nShort.\n\n# B\nAlso short."
    chunker = StructureChunker(_cfg(common={"chunk_size": 500, "chunk_overlap": 0}))
    chunks = chunker.chunk(text)
    for c in chunks:
        assert c.start >= 0
        assert c.end <= len(text)

def test_structure_chunker_max_chunks():
    text = "# S1\nA.\n\n# S2\nB.\n\n# S3\nC.\n\n# S4\nD."
    chunker = StructureChunker(_cfg(common={"chunk_size": 500, "chunk_overlap": 0, "max_chunks": 2}))
    chunks = chunker.chunk(text)
    assert len(chunks) == 2
