"""Docling native reading-order adapter with page/bbox provenance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from textchunker.parsers.docling_adapter import parse_docling
from textchunker.pipeline import run_chunking
from textchunker.tokenizers import CharacterTokenizer
from textchunker.types import StrategyConfig


def _item(label, text, ref, page=1, level=None):
    return SimpleNamespace(
        label=label, text=text, self_ref=ref, level=level,
        prov=[SimpleNamespace(
            page_no=page,
            bbox=SimpleNamespace(
                l=10.0, t=20.0, r=40.0, b=5.0,
                coord_origin="BOTTOMLEFT",
            ),
        )],
    )


class FakeTable:
    label = "table"
    self_ref = "#/tables/0"
    prov = [SimpleNamespace(
        page_no=2,
        bbox=SimpleNamespace(
            l=1, t=2, r=3, b=4, coord_origin="TOPLEFT",
        ),
    )]

    def export_to_markdown(self, doc):
        return "| A | B |\n| - | - |\n| x | y |"


class FakeDocument:
    def iterate_items(self):
        yield _item("section_header", "Chapter One", "#/texts/0", 1, 1), 1
        yield _item("text", "Contents here.", "#/texts/1", 1), 2
        yield FakeTable(), 1
        yield _item("picture", "", "#/pictures/0", 3), 2


def test_docling_source_maps_page_boxes_and_omission_metadata():
    doc = parse_docling(FakeDocument(), source_uri="paper.pdf")
    assert doc.format == "docling"
    assert len(doc.blocks) == 3
    assert doc.metadata["omitted_nontext_items"] == ("#/pictures/0",)
    assert doc.blocks[0].title == "Chapter One"
    assert doc.blocks[2].kind == "table"
    locations = doc.source_map.resolve(doc.blocks[2].span)
    assert locations[0].page_number == 2
    assert locations[0].bbox["coord_origin"] == "TOPLEFT"
    assert locations[0].source_span is None


def test_docling_hybrid_keeps_provenance_in_chunk_result():
    doc = parse_docling(FakeDocument(), source_uri="paper.pdf")
    cfg = StrategyConfig(
        "hybrid", common={"chunk_size": 64, "chunk_overlap": 0},
        hybrid={"oversized_element": "split"},
    )
    result = run_chunking(doc, cfg, CharacterTokenizer())
    assert result.report.ok and result.complete
    assert any(loc.page_number == 2 and loc.bbox
               for record in result.chunks
               for loc in record.source_locations)


def test_docling_adapter_rejects_other_objects():
    with pytest.raises(TypeError, match="DoclingDocument"):
        parse_docling(object())
