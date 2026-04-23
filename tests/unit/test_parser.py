"""Unit tests for AnleParser with a fake nemotron-parse client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from packages.scrapers.anle.parser import AnleParser, ParseResult
from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.schemas import PipelineCfg


class FakeNemotronClient:
    """Returns canned per-page output without touching the network."""

    def __init__(self, pages: int = 2, confidence: float = 0.9) -> None:
        self._pages = pages
        self._confidence = confidence

    def parse(self, pdf_bytes: bytes, *, preserve_tables: bool = True) -> dict[str, Any]:
        return {
            "pages": [
                {"page_number": i + 1,
                 "blocks": [{"type": "Title", "text": f"Page {i+1}"}],
                 "markdown": f"# Page {i+1}\nĐiều 173 BLHS 2015."}
                for i in range(self._pages)
            ],
            "markdown": "# Page 1\nĐiều 173 BLHS 2015.",
            "confidence": self._confidence,
        }


def _seed_pdf(layout: SiteLayout, doc_id: str, n: int = 1) -> None:
    layout.ensure_dirs(layout.pdf_dir, layout.metadata_dir)
    (layout.pdf_dir / f"{doc_id}.pdf").write_bytes(b"%PDF-1.4\n...")


def test_parse_one_happy_path(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed_pdf(layout, "TAND1")
    _seed_pdf(layout, "TAND2")

    cfg = OmegaConf.structured(PipelineCfg)
    parser = AnleParser(
        cfg=cfg,
        layout=layout,
        client=FakeNemotronClient(pages=3),
        num_workers=2,
    )

    counts = parser.run()
    assert counts["seen"] == 2
    assert counts["processed"] == 2
    assert counts["errored"] == 0

    # md/ and json/ exist and pass is_item_complete.
    for doc_id in ("TAND1", "TAND2"):
        assert (layout.md_dir / f"{doc_id}.md").exists()
        assert (layout.json_dir / f"{doc_id}.json").exists()
        assert parser.is_item_complete(doc_id)
        js = json.loads((layout.json_dir / f"{doc_id}.json").read_text(encoding="utf-8"))
        assert js["num_pages"] == 3


def test_parse_skips_already_complete(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed_pdf(layout, "TAND1")
    cfg = OmegaConf.structured(PipelineCfg)

    client = FakeNemotronClient(pages=1)
    p1 = AnleParser(cfg=cfg, layout=layout, client=client)
    p1.run()

    # Second run on the same layout should skip everything.
    p2 = AnleParser(cfg=cfg, layout=layout, client=client)
    counts = p2.run()
    assert counts["skipped"] == 1
    assert counts["processed"] == 0


def test_parse_doc_filter_limits_to_single(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    _seed_pdf(layout, "TANDA")
    _seed_pdf(layout, "TANDB")
    cfg = OmegaConf.structured(PipelineCfg)

    parser = AnleParser(
        cfg=cfg, layout=layout, client=FakeNemotronClient(), doc_filter="TANDA"
    )
    counts = parser.run()
    assert counts["seen"] == 1
    assert counts["processed"] == 1


def test_parse_is_item_complete_rejects_zero_pages(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    layout.ensure_dirs(layout.md_dir, layout.json_dir)
    (layout.md_dir / "X.md").write_text("hi", encoding="utf-8")
    (layout.json_dir / "X.json").write_text(
        json.dumps({"num_pages": 0}), encoding="utf-8"
    )
    cfg = OmegaConf.structured(PipelineCfg)
    parser = AnleParser(cfg=cfg, layout=layout, client=FakeNemotronClient())
    assert not parser.is_item_complete("X")
