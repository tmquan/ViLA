"""Unit tests for AnleScraper (pure logic, no real HTTP).

Covers:
    - listing HTML parsing (_parse_listing)
    - detail HTML parsing (_parse_detail)
    - is_item_complete resume check against filesystem
    - _extract_doc_name helper edge cases
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from packages.scrapers.anle.scraper import (
    AnleScraper,
    _extract_doc_name,
    _first_text,
    _merge_selectors,
)
from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.schemas import PipelineCfg


def _cfg() -> Any:
    return OmegaConf.structured(PipelineCfg)


class _FakeSession:
    """No-op session; scraper tests that need network use mocks or skip."""


@pytest.fixture()
def scraper(tmp_path: Path) -> AnleScraper:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    return AnleScraper(
        cfg=_cfg(),
        layout=layout,
        session=_FakeSession(),    # type: ignore[arg-type]
        limit=None,
        force=False,
        resume=True,
    )


def test_extract_doc_name_prefers_ddocname() -> None:
    assert _extract_doc_name("/foo/chitietanle?dDocName=TAND123") == "TAND123"
    assert _extract_doc_name(
        "https://anle.toaan.gov.vn/chitietanle?dDocName=TAND349038&x=1"
    ) == "TAND349038"


def test_extract_doc_name_falls_back_to_path_tail() -> None:
    assert _extract_doc_name("/some/path/abc123") == "abc123"
    assert _extract_doc_name("/foo/bar.pdf?x=1") == "bar.pdf"


def test_merge_selectors_replaces_per_key() -> None:
    base = {"a": ["x"], "b": ["y"]}
    override = {"a": ["z", "w"]}
    merged = _merge_selectors(base, override)
    assert merged == {"a": ["z", "w"], "b": ["y"]}


def test_parse_listing_finds_precedents(scraper: AnleScraper) -> None:
    html = """
    <html><body>
      <a href="/chitietanle?dDocName=TAND111">Án lệ số 1</a>
      <a href="/chitietanle?dDocName=TAND222">Án lệ số 2</a>
      <a href="/chitietanle?dDocName=TAND111">duplicate</a>
    </body></html>
    """
    items = list(scraper._parse_listing(html, "https://anle.toaan.gov.vn/"))
    ids = [i["doc_name"] for i in items]
    assert "TAND111" in ids
    assert "TAND222" in ids


def test_parse_detail_reads_pdf_link(scraper: AnleScraper) -> None:
    html = """
    <html><body>
      <h1>Án lệ số 47/2021/AL</h1>
      <a href="/webcenter/.../foo.pdf">Tải về</a>
    </body></html>
    """
    header = scraper._parse_detail(html)
    assert header["precedent_number"].startswith("Án lệ số 47/2021/AL")
    assert header["pdf_url"].endswith("foo.pdf")
    assert header["pdf_url"].startswith("https://")


def test_is_item_complete_requires_both_files(
    tmp_path: Path, scraper: AnleScraper
) -> None:
    assert not scraper.is_item_complete("MISSING")

    pdf = scraper.layout.pdf_dir / "TANDX.pdf"
    meta = scraper.layout.metadata_dir / "TANDX.json"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    meta.parent.mkdir(parents=True, exist_ok=True)

    pdf.write_bytes(b"")  # zero byte => incomplete
    meta.write_text("{}", encoding="utf-8")
    assert not scraper.is_item_complete("TANDX")

    pdf.write_bytes(b"%PDF-1.4\n...")
    assert scraper.is_item_complete("TANDX")


def test_is_item_complete_rejects_invalid_metadata_json(
    tmp_path: Path, scraper: AnleScraper
) -> None:
    (scraper.layout.pdf_dir / "A.pdf").parent.mkdir(parents=True, exist_ok=True)
    (scraper.layout.metadata_dir / "A.json").parent.mkdir(parents=True, exist_ok=True)
    (scraper.layout.pdf_dir / "A.pdf").write_bytes(b"pdf")
    (scraper.layout.metadata_dir / "A.json").write_text("not json", encoding="utf-8")
    assert not scraper.is_item_complete("A")
