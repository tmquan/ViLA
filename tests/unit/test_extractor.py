"""Unit tests for the two-layer extractor."""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from packages.scrapers.anle.extractor import (
    AnleExtractor,
    GenericExtractor,
    AnlePrecedentExtractor,
    _iso_date,
)
from packages.scrapers.common.base import SiteLayout
from packages.scrapers.common.schemas import PipelineCfg


def test_iso_date_zero_pads() -> None:
    assert _iso_date("5", "3", "2021") == "2021-03-05"
    assert _iso_date("15", "11", "2023") == "2023-11-15"


def test_generic_extractor_picks_up_articles_dates_courts() -> None:
    # Vietnamese legal writing conventionally places `khoản` before `điều`.
    text = (
        "Tại Tòa án nhân dân cấp cao Hà Nội, ngày 15/06/2020, "
        "hội đồng áp dụng khoản 1 Điều 173 BLHS 2015."
    )
    rec = GenericExtractor().extract("DOC1", text)
    assert rec.doc_id == "DOC1"
    assert rec.char_len == len(text)
    tags = {e.tag for e in rec.entities}
    assert "DATE" in tags
    assert "ORG-COURT" in tags
    assert "ARTICLE" in tags
    # Statute ref resolved with clause.
    assert any(s.article == 173 and s.clause == 1 for s in rec.statute_refs)


def test_precedent_extractor_from_metadata_and_text() -> None:
    text = "Án lệ số 47/2021/AL. Nội dung án lệ: áp dụng Điều 173 BLHS 2015."
    generic = GenericExtractor().extract("DOCX", text)
    prec = AnlePrecedentExtractor().extract(
        doc_id="DOCX",
        markdown=text,
        scraper_metadata={
            "adopted_date": "15/06/2021",
            "applied_article": "Điều 173 BLHS 2015",
        },
        generic=generic,
    )
    assert prec.doc_id == "DOCX"
    assert prec.precedent_number == "Án lệ số 47/2021/AL"
    assert prec.adopted_date == "2021-06-15"
    assert prec.applied_article_code == "BLHS"
    assert prec.applied_article_number == 173
    assert prec.principle_text and "áp dụng Điều 173" in prec.principle_text


def test_extractor_stage_writes_jsonl(tmp_path: Path) -> None:
    layout = SiteLayout(output_root=tmp_path, host="anle.toaan.gov.vn")
    layout.ensure_dirs(layout.md_dir, layout.metadata_dir)
    (layout.md_dir / "A.md").write_text(
        "Án lệ số 10/2021/AL. Điều 174 BLHS 2015.", encoding="utf-8"
    )
    (layout.metadata_dir / "A.json").write_text(
        json.dumps({"adopted_date": "01/01/2021", "applied_article": "Điều 174 BLHS 2015"}),
        encoding="utf-8",
    )

    cfg = OmegaConf.structured(PipelineCfg)
    ex = AnleExtractor(cfg=cfg, layout=layout)
    counts = ex.run()
    assert counts["processed"] == 1

    gen = (layout.jsonl_dir / "generic_extracted.jsonl").read_text(encoding="utf-8")
    rows = [json.loads(line) for line in gen.splitlines() if line.strip()]
    assert rows[0]["doc_id"] == "A"
    assert any(s["article"] == 174 for s in rows[0]["statute_refs"])

    prec = (layout.jsonl_dir / "precedents.jsonl").read_text(encoding="utf-8")
    prows = [json.loads(line) for line in prec.splitlines() if line.strip()]
    assert prows[0]["precedent_number"].startswith("Án lệ số 10/2021/AL")
    assert prows[0]["applied_article_number"] == 174
