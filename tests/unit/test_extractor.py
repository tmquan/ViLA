"""Unit tests for :class:`LegalExtractStage` and the two algorithm layers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

from packages.common.schemas import PipelineCfg
from packages.extractor.generic import GenericExtractor
from packages.extractor.precedent import PrecedentExtractor, _iso_date
from packages.extractor.stage import LegalExtractStage


def test_iso_date_zero_pads() -> None:
    assert _iso_date("5", "3", "2021") == "2021-03-05"
    assert _iso_date("15", "11", "2023") == "2023-11-15"


def test_generic_extractor_picks_up_articles_dates_courts() -> None:
    text = (
        "Tại Tòa án nhân dân cấp cao Hà Nội, ngày 15/06/2020, "
        "hội đồng áp dụng khoản 1 Điều 173 BLHS 2015."
    )
    rec = GenericExtractor().extract("DOC1", text)
    assert rec.doc_id == "DOC1"
    tags = {e.tag for e in rec.entities}
    assert {"DATE", "ORG-COURT", "ARTICLE"} <= tags
    assert any(s.article == 173 and s.clause == 1 for s in rec.statute_refs)


def test_precedent_extractor_from_metadata_and_text() -> None:
    text = "Án lệ số 47/2021/AL. Nội dung án lệ: áp dụng Điều 173 BLHS 2015."
    generic = GenericExtractor().extract("DOCX", text)
    prec = PrecedentExtractor().extract(
        doc_id="DOCX",
        markdown=text,
        scraper_metadata={
            "adopted_date": "15/06/2021",
            "applied_article": "Điều 173 BLHS 2015",
        },
        generic=generic,
    )
    assert prec.precedent_number == "Án lệ số 47/2021/AL"
    assert prec.adopted_date == "2021-06-15"
    assert prec.applied_article_code == "BLHS"
    assert prec.applied_article_number == 173


def _cfg(*, run_site: bool = True) -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.extractor.run_site_layer = run_site
    return cfg


def _make_batch() -> DocumentBatch:
    df = pd.DataFrame(
        {
            "doc_name": ["A"],
            "markdown": ["Án lệ số 10/2021/AL. Điều 174 BLHS 2015."],
            "adopted_date": ["01/01/2021"],
            "applied_article": ["Điều 174 BLHS 2015"],
        }
    )
    return DocumentBatch(task_id="t", dataset_name="anle", data=df)


def test_legal_extract_stage_adds_expected_columns() -> None:
    stage = LegalExtractStage(cfg=_cfg())
    stage.setup(None)
    out = stage.process(_make_batch()).to_pandas()

    assert out.loc[0, "text_hash"]
    assert out.loc[0, "char_len"] == len("Án lệ số 10/2021/AL. Điều 174 BLHS 2015.")
    assert out.loc[0, "precedent_number"].startswith("Án lệ số 10/2021/AL")
    assert out.loc[0, "applied_article_number"] == 174
    extracted = out.loc[0, "extracted"]
    assert any(s["article"] == 174 for s in extracted["statute_refs"])


def test_legal_extract_stage_respects_site_layer_flag() -> None:
    stage = LegalExtractStage(cfg=_cfg(run_site=False))
    stage.setup(None)
    out = stage.process(_make_batch()).to_pandas()
    # precedent_* columns are None when site layer is disabled but still present.
    assert out.loc[0, "precedent_number"] is None
    assert "precedent_number" in out.columns
