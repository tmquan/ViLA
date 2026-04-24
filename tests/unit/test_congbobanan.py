"""Unit tests for the congbobanan datasite primitives + pipeline build."""

from __future__ import annotations

from typing import Any

import pytest
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from omegaconf import OmegaConf

from packages.common.schemas import PipelineCfg
from packages.datasites.congbobanan.components import (
    CongbobananDocumentExtractor,
    CongbobananURLGenerator,
    doc_id_from_url,
)
from packages.datasites.congbobanan.components.downloader import page_has_metadata
from packages.datasites.congbobanan.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_pipeline,
)


def _cfg(tmp_path: Any, **overrides: Any) -> Any:
    cfg = OmegaConf.structured(PipelineCfg)
    cfg.host = "congbobanan.toaan.gov.vn"
    cfg.output_dir = str(tmp_path)
    cfg.parser.runtime = "local"
    cfg.extractor.run_site_layer = False
    cfg.scraper.start_id = 1
    cfg.scraper.end_id = 5
    cfg.scraper.verify_tls = False
    for k, v in overrides.items():
        OmegaConf.update(cfg, k, v, merge=False)
    return cfg


# --------------------------------------------------------------------- URL generator


def test_url_generator_emits_integer_id_range(tmp_path: Any) -> None:
    gen = CongbobananURLGenerator(_cfg(tmp_path, **{"scraper.start_id": 10, "scraper.end_id": 12}))
    urls = gen.generate_urls()
    assert urls == [
        "https://congbobanan.toaan.gov.vn/2ta10t1cvn/chi-tiet-ban-an",
        "https://congbobanan.toaan.gov.vn/2ta11t1cvn/chi-tiet-ban-an",
        "https://congbobanan.toaan.gov.vn/2ta12t1cvn/chi-tiet-ban-an",
    ]


def test_url_generator_raises_on_inverted_range(tmp_path: Any) -> None:
    with pytest.raises(ValueError):
        CongbobananURLGenerator(_cfg(tmp_path, **{"scraper.start_id": 100, "scraper.end_id": 10}))


def test_doc_id_from_url_handles_every_url_family() -> None:
    assert doc_id_from_url(
        "https://congbobanan.toaan.gov.vn/2ta12345t1cvn/chi-tiet-ban-an"
    ) == "12345"
    assert doc_id_from_url(
        "https://congbobanan.toaan.gov.vn/3ta12345t1cvn/"
    ) == "12345"
    assert doc_id_from_url(
        "https://congbobanan.toaan.gov.vn/5ta12345t1cvn/filename.pdf"
    ) == "12345"
    assert doc_id_from_url("https://example.com/no-id-here") is None


# --------------------------------------------------------------------- ghost-page guard


def test_page_has_metadata_detects_real_panel() -> None:
    assert page_has_metadata(
        '<div class="panel panel-blue search_left_pub details_pub">'
        '<label>Bản án số:</label><span>03/2022/DSST</span>'
        "</div>"
    )
    assert page_has_metadata(
        '<section class="search_left_pub details_pub">'
        '<label>Quyết định số:</label><span>77/2021</span>'
        "</section>"
    )


def test_page_has_metadata_rejects_ghost() -> None:
    assert not page_has_metadata("")
    assert not page_has_metadata("<html><body>empty</body></html>")
    # Sidebar class but no case number label.
    assert not page_has_metadata(
        '<div class="search_left_pub details_pub">nothing useful</div>'
    )


# --------------------------------------------------------------------- extractor


_FIXTURE_HTML = """
<html><body>
  <div class="panel panel-blue search_left_pub details_pub">
    <label>Bản án số:</label><span>03/2022/DSST ngày 23/11/2022</span>
    <i class="fa-eye"></i> 1,234
    <i class="fa-download"></i> 56
    <label>Tên bản án:</label><span>Vụ án Tranh chấp hợp đồng (15.12.2022)</span>
    <label>Quan hệ pháp luật:</label><span>Tranh chấp hợp đồng mua bán tài sản</span>
    <label>Cấp xét xử:</label><span>Sơ thẩm</span>
    <label>Loại vụ/việc:</label><span>Dân sự</span>
    <label>Tòa án xét xử:</label><span>TAND tỉnh Bắc Ninh</span>
    <label>Áp dụng án lệ:</label><span>Không</span>
    <label>Đính chính:</label><span>Không</span>
    <label>Thông tin về vụ/việc:</label><span>Hai bên tranh chấp việc thanh toán.</span>
    <span>Tổng số lượt được bình chọn làm nguồn phát triển án lệ: 7</span>
    <a href="/5ta1213296t1cvn/03-2022-DSST_ban-an.pdf">Tải về</a>
  </div>
  <div class="Detail_Feedback_pub"></div>
</body></html>
"""


def test_extractor_parses_every_sidebar_field(tmp_path: Any) -> None:
    ex = CongbobananDocumentExtractor(_cfg(tmp_path))
    out = ex.extract(
        {
            "doc_name": "1213296",
            "case_id": "1213296",
            "pdf_path": "/tmp/1213296.pdf",
            "pdf_bytes": b"%PDF-1.4",
            "detail_html": _FIXTURE_HTML,
            "detail_url": "https://congbobanan.toaan.gov.vn/2ta1213296t1cvn/chi-tiet-ban-an",
        }
    )
    assert out is not None
    assert out["case_id"] == "1213296"
    assert out["source"] == "congbobanan.toaan.gov.vn"
    assert out["doc_type"] == "ban-an"
    assert out["ban_an_so"] == "03/2022/DSST"
    assert out["ngay"] == "23/11/2022"
    assert out["luot_xem"] == 1234
    assert out["luot_tai"] == 56
    assert out["ten_ban_an"] == "Vụ án Tranh chấp hợp đồng"
    assert out["ngay_cong_bo"] == "15.12.2022"
    assert out["quan_he_phap_luat"].startswith("Tranh chấp hợp đồng")
    assert out["cap_xet_xu"] == "Sơ thẩm"
    assert out["loai_vu_viec"] == "Dân sự"
    assert out["toa_an_xet_xu"] == "TAND tỉnh Bắc Ninh"
    assert out["tong_binh_chon"] == "7"
    assert out["pdf_filename"] == "03-2022-DSST_ban-an.pdf"


def test_extractor_handles_quyet_dinh_variant(tmp_path: Any) -> None:
    html = (
        '<div class="panel panel-blue search_left_pub details_pub">'
        "<label>Quyết định số:</label><span>77/2021 ngày 01/02/2021</span>"
        "</div>"
        '<div class="Detail_Feedback_pub"></div>'
    )
    ex = CongbobananDocumentExtractor(_cfg(tmp_path))
    out = ex.extract(
        {
            "doc_name": "99",
            "case_id": "99",
            "detail_html": html,
            "detail_url": "",
            "pdf_bytes": b"",
            "pdf_path": "",
        }
    )
    assert out is not None
    assert out["doc_type"] == "quyet-dinh"
    assert out["ban_an_so"] == "77/2021"
    assert out["ngay"] == "01/02/2021"


def test_extractor_on_empty_html_returns_blank_row(tmp_path: Any) -> None:
    ex = CongbobananDocumentExtractor(_cfg(tmp_path))
    out = ex.extract(
        {
            "doc_name": "42",
            "case_id": "42",
            "detail_html": "",
            "detail_url": "",
            "pdf_bytes": b"",
            "pdf_path": "",
        }
    )
    assert out is not None
    assert out["case_id"] == "42"
    assert out["ban_an_so"] is None
    assert out["toa_an_xet_xu"] is None
    assert out["luot_xem"] == 0


# --------------------------------------------------------------------- pipeline build


def test_pipeline_registry_shape() -> None:
    assert list(PIPELINES.keys()) == ["download", "parse", "extract", "embed", "reduce"]
    assert ALL_PIPELINES_ORDER == list(PIPELINES.keys())


def test_every_pipeline_builds(tmp_path: Any) -> None:
    for name in ALL_PIPELINES_ORDER:
        pipeline = build_pipeline(_cfg(tmp_path), name)
        assert isinstance(pipeline, Pipeline)
        assert name in pipeline.name


def test_every_stage_is_a_processing_or_composite_stage(tmp_path: Any) -> None:
    for name in ALL_PIPELINES_ORDER:
        pipeline = build_pipeline(_cfg(tmp_path), name)
        for stage in pipeline.stages:
            assert isinstance(stage, (ProcessingStage, CompositeStage)), (
                f"pipeline={name} stage={stage!r} is not a Curator stage"
            )


def test_every_pipeline_describes_without_error(tmp_path: Any) -> None:
    for name in ALL_PIPELINES_ORDER:
        pipeline = build_pipeline(_cfg(tmp_path), name)
        text = pipeline.describe()
        assert "Pipeline:" in text
        pipeline.build()
