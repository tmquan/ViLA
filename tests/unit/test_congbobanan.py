"""Unit tests for the congbobanan datasite primitives + pipeline build."""

from __future__ import annotations

from typing import Any

import pytest

from packages.datasites.congbobanan.components import (
    CBBADocumentExtractor,
    CBBADocumentURLGenerator,
    doc_id_from_url,
)
from packages.datasites.congbobanan.components.downloader import (
    _MIN_VALID_PDF_BYTES,
    ACCEPTED_BODY_EXTENSIONS,
    CBBADocumentPDFDownloader,
    _is_valid_pdf,
    _sniff_body_ext,
    page_has_metadata,
)


# --------------------------------------------------------------------- URL generator


def test_url_generator_emits_integer_id_range() -> None:
    gen = CBBADocumentURLGenerator(start_id=10, end_id=12)
    urls = gen.generate_urls()
    assert urls == [
        "https://congbobanan.toaan.gov.vn/2ta10t1cvn/chi-tiet-ban-an",
        "https://congbobanan.toaan.gov.vn/2ta11t1cvn/chi-tiet-ban-an",
        "https://congbobanan.toaan.gov.vn/2ta12t1cvn/chi-tiet-ban-an",
    ]


def test_url_generator_raises_on_inverted_range() -> None:
    with pytest.raises(ValueError):
        CBBADocumentURLGenerator(start_id=100, end_id=10)


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


# --------------------------------------------------------------------- body-format sniff


def _well_formed_pdf(size: int = 4096) -> bytes:
    """Synthetic PDF body with the required header and ``%%EOF`` trailer.

    Sized at ``size`` bytes (>= ``_MIN_VALID_PDF_BYTES``) so the
    sniffer's size + magic + trailer gates all pass.
    """
    body = b"%PDF-1.5\n%\xb5\xb5\xb5\xb5\n1 0 obj\n"
    trailer = b"\n%%EOF\n"
    pad = b"\x00" * max(0, size - len(body) - len(trailer))
    return body + pad + trailer


def test_is_valid_pdf_accepts_real_pdf(tmp_path: Any) -> None:
    """A %PDF-prefixed, ``%%EOF``-terminated body must pass validation."""
    p = tmp_path / "real.pdf"
    p.write_bytes(_well_formed_pdf())
    assert _is_valid_pdf(str(p))


def test_is_valid_pdf_rejects_html_error_page(tmp_path: Any) -> None:
    """500-error pages claim ``application/pdf`` but ship HTML; reject them."""
    p = tmp_path / "error.html_pretending_to_be.pdf"
    # 6 475 bytes of HTML is the typical congbobanan 500 body shape; the
    # important bit is that the first 5 bytes are *not* ``%PDF-``.
    p.write_bytes(b"<html><head><title>500 Internal Server Error</title>" + b"x" * 8192)
    assert not _is_valid_pdf(str(p))


def test_is_valid_pdf_rejects_truncated_body(tmp_path: Any) -> None:
    """Anything below ``_MIN_VALID_PDF_BYTES`` is rejected outright."""
    p = tmp_path / "tiny.pdf"
    p.write_bytes(b"%PDF-1.5\n" + b"\x00" * (_MIN_VALID_PDF_BYTES // 2))
    assert not _is_valid_pdf(str(p))


def test_is_valid_pdf_returns_false_on_missing_path(tmp_path: Any) -> None:
    """Non-existent paths must not raise; callers depend on a clean bool."""
    assert not _is_valid_pdf(str(tmp_path / "does-not-exist.pdf"))


def test_is_valid_pdf_rejects_pdf_without_eof_trailer(tmp_path: Any) -> None:
    """Truncated PDFs (header OK, no ``%%EOF`` in tail) are rejected.

    Streaming downloads that drop mid-transfer leave behind a file that
    starts with ``%PDF-`` but never reaches the end-of-file marker; the
    parser stage can't recover from those.
    """
    p = tmp_path / "truncated.pdf"
    p.write_bytes(b"%PDF-1.5\n" + b"\x00" * 4096)  # no %%EOF anywhere
    assert not _is_valid_pdf(str(p))


def test_sniff_body_ext_classifies_each_known_format(tmp_path: Any) -> None:
    """Every magic header in :data:`_BODY_MAGICS` must round-trip to its extension."""
    pdf = tmp_path / "case.pdf"
    pdf.write_bytes(_well_formed_pdf())
    assert _sniff_body_ext(str(pdf)) == ".pdf"

    # DOCX (Office Open XML): ZIP magic ``PK\x03\x04``.
    docx = tmp_path / "case.docx"
    docx.write_bytes(b"PK\x03\x04" + b"\x00" * 4096)
    assert _sniff_body_ext(str(docx)) == ".docx"

    # Legacy DOC: OLE2 / CFB magic.
    doc = tmp_path / "case.doc"
    doc.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 4096)
    assert _sniff_body_ext(str(doc)) == ".doc"

    # RTF: ``{\rtf`` text header. Allowed to be small (100-byte min).
    rtf = tmp_path / "case.rtf"
    rtf.write_bytes(b"{\\rtf1\\ansi\\deff0 Hello}" + b"\x00" * 200)
    assert _sniff_body_ext(str(rtf)) == ".rtf"


def test_sniff_body_ext_rejects_unknown_payloads(tmp_path: Any) -> None:
    """JSON / HTML / RAR / executables observed in the live corpus are rejected."""
    for name, payload in (
        ("json", b'{"sections":[]}' + b" " * 4096),
        ("html", b"<html><body>404</body></html>" + b"x" * 4096),
        ("rar",  b"Rar!\x1a\x07\x01\x00" + b"\x00" * 4096),
        ("exe",  b"MZ\x90\x00\x03\x00" + b"\x00" * 4096),
        ("rand", b"\x00" * 4096),
    ):
        p = tmp_path / f"junk_{name}.bin"
        p.write_bytes(payload)
        assert _sniff_body_ext(str(p)) is None, f"sniffer accepted {name}"


def test_existing_body_path_finds_any_accepted_extension(tmp_path: Any) -> None:
    """A previous run may have saved ``<id>.docx``; the existence check must see it.

    Locks the contract that prevents the downloader from re-fetching a
    case that was already saved under a non-``.pdf`` extension.
    """
    dl = CBBADocumentPDFDownloader(str(tmp_path), pages_dir=str(tmp_path / "pages"))
    for ext in ACCEPTED_BODY_EXTENSIONS:
        case_id = f"42{ext.replace('.', '_')}"
        body = tmp_path / f"{case_id}{ext}"
        body.write_bytes(b"\x00" * 4096)
        found = dl._existing_body_path(case_id)
        assert found == body, f"missed {ext}"

    # Empty body must be ignored (otherwise we'd skip a re-download that needs to happen).
    empty_id = "99"
    (tmp_path / f"{empty_id}.pdf").write_bytes(b"")
    assert dl._existing_body_path(empty_id) is None

    # Truly absent case_id returns None.
    assert dl._existing_body_path("does-not-exist") is None


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
    ex = CBBADocumentExtractor()
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
    ex = CBBADocumentExtractor()
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
    ex = CBBADocumentExtractor()
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


# NOTE: congbobanan's retired five-pipeline `--pipeline` registry
# (ALL_PIPELINES_ORDER / build_pipeline) moved out with the migration to
# the single-IP in-process runner; those build-shape tests were removed
# with it. The crawl+extract composite is exercised end-to-end via
# `pipeline.py` / `extract_text.py`.
