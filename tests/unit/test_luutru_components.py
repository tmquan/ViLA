"""Unit tests for the four luutru Curator primitives.

Exercises the site-specific logic with offline HTML/PDF fixtures (no
network): the GUID ``id=`` doc-name regex, the listing-page link
parser, the detail-page label/value metadata extractor, the
attachment-anchor PDF resolution, and the per-doc iterator sidecar
round-trip.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from packages.datasites.luutru.components import (
    LuutruDocumentExtractor,
    LuutruDocumentIterator,
    LuutruURLGenerator,
    extract_doc_name,
    extract_doc_name_from_url,
)
from packages.datasites.luutru.components.downloader import LuutruDocumentDownloader

_GUID = "24c897ce-09b7-4962-98aa-a0588fffa617"

_DETAIL_HTML = f"""
<html><body>
  <table class="table table-bordered">
    <tbody>
      <tr><td class="col1"><b>Số hiệu</b></td><td>08/2026/TT-BNV</td></tr>
      <tr><td><b>Trích yếu nội dung</b></td><td>Thông tư quy định chi tiết ...</td></tr>
      <tr><td><b>Ngày ban hành</b></td><td>15/05/2026</td></tr>
      <tr><td><b>Ngày hiệu lực</b></td><td>15/05/2026</td></tr>
      <tr><td><b>Ngày hết hiệu lực</b></td><td></td></tr>
      <tr><td><b>Hình thức văn bản</b></td><td>Thông tư</td></tr>
      <tr><td><b>Lĩnh vực</b></td><td>Văn bản quy phạm pháp luật và hướng dẫn nghiệp vụ</td></tr>
      <tr><td><b>Cơ quan ban hành</b></td><td>Bộ Nội vụ</td></tr>
      <tr><td><b>Người ký duyệt</b></td><td>Thứ trưởng Nguyễn Mạnh Khương</td></tr>
      <tr><td><b>Tệp đính kèm</b></td>
          <td><a href="https://dms.luutru.gov.vn/files/ecm/source_files/2026/05/25/x.pdf">Tải về</a></td></tr>
    </tbody>
  </table>
</body></html>
"""

_LISTING_HTML = """
<html><body>
  <a href="/xemchitietvanban.htm?id=11111111-1111-1111-1111-111111111111">Doc A</a>
  <a href="/xemchitietvanban.htm?id=22222222-2222-2222-2222-222222222222">Doc B</a>
  <a href="/xemchitietvanban.htm?id=11111111-1111-1111-1111-111111111111">Doc A dup</a>
  <a href="/home.htm">not a doc</a>
  <a href="/vanban.aspx?type=all&p=2">page 2</a>
</body></html>
"""


def _cfg() -> object:
    return OmegaConf.create({
        "host": "luutru.gov.vn",
        "scraper": {
            "qps": 5.0,
            "user_agent": "test",
            "timeout_s": 5.0,
            "max_retries": 1,
            "verify_tls": True,
            "fetch_detail_page": True,
            "fetch_head_before_download": False,
            "num_workers": 2,
            "extra_params": {"type": "all", "p": ""},
        },
    })


# ----------------------------------------------------- doc-name regex


def test_extract_doc_name_from_id_query() -> None:
    url = f"https://luutru.gov.vn/xemchitietvanban.htm?id={_GUID}"
    assert extract_doc_name(url) == _GUID
    assert extract_doc_name_from_url(url) == _GUID


def test_extract_doc_name_rejects_non_detail() -> None:
    assert extract_doc_name("https://luutru.gov.vn/home.htm") is None
    assert extract_doc_name("/vanban.aspx?type=all&p=2") is None


# ----------------------------------------------------- listing parse


def test_listing_parser_dedupes_guids() -> None:
    gen = LuutruURLGenerator(_cfg())
    rows = list(gen._parse_listing(_LISTING_HTML))
    # Two distinct GUIDs; the duplicate anchor still yields the GUID
    # (dedup happens in generate_urls, not _parse_listing).
    assert rows.count("11111111-1111-1111-1111-111111111111") == 2
    assert "22222222-2222-2222-2222-222222222222" in rows
    assert all(len(r) == 36 for r in rows)


def test_url_generator_detail_template() -> None:
    gen = LuutruURLGenerator(_cfg())
    assert gen._detail_template.format(doc_name=_GUID).endswith(f"id={_GUID}")
    assert gen._page_param == "p"


# ----------------------------------------------------- detail-page metadata


def test_extractor_parses_metadata_table() -> None:
    ex = LuutruDocumentExtractor(_cfg())
    row = ex.extract({
        "doc_name": _GUID,
        "detail_url": f"https://luutru.gov.vn/xemchitietvanban.htm?id={_GUID}",
        "pdf_path": f"pdf/{_GUID}.pdf",
        "pdf_bytes": b"%PDF-1.4",
        "detail_html": _DETAIL_HTML,
    })
    assert row is not None
    assert row["doc_name"] == _GUID
    assert row["source"] == "luutru.gov.vn"
    assert row["doc_number"] == "08/2026/TT-BNV"
    assert row["legal_type"] == "Thông tư"
    assert row["doc_type"] == "TT"               # short code derived from form
    assert row["legal_area"].startswith("Văn bản quy phạm")
    assert row["issuing_authority"] == "Bộ Nội vụ"
    assert row["signer"] == "Thứ trưởng Nguyễn Mạnh Khương"
    assert row["issue_date"] == "15/05/2026"
    assert row["effective_date"] == "15/05/2026"
    assert row["expiry_date"] is None             # empty cell -> None
    assert row["summary"].startswith("Thông tư quy định")
    assert row["pdf_url"].startswith("https://dms.luutru.gov.vn/")


def test_extractor_output_columns_stable_on_empty_html() -> None:
    ex = LuutruDocumentExtractor(_cfg())
    row = ex.extract({
        "doc_name": _GUID, "detail_url": "", "pdf_path": "",
        "pdf_bytes": b"", "detail_html": "",
    })
    assert row is not None
    # Every metadata column is present (None-valued) so the row shape
    # stays stable across documents.
    for col in ex.output_columns():
        assert col in row


def test_extractor_doc_type_code_mapping() -> None:
    from packages.datasites.luutru.components.extractor import _doc_type_code

    assert _doc_type_code("Thông tư") == "TT"
    assert _doc_type_code("Nghị định") == "NĐ"
    assert _doc_type_code("Quyết định") == "QĐ"
    assert _doc_type_code("Thông tư liên tịch") == "TTLT"
    # Unmapped form falls back to the full Vietnamese string.
    assert _doc_type_code("Văn bản lạ") == "Văn bản lạ"


# ----------------------------------------------------- PDF anchor resolution


def test_downloader_resolves_dms_attachment() -> None:
    dl = LuutruDocumentDownloader(_cfg(), download_dir="/tmp/x")
    pdf_url = dl._resolve_pdf_url(_DETAIL_HTML)
    assert pdf_url == "https://dms.luutru.gov.vn/files/ecm/source_files/2026/05/25/x.pdf"


def test_downloader_resolves_relative_attachment() -> None:
    dl = LuutruDocumentDownloader(_cfg(), download_dir="/tmp/x")
    html = '<a href="/files/abc.pdf">Tải về</a>'
    assert dl._resolve_pdf_url(html) == "https://luutru.gov.vn/files/abc.pdf"


def test_downloader_no_attachment_returns_none() -> None:
    dl = LuutruDocumentDownloader(_cfg(), download_dir="/tmp/x")
    assert dl._resolve_pdf_url("<html><body>no link</body></html>") is None


def test_downloader_idempotent_skip(tmp_path: Path) -> None:
    existing = tmp_path / f"{_GUID}.pdf"
    existing.write_bytes(b"%PDF-1.4 already here")
    dl = LuutruDocumentDownloader(_cfg(), download_dir=str(tmp_path))
    out = dl.download(f"https://luutru.gov.vn/xemchitietvanban.htm?id={_GUID}")
    assert out == str(existing)


# ----------------------------------------------------- iterator round-trip


def test_iterator_reads_sidecars(tmp_path: Path) -> None:
    pdf = tmp_path / f"{_GUID}.pdf"
    pdf.write_bytes(b"%PDF-1.4 body")
    pdf.with_suffix(".html").write_text(_DETAIL_HTML, encoding="utf-8")
    url = f"https://luutru.gov.vn/xemchitietvanban.htm?id={_GUID}"
    pdf.with_suffix(".url").write_text(url, encoding="utf-8")

    it = LuutruDocumentIterator()
    records = list(it.iterate(str(pdf)))
    assert len(records) == 1
    rec = records[0]
    assert rec["doc_name"] == _GUID
    assert rec["pdf_bytes"] == b"%PDF-1.4 body"
    assert rec["detail_url"] == url
    assert "Số hiệu" in rec["detail_html"]
    assert it.output_columns() == [
        "doc_name", "pdf_path", "pdf_bytes", "detail_html", "detail_url",
    ]
