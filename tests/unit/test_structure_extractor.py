"""Unit tests for :class:`LegalStructureExtractor`."""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.extractor.normalization import normalize_text
from packages.extractor.structure import (
    SCHEMA_VERSION,
    SECTION_KINDS,
    DocumentStructure,
    LegalStructureExtractor,
)


def _ext() -> LegalStructureExtractor:
    return LegalStructureExtractor()


def test_split_pages_uses_page_headings() -> None:
    md = (
        "## Page 1\n\nHello body 1\n\n## Page 2\n\nBody 2 line A\nBody 2 line B"
    )
    out = _ext().extract(doc_id="X", markdown=md)
    assert out.stats.num_pages == 2
    assert out.sections[0].kind == "header"
    assert {p.page for p in out.paragraphs} == {1, 2}


def test_pages_default_to_one_when_no_heading() -> None:
    md = "A first sentence. A second sentence."
    out = _ext().extract(doc_id="X", markdown=md)
    assert out.stats.num_pages == 1
    assert out.paragraphs[0].page == 1


def test_canonical_sections_are_detected() -> None:
    md = (
        "## Page 1\n\n"
        "TÒA ÁN NHÂN DÂN HUYỆN A\nBản án số: 12/2021/DS-PT\nNgày: 11-3-2021\n"
        "“V/v: Tranh chấp hợp đồng”\n\n"
        "NỘI DUNG VỤ ÁN:\n\n"
        "Nguyên đơn trình bày rằng A là chủ sở hữu hợp pháp.\n\n"
        "NHẬN ĐỊNH CỦA TÒA ÁN:\n\n"
        "[1] Hội đồng xét xử thấy rằng yêu cầu khởi kiện có căn cứ.\n"
        "[2] Bị đơn không cung cấp được chứng cứ phản bác.\n\n"
        "QUYẾT ĐỊNH:\n\n"
        "1. Chấp nhận yêu cầu khởi kiện.\n\n"
        "2. Bị đơn phải bồi thường 100.000.000đ.\n\n"
        "Nơi nhận:\n- Đương sự\n- Lưu hồ sơ.\n"
    )
    out = _ext().extract(doc_id="DOCX", markdown=md)

    kinds = [s.kind for s in out.sections]
    # All five canonical section kinds appear in declaration order.
    assert kinds == ["header", "case_summary", "findings", "decision", "footer"]
    # Section labels keep the raw heading text.
    findings = next(s for s in out.sections if s.kind == "findings")
    assert "NHẬN ĐỊNH" in (findings.label or "")
    assert findings.page_start == 1


def test_paragraph_markers_classify_kinds() -> None:
    md = (
        "## Page 1\n\nNHẬN ĐỊNH:\n\n"
        "[1] Hội đồng xét xử nhận định.\n\n"
        "[4.1] Điều 3 của hợp đồng quy định.\n\n"
        "QUYẾT ĐỊNH:\n\n"
        "1/ Chấp nhận yêu cầu.\n\n"
        "- Điểm bổ sung.\n"
    )
    out = _ext().extract(doc_id="P", markdown=md)
    by_kind: dict[str, list[str]] = {}
    for p in out.paragraphs:
        by_kind.setdefault(p.kind, []).append(p.marker or "")
    assert "[1]" in by_kind.get("numbered_finding", [])
    assert "[4.1]" in by_kind.get("numbered_finding", [])
    assert "1/" in by_kind.get("numbered_decision", [])
    assert any(m == "-" for m in by_kind.get("list_item", []))


def test_sentence_segmentation_links_back_to_paragraph() -> None:
    md = (
        "## Page 1\n\nNHẬN ĐỊNH:\n\n"
        "[1] Câu thứ nhất kết thúc đây. Câu thứ hai cũng kết thúc đây."
        " Còn một câu nữa.\n"
    )
    out = _ext().extract(doc_id="S", markdown=md)
    finding_pars = [p for p in out.paragraphs if p.section_kind == "findings"]
    assert finding_pars, "expected at least one finding paragraph"
    par = finding_pars[0]
    assert len(par.sentence_ids) >= 2
    assert all(
        sid.startswith(par.paragraph_id.split("#")[0]) for sid in par.sentence_ids
    )
    par_sentences = [s for s in out.sentences if s.paragraph_id == par.paragraph_id]
    # Sentences cover the paragraph in order.
    indices = [s.index_in_paragraph for s in par_sentences]
    assert indices == sorted(indices)


def test_meta_extraction_pulls_doc_code_subject_court() -> None:
    md = (
        "## Page 1\n\n"
        "TÒA ÁN NHÂN DÂN THÀNH PHỐ CẦN THƠ\n"
        "Bản án số: 38/2021/DS-PT\n"
        "Ngày: 11-3-2021\n"
        "“V/v: Tranh chấp hợp đồng đặt cọc”\n\n"
        "NỘI DUNG VỤ ÁN:\n\nNguyên đơn trình bày...\n"
    )
    out = _ext().extract(doc_id="TAND_X", markdown=md)
    assert out.meta is not None
    assert out.meta.doc_code == "38/2021/DS-PT"
    assert out.meta.doc_number == "38"
    assert out.meta.year == 2021
    assert out.meta.case_type == "dan_su"
    assert out.meta.doc_subtype == "phuc_tham"
    assert out.meta.doc_type == "ban_an"
    assert out.meta.subject == "Tranh chấp hợp đồng đặt cọc"
    assert out.meta.issue_date == "2021-03-11"
    assert out.meta.court_level == "tinh"
    assert "CẦN THƠ" in (out.meta.issuing_authority or "")


def test_meta_falls_back_to_scraper_metadata() -> None:
    md = "## Page 1\n\nĐây là một văn bản không có header rõ ràng.\n"
    out = _ext().extract(
        doc_id="X",
        markdown=md,
        scraper_metadata={
            "title": "Án lệ số 47/2021/AL về tranh chấp",
            "adopted_date": "15/06/2021",
            "precedent_number": "Án lệ số 47/2021/AL",
            "court": "Hội đồng Thẩm phán",
        },
    )
    assert out.meta is not None
    assert out.meta.precedent_number == "Án lệ số 47/2021/AL"
    assert out.meta.issue_date == "2021-06-15"
    assert out.meta.title == "Án lệ số 47/2021/AL về tranh chấp"


def test_to_jsonable_round_trips() -> None:
    md = "## Page 1\n\nQUYẾT ĐỊNH:\n\n1. Chấp nhận yêu cầu.\n"
    out = _ext().extract(doc_id="J", markdown=md)
    payload = out.to_jsonable()
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["doc_id"] == "J"
    assert all(s["kind"] in SECTION_KINDS for s in payload["sections"])
    assert all("sentence_ids" in p for p in payload["paragraphs"])
    assert all("text" in s and "page" in s for s in payload["sentences"])


def test_paragraph_text_collapses_pdf_softwraps() -> None:
    md = (
        "## Page 1\n\nNHẬN ĐỊNH:\n\n"
        "[1] Trong  ngày 11 tháng 3 năm 2021 tại Trụ sở Tòa án nhân dân thà nh\n"
        "phố Cần Thơ xét xử phúc thẩm vụ án.\n"
    )
    out = _ext().extract(doc_id="W", markdown=md)
    finding = next(p for p in out.paragraphs if p.kind == "numbered_finding")
    assert "\n" not in finding.text
    assert "  " not in finding.text  # all whitespace collapsed


# ----------------------------------------------------- real-doc smoke


_MD_DIR = Path(__file__).resolve().parents[2] / "data" / "anle.toaan.gov.vn" / "md"


@pytest.mark.skipif(
    not _MD_DIR.exists(), reason="anle md fixture not available"
)
def test_real_anle_doc_segments_into_canonical_sections() -> None:
    sample = _MD_DIR / "TAND192001.md"
    if not sample.exists():
        pytest.skip("TAND192001.md not present")
    # Match the upstream contract: stage runs `normalize_text` first.
    md = normalize_text(sample.read_text(encoding="utf-8"))
    out = _ext().extract(doc_id="TAND192001", markdown=md)
    kinds = {s.kind for s in out.sections}
    # Real anle bản án has all five canonical kinds.
    assert {"header", "case_summary", "findings", "decision", "footer"} <= kinds
    assert isinstance(out, DocumentStructure)
    assert out.stats.num_paragraphs > 10
    assert out.stats.num_sentences > out.stats.num_paragraphs
    # Meta from the real doc.
    assert out.meta is not None
    assert out.meta.doc_code == "38/2021/DS-PT"
    assert out.meta.case_type == "dan_su"
    assert out.meta.doc_subtype == "phuc_tham"


@pytest.mark.skipif(
    not _MD_DIR.exists(), reason="anle md fixture not available"
)
def test_real_anle_doc_with_old_orthography() -> None:
    """TAND192022 uses the pre-1984 orthography 'TOÀ ÁN NHÂN DÂN'.

    After upstream normalization the heading should canonicalize to
    'TÒA ÁN NHÂN DÂN' so the issuing-authority extractor latches on
    to the letterhead and not the secretary's affiliation later in
    the doc.
    """
    sample = _MD_DIR / "TAND192022.md"
    if not sample.exists():
        pytest.skip("TAND192022.md not present")
    md = normalize_text(sample.read_text(encoding="utf-8"))
    out = _ext().extract(doc_id="TAND192022", markdown=md)
    assert out.meta is not None
    body = out.meta.issuing_authority or ""
    # First match is the letterhead, with full multi-line qualifier.
    assert "TÒA ÁN NHÂN DÂN" in body
    assert "QUẬN" in body
    assert "HÀ NỘI" in body
    # Court level is the most specific qualifier present (district).
    assert out.meta.court_level == "huyen"
