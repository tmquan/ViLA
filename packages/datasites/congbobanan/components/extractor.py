"""congbobanan DocumentExtractor.

Parses the Vietnamese-labeled sidebar in the detail HTML into flat
columns. Mirrors the reference scraper's ``parse_metadata`` (see
https://github.com/tmquan/datascraper/blob/main/congbobanan/scraper.py)
so downstream stages see identical fields whether the pipeline was
bootstrapped from the Curator path or the legacy scraper tree.

Fields emitted (all strings unless noted):

* ``doc_type``            -- ``"ban-an"`` (Bản án / judgment) or
                             ``"quyet-dinh"`` (Quyết định / decision)
* ``ban_an_so``           -- e.g. ``"03/2022/DSST"``
* ``ngay``                -- judgment date in site format (``dd/mm/yyyy``)
* ``ten_ban_an``          -- human-readable case title
* ``ngay_cong_bo``        -- publication date (``dd.mm.yyyy``)
* ``quan_he_phap_luat``   -- legal relationship / subject-matter label
* ``cap_xet_xu``          -- trial level (Sơ thẩm / Phúc thẩm / ...)
* ``loai_vu_viec``        -- case type (Dân sự / Hình sự / ...)
* ``toa_an_xet_xu``       -- court name
* ``ap_dung_an_le``       -- applied precedent (if any)
* ``dinh_chinh``          -- corrections (``đính chính``)
* ``thong_tin_vu_viec``   -- case info / summary
* ``tong_binh_chon``      -- precedent-vote count
* ``luot_xem`` / ``luot_tai`` (int) -- view / download counts
* ``pdf_filename``        -- original server-side filename pulled from
                             the ``/5ta<id>t1cvn/<filename>`` link
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any
from urllib.parse import unquote

from nemo_curator.stages.text.download.base import DocumentExtractor


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_HEADING_RE = re.compile(
    r"<label>\s*(Bản án|Quyết định) số:\s*</label>\s*<span>(.*?)</span>",
    re.DOTALL,
)
_EYE_RE = re.compile(r"fa-eye[^<]*</i>\s*([\d,.\s]+)")
_DL_RE = re.compile(r"fa-download[^<]*</i>\s*([\d,.\s]+)")
_VOTE_RE = re.compile(
    r"Tổng số lượt được bình chọn làm nguồn phát triển án lệ:\s*([\d]+)"
)
_DATE_IN_TITLE_RE = re.compile(r"\((\d{2}\.\d{2}\.\d{4})\)")
_PDF_HREF_TEMPLATE = r'href="/5ta{case_id}t1cvn/([^"]+)"'


def _strip_tags(html: str) -> str:
    text = _TAG_RE.sub(" ", html)
    text = unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _extract_between(html: str, after: str, before: str) -> str:
    idx = html.find(after)
    if idx == -1:
        return ""
    start = idx + len(after)
    end = html.find(before, start)
    if end == -1:
        return html[start : start + 500]
    return html[start:end]


def _parse_label_span(html: str, label: str) -> str:
    pattern = re.compile(
        rf"<label[^>]*>\s*{re.escape(label)}\s*</label>\s*<span[^>]*>(.*?)</span>",
        re.DOTALL | re.IGNORECASE,
    )
    m = pattern.search(html)
    return _strip_tags(m.group(1)).strip() if m else ""


class CongbobananDocumentExtractor(DocumentExtractor):
    """Parse the congbobanan sidebar panel into flat row fields."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._host = str(cfg.host)

    def input_columns(self) -> list[str]:
        # Columns produced by :class:`CongbobananDocumentIterator`.
        return [
            "doc_name",
            "case_id",
            "pdf_path",
            "pdf_bytes",
            "detail_html",
            "detail_url",
        ]

    def output_columns(self) -> list[str]:
        return [
            "doc_name",
            "case_id",
            "source",
            "detail_url",
            "pdf_path",
            "pdf_bytes",
            "doc_type",
            "ban_an_so",
            "ngay",
            "ten_ban_an",
            "ngay_cong_bo",
            "quan_he_phap_luat",
            "cap_xet_xu",
            "loai_vu_viec",
            "toa_an_xet_xu",
            "ap_dung_an_le",
            "dinh_chinh",
            "thong_tin_vu_viec",
            "tong_binh_chon",
            "luot_xem",
            "luot_tai",
            "pdf_filename",
        ]

    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        case_id = str(record.get("case_id") or record.get("doc_name") or "")
        parsed = self._parse_detail(
            record.get("detail_html", "") or "",
            case_id=case_id,
        )
        return {
            "doc_name": case_id,
            "case_id": case_id,
            "source": self._host,
            "detail_url": record.get("detail_url", ""),
            "pdf_path": record.get("pdf_path", ""),
            "pdf_bytes": record.get("pdf_bytes", b""),
            **parsed,
        }

    # --------------------------------------------------- internals

    def _parse_detail(self, html: str, *, case_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "doc_type": None,
            "ban_an_so": None,
            "ngay": None,
            "ten_ban_an": None,
            "ngay_cong_bo": None,
            "quan_he_phap_luat": None,
            "cap_xet_xu": None,
            "loai_vu_viec": None,
            "toa_an_xet_xu": None,
            "ap_dung_an_le": None,
            "dinh_chinh": None,
            "thong_tin_vu_viec": None,
            "tong_binh_chon": None,
            "luot_xem": 0,
            "luot_tai": 0,
            "pdf_filename": None,
        }
        if not html:
            return out

        panel = _extract_between(
            html,
            'class="panel panel-blue"',
            'class="Detail_Feedback_pub"',
        )
        if not panel:
            panel = html

        m = _HEADING_RE.search(panel)
        if m:
            out["doc_type"] = (
                "ban-an" if "Bản án" in m.group(1) else "quyet-dinh"
            )
            raw = _strip_tags(m.group(2))
            parts = re.split(r"\s*ngày\s*", raw, maxsplit=1)
            out["ban_an_so"] = parts[0].strip() or None
            if len(parts) > 1:
                out["ngay"] = parts[1].strip() or None

        eye = _EYE_RE.search(panel)
        if eye:
            out["luot_xem"] = int(re.sub(r"\D", "", eye.group(1)) or 0)
        dl = _DL_RE.search(panel)
        if dl:
            out["luot_tai"] = int(re.sub(r"\D", "", dl.group(1)) or 0)

        ten_raw = _parse_label_span(panel, "Tên bản án:") or _parse_label_span(
            panel, "Tên quyết định:"
        )
        if ten_raw:
            pub = _DATE_IN_TITLE_RE.search(ten_raw)
            if pub:
                out["ngay_cong_bo"] = pub.group(1)
                out["ten_ban_an"] = ten_raw[: ten_raw.find("(")].strip() or None
            else:
                out["ten_ban_an"] = ten_raw

        for label, key in (
            ("Quan hệ pháp luật:", "quan_he_phap_luat"),
            ("Cấp xét xử:", "cap_xet_xu"),
            ("Loại vụ/việc:", "loai_vu_viec"),
            ("Tòa án xét xử:", "toa_an_xet_xu"),
            ("Áp dụng án lệ:", "ap_dung_an_le"),
            ("Đính chính:", "dinh_chinh"),
            ("Thông tin về vụ/việc:", "thong_tin_vu_viec"),
        ):
            value = _parse_label_span(panel, label)
            if value:
                out[key] = value

        vote = _VOTE_RE.search(panel)
        if vote:
            out["tong_binh_chon"] = vote.group(1)

        if case_id:
            pdf_href = re.search(
                _PDF_HREF_TEMPLATE.format(case_id=re.escape(case_id)),
                html,
            )
            if pdf_href:
                out["pdf_filename"] = unquote(pdf_href.group(1))

        return out


__all__ = ["CongbobananDocumentExtractor", "page_has_metadata_fields"]


def page_has_metadata_fields(html: str) -> bool:
    """Convenience re-export used by unit tests.

    Duplicates :func:`packages.datasites.congbobanan.components.downloader.page_has_metadata`
    to avoid a circular import between extractor and downloader modules.
    """
    if not html:
        return False
    return ("Bản án số:" in html or "Quyết định số:" in html) and (
        "search_left_pub details_pub" in html
    )
