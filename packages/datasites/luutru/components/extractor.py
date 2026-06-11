"""luutru DocumentExtractor: detail HTML -> structured row fields.

Subclasses :class:`nemo_curator.stages.text.download.base.DocumentExtractor`.
The ``xemchitietvanban.htm`` detail page renders document metadata as a
``<table class="table table-bordered">`` of label/value rows::

    <tr><td><b>Số hiệu</b></td><td>08/2026/TT-BNV</td></tr>
    <tr><td><b>Trích yếu nội dung</b></td><td>...</td></tr>
    <tr><td><b>Ngày ban hành</b></td><td>15/05/2026</td></tr>
    ...

We parse that label->value map and project it onto English-stem
columns (``doc_number``, ``issue_date``, ``issuing_authority``, ...);
the source values stay Vietnamese per wiki/DATASITES.md § 3.4. Curator
runs this per iterated record; the returned dict becomes one row of the
produced :class:`DocumentBatch`.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from bs4 import BeautifulSoup
from nemo_curator.stages.text.download.base import DocumentExtractor

#: Vietnamese label (normalised, lower-cased, accent-stripped) ->
#: English-stem column. Matching is accent-insensitive + substring so
#: minor portal wording changes ("Ngày ban hành" vs "Ngày ký") still
#: land on the right column.
_LABEL_TO_FIELD: tuple[tuple[str, str], ...] = (
    ("so hieu", "doc_number"),
    ("trich yeu", "summary"),
    ("hinh thuc van ban", "legal_type"),
    ("linh vuc", "legal_area"),
    ("co quan ban hanh", "issuing_authority"),
    ("nguoi ky", "signer"),
    ("ngay ban hanh", "issue_date"),
    ("ngay het hieu luc", "expiry_date"),
    ("ngay hieu luc", "effective_date"),
)

#: Document-form short codes derived from ``legal_type`` (Hình thức văn
#: bản). ``doc_type`` is the short code; ``legal_type`` keeps the full
#: Vietnamese name. Unmapped forms fall back to the full string.
_DOC_TYPE_CODES: tuple[tuple[str, str], ...] = (
    ("thong tu lien tich", "TTLT"),
    ("thong tu", "TT"),
    ("nghi dinh", "NĐ"),
    ("nghi quyet", "NQ"),
    ("quyet dinh", "QĐ"),
    ("chi thi", "CT"),
    ("cong van", "CV"),
    ("thong bao", "TB"),
    ("ke hoach", "KH"),
    ("huong dan", "HD"),
    ("bao cao", "BC"),
    ("to trinh", "TTr"),
    ("phap lenh", "PL"),
    ("quy che", "QC"),
    ("quy dinh", "QyĐ"),
    ("luat", "Luật"),
)

#: All metadata columns this extractor emits (besides the iterator
#: passthrough keys). Kept in one place so the row shape stays stable
#: even when a label is missing on a given detail page.
_META_FIELDS: tuple[str, ...] = (
    "doc_number",
    "doc_type",
    "legal_type",
    "legal_area",
    "issuing_authority",
    "signer",
    "summary",
    "issue_date",
    "effective_date",
    "expiry_date",
)


def _ascii_fold(text: str) -> str:
    """Lower-case + strip Vietnamese diacritics for label matching."""
    text = text.replace("đ", "d").replace("Đ", "D")
    nfkd = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", no_accents).strip().lower()


class LuutruDocumentExtractor(DocumentExtractor):
    """Parse the vanban detail HTML into the luutru row shape."""

    def __init__(self, cfg: Any) -> None:
        self._host = str(cfg.host)

    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        meta = self._parse_detail(record.get("detail_html", ""))
        row: dict[str, Any] = {
            "doc_name": record["doc_name"],
            "source": self._host,
            "detail_url": record.get("detail_url", ""),
            "pdf_path": record.get("pdf_path", ""),
            "pdf_bytes": record.get("pdf_bytes", b""),
            "pdf_url": meta.get("pdf_url"),
        }
        for field in _META_FIELDS:
            row[field] = meta.get(field)
        return row

    def input_columns(self) -> list[str]:
        # Columns produced by :class:`LuutruDocumentIterator`.
        return ["doc_name", "pdf_path", "pdf_bytes", "detail_html", "detail_url"]

    def output_columns(self) -> list[str]:
        return [
            "doc_name",
            "source",
            "detail_url",
            "pdf_path",
            "pdf_bytes",
            "pdf_url",
            *_META_FIELDS,
        ]

    # ------------------------------------------------------ internals

    def _parse_detail(self, html: str) -> dict[str, Any]:
        if not html:
            return {}
        soup = BeautifulSoup(html, "html.parser")
        label_map = _parse_label_table(soup)

        out: dict[str, Any] = {}
        for folded_label, value in label_map.items():
            for needle, field in _LABEL_TO_FIELD:
                if needle in folded_label and field not in out:
                    out[field] = value or None
                    break

        legal_type = out.get("legal_type")
        if legal_type:
            out["doc_type"] = _doc_type_code(legal_type)

        out["pdf_url"] = _first_pdf_href(soup)
        return out


def _parse_label_table(soup: BeautifulSoup) -> dict[str, str]:
    """Return ``{folded_label: value}`` from the bordered metadata table.

    Walks every ``<tr>`` with two-plus ``<td>`` cells; the first cell's
    text is the label (typically wrapped in ``<b>``), the second cell's
    text is the value. Accent-folds the label key for robust matching.
    """
    out: dict[str, str] = {}
    for table in soup.select("table.table-bordered, table.table"):
        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            label = _ascii_fold(cells[0].get_text(separator=" "))
            value = re.sub(r"\s+", " ", cells[1].get_text(separator=" ")).strip()
            if label and label not in out:
                out[label] = value
    return out


def _first_pdf_href(soup: BeautifulSoup) -> str | None:
    for selector in (
        "a[href*='dms.luutru.gov.vn']",
        "a[href$='.pdf']",
        "a[href*='.pdf']",
    ):
        node = soup.select_one(selector)
        if node is not None and node.get("href"):
            return str(node["href"])
    return None


def _doc_type_code(legal_type: str) -> str:
    folded = _ascii_fold(legal_type)
    for needle, code in _DOC_TYPE_CODES:
        if needle in folded:
            return code
    return legal_type


__all__ = ["LuutruDocumentExtractor"]
