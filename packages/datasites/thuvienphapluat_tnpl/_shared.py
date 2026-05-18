"""Shared helpers for the thuvienphapluat_tnpl crawler.

Holds the output-path layout builder, the JSONL field schemas for the
three on-disk artifacts emitted by the pipeline (``listings.jsonl``,
``terms.jsonl``, ``terms_translated.jsonl``), and the small fixed
VI→EN dictionaries used by the translator stage for the closed-set
columns (``lĩnh_vực``, ``tình_trạng``).

Naming convention (carried throughout this package):

* Persisted table columns are stable ASCII snake_case.
* Vietnamese text columns end in ``_vi``.
* English text columns end in ``_en``.
* Language-neutral ids / timestamps / hashes / runtime fields stay
  unsuffixed (``term_id``, ``area_id``, ``updated_at``, ...).
* The crawler never persists ``*_html`` columns; cached source HTML is
  available under ``html/items/<id>.html`` when operators need audit
  access to the original markup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.common import SiteLayout, build_layout as _build_layout_common


# -----------------------------------------------------------------------
# JSONL column schemas. Order is canonical for downstream consumers
# reading with ``pyarrow.json.read_json`` or ``pandas.read_json(lines=True)``.
# -----------------------------------------------------------------------

#: Listing-stub columns emitted by the harvester (``listings.jsonl``).
#: One row per *probed* ID (existence-of-content is decided at detail time).
LISTING_JSONL_FIELDS: list[str] = [
    "term_id",
    "is_bootstrap",     # appeared on the homepage's "latest" set
    "harvested_at",
]


#: Detail columns emitted by the downloader (``terms.jsonl``).
#:
#: The raw public JSONL deliberately contains **no** ``*_html`` columns:
#: parser internals may use source HTML, but persisted term rows keep
#: only the non-HTML text projection in ``định_nghĩa``.
DETAIL_JSONL_FIELDS: list[str] = [
    # ---- provenance --------------------------------------------------
    "term_id",
    "source",
    "source_url",
    "slug",
    "scraped_at",
    "scrape_run_id",
    # ---- Vietnamese content + classification -------------------------
    "term_name_vi",
    "term_name_en_native",
    "definition_vi",
    "area_name_vi",
    "area_id",
    "status_vi",
    "updated_by_vi",
    "updated_at_raw",
    "updated_at",
    "related_term_ids",
    "related_term_names_vi",
    # ---- derived -----------------------------------------------------
    "definition_char_len",
    "definition_word_count",
    "definition_hash",
    # ---- runtime -----------------------------------------------------
    "html_path",
    "fetch_status",
    "fetch_error",
]


#: Bilingual columns emitted by the translator (``terms_translated.jsonl``).
#: Superset of ``DETAIL_JSONL_FIELDS`` with English columns appended.
TRANSLATED_JSONL_FIELDS: list[str] = [
    *DETAIL_JSONL_FIELDS,
    # ---- English-named content twins --------------------------------
    "term_name_en",
    "definition_en",
    "area_name_en",
    "status_en",
    "updated_by_en",
    "related_term_names_en",
    # ---- translation provenance -------------------------------------
    "term_name_source",       # site | mt | null
    "definition_source",      # mt | null
    "translation_model_id",
    "translated_at",
]


#: VI column -> EN column map (content fields only).
VI_TO_EN_COLUMN_MAP: dict[str, str] = {
    "term_name_vi":            "term_name_en",
    "definition_vi":           "definition_en",
    "area_name_vi":            "area_name_en",
    "status_vi":               "status_en",
    "updated_by_vi":           "updated_by_en",
    "related_term_names_vi":   "related_term_names_en",
}


# -----------------------------------------------------------------------
# Fixed VI→EN dictionaries for the closed-set columns. Hand-curated so
# the translator never burns LLM cost on them and the same legal-domain
# name is rendered identically across every row.
# -----------------------------------------------------------------------

#: LinhVuc id -> (Vietnamese name as published by the source,
#: concise English translation). Pulled verbatim from the
#: ``<select name="ctl00$Content$SearchTNPL$ddlField">`` dropdown
#: on https://thuvienphapluat.vn/tnpl/home, then hand-translated.
#: Keys 1..47 are the closed taxonomy as of 2026-05.
LINH_VUC_VI_TO_EN: dict[str, str] = {
    "An toàn thực phẩm":                     "Food safety",
    "Bảo hiểm":                              "Insurance",
    "Bộ máy hành chính":                     "Administrative apparatus",
    "Bổ trợ Tư pháp":                        "Judicial support services",
    "Bưu chính - Viễn thông":                "Post and telecommunications",
    "Cán bộ - Công chức – Viên chức":        "Civil servants and public employees",
    "Công nghệ thông tin":                   "Information technology",
    "Chính sách xã hội":                     "Social policy",
    "Chứng khoán":                           "Securities",
    "Dân sự":                                "Civil",
    "Doanh nghiệp":                          "Enterprise",
    "Đảng":                                  "Communist Party",
    "Đất đai – Nhà ở":                       "Land and housing",
    "Đấu thầu":                              "Procurement and bidding",
    "Đầu tư":                                "Investment",
    "Điện":                                  "Electricity",
    "Giao thông vận tải":                    "Transportation",
    "Giáo dục":                              "Education",
    "Hoá chất":                              "Chemicals",
    "Hôn nhân – Gia đình – Thừa kế":         "Marriage, family and inheritance",
    "Kế toán – Kiểm toán":                   "Accounting and auditing",
    "Khiếu nại – Tố cáo":                    "Complaints and denunciations",
    "Khoa học – Công nghệ":                  "Science and technology",
    "Lao động – Tiền lương":                 "Labor and wages",
    "Lĩnh vực khác":                         "Other",
    "Nông – Lâm - Ngư nghiệp":               "Agriculture, forestry and fisheries",
    "Phòng cháy chữa cháy":                  "Fire prevention and firefighting",
    "Quốc phòng – An ninh":                  "National defense and security",
    "Sở hữu trí tuệ":                        "Intellectual property",
    "Tài chính":                             "Finance",
    "Tài nguyên – Môi trường":               "Natural resources and environment",
    "Thủ tục tố tụng":                       "Litigation procedure",
    "Thủ tục hành chính":                    "Administrative procedure",
    "Thi đua - Khen thưởng - Kỷ luật":       "Emulation, commendation and discipline",
    "Thuế - Phí – Lệ phí":                   "Taxes, fees and charges",
    "Thương mại":                            "Commerce",
    "Tiền tệ - Ngân hàng":                   "Currency and banking",
    "Trách nhiệm hình sự":                   "Criminal liability",
    "Tư pháp – Hộ tịch":                     "Justice and civil status",
    "Văn hoá – Thể thao – Du lịch":          "Culture, sports and tourism",
    "Văn thư - Lưu trữ":                     "Records management and archives",
    "Vi phạm hành chính":                    "Administrative violations",
    "Xăng dầu":                              "Petroleum",
    "Xây dựng - Đô thị":                     "Construction and urban planning",
    "Xuất nhập cảnh":                        "Immigration",
    "Xuất nhập khẩu":                        "Import and export",
    "Y tế":                                  "Healthcare",
}


#: Status string VI→EN map. The source portal only ever emits these
#: four values for the "Tình trạng" line as of 2026-05; unknown values
#: are passed through verbatim with a warning so future additions are
#: never silently dropped.
STATUS_VI_TO_EN: dict[str, str] = {
    "Còn hiệu lực":          "Effective",
    "Hết hiệu lực":          "Expired",
    "Hết hiệu lực một phần": "Partially expired",
    "Chưa có hiệu lực":      "Not yet effective",
}


#: Updated-by passthrough exception: the well-known
#: ``Người dùng không đăng nhập`` placeholder (anonymous editor) is
#: the only ``cập_nhật_bởi`` value we translate; everything else is a
#: proper name we copy verbatim.
UPDATED_BY_VI_TO_EN: dict[str, str] = {
    "Người dùng không đăng nhập": "Unauthenticated user",
}


# -----------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------

def build_layout(cfg: Any) -> SiteLayout:
    """Ensure every output directory exists and return the layout.

    thuvienphapluat_tnpl's data root layout under ``<output_dir>/<host>/``
    is::

        html/index.html                    # /tnpl/home cache (taxonomy)
        html/items/<term_id>.html          # raw detail fragments
        translations/<term_id>.json        # per-row LLM translation cache
        jsonl/listings.jsonl               # one stub row per probed id
        jsonl/terms.jsonl                  # one row per term (raw VI)
        jsonl/terms_translated.jsonl       # bilingual deliverable
        jsonl/taxonomy.json                # bilingual LinhVuc + statuses
        jsonl/manifest.json                # detail-run summary
        jsonl/translation_manifest.json    # translate-run summary
        jsonl/analytics.json               # analyze.py output
        logs/                              # reserved for run logs

    Uses the shared ``"html"`` layout profile + two tnpl-specific
    extra dirs (``html/items/`` and ``translations/``).
    """
    layout = SiteLayout.from_cfg(cfg)
    return _build_layout_common(
        cfg,
        profile="html",
        extra_dirs=(
            layout.html_dir / "items",
            layout.site_root / "translations",
        ),
    )


def items_dir(layout: SiteLayout) -> Path:
    return layout.html_dir / "items"


def translations_dir(layout: SiteLayout) -> Path:
    return layout.site_root / "translations"


__all__ = [
    "DETAIL_JSONL_FIELDS",
    "LINH_VUC_VI_TO_EN",
    "LISTING_JSONL_FIELDS",
    "STATUS_VI_TO_EN",
    "TRANSLATED_JSONL_FIELDS",
    "UPDATED_BY_VI_TO_EN",
    "VI_TO_EN_COLUMN_MAP",
    "build_layout",
    "items_dir",
    "translations_dir",
]
