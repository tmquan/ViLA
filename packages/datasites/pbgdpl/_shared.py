"""Shared helpers for the pbgdpl crawler.

Holds the output-path layout builder and the field lists used by the
listing / detail writers so the JSONL / JSON schemas stay consistent
across the harvester, the LinhVuc taxonomy walker, and the detail
parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.common import SiteLayout, build_layout as _build_layout_common


#: Detail JSONL columns emitted by :func:`packages.datasites.pbgdpl.scraper.run_detail`.
#:
#: Every column is documented in :doc:`README` under "Output schema".
#: Order matters: it is the canonical column order for downstream
#: consumers that read this file with ``pyarrow.json.read_json`` or
#: ``pandas.read_json(lines=True)``.
DETAIL_JSONL_FIELDS: list[str] = [
    "item_id",
    "source",
    "source_url",
    "scraped_at",
    "scrape_run_id",
    "listing_page",
    "listing_position",
    "is_featured",
    "title_listing",
    "question_summary_listing",
    "lv_ids",
    "lv_names",
    "title",
    "question_html",
    "question_text",
    "answer_html",
    "answer_text",
    "date_sent_raw",
    "date_sent",
    "sender_name",
    "disclaimer",
    "question_char_len",
    "answer_char_len",
    "question_word_count",
    "answer_word_count",
    "answer_text_hash",
    "html_path",
    "fetch_status",
    "fetch_error",
]


#: Listing JSONL columns emitted by :func:`packages.datasites.pbgdpl.scraper.run_harvest`.
LISTING_JSONL_FIELDS: list[str] = [
    "item_id",
    "listing_page",
    "listing_position",
    "title_listing",
    "question_summary_listing",
    "sender_name_listing",
    "is_featured",
    "lv_ids",
    "lv_names",
    "harvested_at",
]


def build_layout(cfg: Any) -> SiteLayout:
    """Ensure every output directory exists and return the :class:`SiteLayout`.

    pbgdpl's data root layout under ``<output_dir>/<host>/`` is::

        html/listings/page-NNNN.html       # raw listing fragments
        html/items/<item_id>.html          # raw detail fragments
        html/lv/<lv_id>.html               # raw per-topic listings (1st page)
        html/index.html                    # the /Pages/hoi-dap-pl.aspx homepage
        jsonl/listings.jsonl               # one row per harvested listing entry
        jsonl/qa.jsonl                     # one row per detail Q&A
        jsonl/taxonomy.json                # LinhVuc id -> name (~535 entries)
        jsonl/manifest.json                # last-run summary
        logs/run-<ts>.jsonl                # per-request operational log

    Uses the shared ``"html"`` layout profile + three pbgdpl-specific
    HTML cache subdirs (``listings/`` / ``items/`` / ``lv/``).
    """
    layout = SiteLayout.from_cfg(cfg)
    return _build_layout_common(
        cfg,
        profile="html",
        extra_dirs=(
            layout.html_dir / "listings",
            layout.html_dir / "items",
            layout.html_dir / "lv",
        ),
    )


def listings_dir(layout: SiteLayout) -> Path:
    return layout.html_dir / "listings"


def items_dir(layout: SiteLayout) -> Path:
    return layout.html_dir / "items"


def lv_dir(layout: SiteLayout) -> Path:
    return layout.html_dir / "lv"


__all__ = [
    "DETAIL_JSONL_FIELDS",
    "LISTING_JSONL_FIELDS",
    "build_layout",
    "items_dir",
    "listings_dir",
    "lv_dir",
]
