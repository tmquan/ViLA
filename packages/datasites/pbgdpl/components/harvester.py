"""Harvest the full ItemID set + listing-derived metadata.

The pbgdpl site has no JSON / OData / SOAP API for the Q&A content
(see README "Why no JSON API?" for the probe results). The only
public surface is the AJAX user control at
``/SMPT_Publishing_UC/HoiDapPL/frmDSCauHoi.aspx``.

This module walks that endpoint with a :class:`PoliteSession` and
captures:

1. **All ItemIDs** by paging ``?page=1..N`` over the global listing
   (``N`` is auto-detected from the pagination footer).
2. **Per-item LinhVuc assignment** by paging ``?lv=<id>&page=1..M``
   over each populated topic in the LinhVuc taxonomy. A question may
   belong to multiple topics; we accumulate ``lv_ids`` accordingly.
3. **Featured set** ("Câu hỏi được quan tâm") from the homepage at
   ``/Pages/hoi-dap-pl.aspx``.
4. **Topic taxonomy** (LinhVuc id -> Vietnamese name) from the same
   homepage's ``<select name="LinhVuc">``.

Outputs are written to disk in two passes so a long crawl can
resume cheaply:

* Raw HTML caches under ``html/index.html``, ``html/listings/page-NNNN.html``,
  and ``html/lv/<lv_id>.html``.
* A consolidated ``jsonl/listings.jsonl`` (one row per question, with
  page+position+featured+lv_ids attached) plus ``jsonl/taxonomy.json``.

The fetcher is single-threaded by design: with QPS=2 the global walk
finishes in ~5 minutes and the additional taxonomy walk in ~10-15
minutes more (~530 lv ids, with several heavily-populated busy ones).
Adding parallelism would burn the QPS budget without shortening
wall-clock noticeably because the endpoint serialises requests under
a per-IP semaphore.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from packages.common import PoliteSession, SiteLayout
from packages.common.http import session_from_scraper_cfg
from packages.datasites.pbgdpl._shared import LISTING_JSONL_FIELDS, lv_dir
from packages.datasites.pbgdpl.components.parser import (
    ListingEntry,
    parse_featured_ids,
    parse_listing_fragment,
    parse_taxonomy,
)

logger = logging.getLogger(__name__)


DEFAULT_INDEX_URL = "https://pbgdpl.gov.vn/Pages/hoi-dap-pl.aspx"
DEFAULT_LISTING_URL = (
    "https://pbgdpl.gov.vn/SMPT_Publishing_UC/HoiDapPL/frmDSCauHoi.aspx"
)


@dataclass
class HarvestState:
    """Accumulated listing metadata, keyed by ItemID."""

    items: dict[int, dict[str, Any]] = field(default_factory=dict)
    last_page: int = 0
    page_count: int = 0
    lv_pages_fetched: int = 0
    featured_ids: list[int] = field(default_factory=list)
    taxonomy: dict[int, str] = field(default_factory=dict)


class PbgdplHarvester:
    """Top-level orchestrator for the ItemID harvest.

    Stores only pickle-safe state on the driver (cfg + URLs). The
    :class:`PoliteSession` is built lazily inside :meth:`run` so the
    object is safe to instantiate before Ray init (kept for symmetry
    with the anle / congbobanan crawlers, which run under Ray).
    """

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._listing_url: str = str(
            cfg.scraper.get("listing_url", DEFAULT_LISTING_URL)
        )
        self._index_url: str = str(cfg.scraper.get("index_url", DEFAULT_INDEX_URL))
        self._max_pages_cfg = cfg.scraper.get("max_pages", None)
        self._walk_lv: bool = bool(cfg.scraper.get("walk_lv", True))
        self._lv_max_pages: int = int(cfg.scraper.get("lv_max_pages", 100))
        self._cache_listings: bool = bool(
            cfg.scraper.get("cache_listings", True),
        )
        self.session: PoliteSession | None = None

    # ---- public entrypoint ----------------------------------------------

    def run(self) -> HarvestState:
        """Walk every required listing surface; return accumulated state."""
        if self.session is None:
            self.session = session_from_scraper_cfg(self.cfg)

        state = HarvestState()

        index_html = self._fetch_index()
        state.taxonomy = parse_taxonomy(index_html)
        state.featured_ids = parse_featured_ids(index_html)
        logger.info(
            "index ok: %d LinhVuc topics, %d featured ItemIDs",
            len(state.taxonomy),
            len(state.featured_ids),
        )

        self._walk_global_pages(state)

        if self._walk_lv:
            self._walk_lv_pages(state)

        for fid in state.featured_ids:
            row = state.items.setdefault(
                fid,
                _empty_row(fid),
            )
            row["is_featured"] = True

        return state

    # ---- step 1: homepage -----------------------------------------------

    def _fetch_index(self) -> str:
        """GET the homepage, write to ``html/index.html``, return the body."""
        assert self.session is not None
        cache = self.layout.html_dir / "index.html"
        if cache.exists() and cache.stat().st_size > 0:
            return cache.read_text(encoding="utf-8")
        resp = self.session.get(self._index_url)
        if resp.status_code != 200:
            logger.warning(
                "index fetch failed: status=%d url=%s",
                resp.status_code,
                self._index_url,
            )
            return ""
        cache.write_text(resp.text, encoding="utf-8")
        return resp.text

    # ---- step 2: global page walk ---------------------------------------

    def _walk_global_pages(self, state: HarvestState) -> None:
        """Page through ``?page=1..last_page``."""
        first_html = self._fetch_listing(page=1)
        first_entries, last_page = parse_listing_fragment(first_html)
        if self._max_pages_cfg:
            last_page = min(int(self._max_pages_cfg), last_page)
        state.last_page = last_page
        logger.info("global listing: pages 1..%d", last_page)

        self._merge_entries(state, page=1, entries=first_entries, lv_id=None)
        state.page_count = 1
        for page in range(2, last_page + 1):
            html = self._fetch_listing(page=page)
            entries, _ = parse_listing_fragment(html)
            if not entries:
                logger.warning("page %d empty; continuing", page)
                continue
            self._merge_entries(state, page=page, entries=entries, lv_id=None)
            state.page_count += 1
            if page % 25 == 0:
                logger.info(
                    "global listing progress: page=%d/%d items=%d",
                    page, last_page, len(state.items),
                )

    # ---- step 3: per-LinhVuc walk ---------------------------------------

    def _walk_lv_pages(self, state: HarvestState) -> None:
        """Walk every populated LinhVuc topic to attach lv_ids per item."""
        names = state.taxonomy
        if not names:
            return
        ordered = sorted(names.keys())
        logger.info("lv walk: %d topics to probe", len(ordered))
        for i, lv_id in enumerate(ordered, 1):
            self._walk_one_lv(state, lv_id, names[lv_id])
            if i % 25 == 0:
                logger.info(
                    "lv walk progress: %d/%d topics; lv pages fetched=%d",
                    i, len(ordered), state.lv_pages_fetched,
                )

    def _walk_one_lv(self, state: HarvestState, lv_id: int, lv_name: str) -> None:
        """Walk every page within one LinhVuc; attach lv_id to each item."""
        first = self._fetch_listing(page=1, lv_id=lv_id)
        state.lv_pages_fetched += 1
        entries, last_page = parse_listing_fragment(first)
        if not entries:
            return
        last_page = min(last_page, self._lv_max_pages)
        for entry in entries:
            self._attach_lv(state, entry.item_id, lv_id, lv_name)
        for page in range(2, last_page + 1):
            html = self._fetch_listing(page=page, lv_id=lv_id)
            state.lv_pages_fetched += 1
            page_entries, _ = parse_listing_fragment(html)
            if not page_entries:
                break
            for entry in page_entries:
                self._attach_lv(state, entry.item_id, lv_id, lv_name)

    @staticmethod
    def _attach_lv(
        state: HarvestState, item_id: int, lv_id: int, lv_name: str,
    ) -> None:
        row = state.items.setdefault(item_id, _empty_row(item_id))
        if lv_id not in row["lv_ids"]:
            row["lv_ids"].append(lv_id)
            row["lv_names"].append(lv_name)

    # ---- HTTP -----------------------------------------------------------

    def _fetch_listing(self, *, page: int, lv_id: int | None = None) -> str:
        """GET one listing fragment; cache HTML on disk; return body.

        Cache layout:

        * ``html/listings/page-NNNN.html`` for ``lv_id=None``.
        * ``html/lv/<lv_id>.html``         for ``lv_id`` page 1.
        * ``html/lv/<lv_id>-pNN.html``     for ``lv_id`` deep pages.
        """
        assert self.session is not None
        url = self._page_url(page=page, lv_id=lv_id)
        cache = self._listing_cache_path(page=page, lv_id=lv_id)
        if (
            self._cache_listings
            and cache.exists()
            and cache.stat().st_size > 0
        ):
            return cache.read_text(encoding="utf-8")

        resp = self.session.get(url)
        if resp.status_code != 200:
            logger.warning(
                "listing fetch failed: status=%d url=%s",
                resp.status_code, url,
            )
            return ""
        if self._cache_listings:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(resp.text, encoding="utf-8")
        return resp.text

    def _page_url(self, *, page: int, lv_id: int | None) -> str:
        params: dict[str, str] = {"page": str(page)}
        if lv_id is not None:
            params["lv"] = str(lv_id)
        return f"{self._listing_url}?{urlencode(params)}"

    def _listing_cache_path(self, *, page: int, lv_id: int | None) -> Path:
        if lv_id is None:
            return self.layout.html_dir / "listings" / f"page-{page:04d}.html"
        if page == 1:
            return lv_dir(self.layout) / f"{lv_id}.html"
        return lv_dir(self.layout) / f"{lv_id}-p{page:02d}.html"

    # ---- accumulation ---------------------------------------------------

    @staticmethod
    def _merge_entries(
        state: HarvestState,
        *,
        page: int,
        entries: list[ListingEntry],
        lv_id: int | None,
    ) -> None:
        for pos, entry in enumerate(entries, start=1):
            row = state.items.setdefault(entry.item_id, _empty_row(entry.item_id))
            if lv_id is None:
                row["listing_page"] = page
                row["listing_position"] = pos
                row["title_listing"] = entry.title or row["title_listing"]
                row["question_summary_listing"] = (
                    entry.question_summary_text
                    or row["question_summary_listing"]
                )
                row["sender_name_listing"] = (
                    entry.sender_name or row["sender_name_listing"]
                )

    # ---- writers --------------------------------------------------------

    def write_outputs(self, state: HarvestState) -> tuple[Path, Path]:
        """Write taxonomy.json + listings.jsonl. Returns the two paths."""
        tax_path = self.layout.jsonl_dir / "taxonomy.json"
        tax_payload = {
            "host": self.layout.host,
            "captured_at": _utc_now_iso(),
            "source_url": self._index_url,
            "linh_vuc": [
                {"id": k, "name": state.taxonomy[k]}
                for k in sorted(state.taxonomy.keys())
            ],
            "featured_ids": list(state.featured_ids),
        }
        tax_path.write_text(
            json.dumps(tax_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        lst_path = self.layout.jsonl_dir / "listings.jsonl"
        harvested_at = _utc_now_iso()
        with lst_path.open("w", encoding="utf-8") as f:
            for item_id in sorted(state.items.keys()):
                row = state.items[item_id]
                row["harvested_at"] = harvested_at
                f.write(
                    json.dumps(
                        {k: row.get(k) for k in LISTING_JSONL_FIELDS},
                        ensure_ascii=False,
                    )
                )
                f.write("\n")
        logger.info(
            "harvest written: %s (%d rows), %s (%d topics)",
            lst_path, len(state.items), tax_path, len(state.taxonomy),
        )
        return lst_path, tax_path


# ---- helpers -------------------------------------------------------------


def _empty_row(item_id: int) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "listing_page": None,
        "listing_position": None,
        "title_listing": "",
        "question_summary_listing": "",
        "sender_name_listing": None,
        "is_featured": False,
        "lv_ids": [],
        "lv_names": [],
        "harvested_at": None,
    }


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_INDEX_URL",
    "DEFAULT_LISTING_URL",
    "HarvestState",
    "PbgdplHarvester",
]
