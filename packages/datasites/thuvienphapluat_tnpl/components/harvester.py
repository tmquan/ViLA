"""Discover the term-ID probe range + bilingual taxonomy for tnpl.

The thuvienphapluat ``/tnpl/`` portal exposes no JSON / OData / SOAP
listing API; only a fuzzy ``/tnpl/search`` endpoint that returns ~4
near-matches per query. There is also no sitemap shard for ``/tnpl/``
URLs (we probed ``sitemap.xml``, ``resitemap1..575.xml``,
``sitemap_tnpl.xml`` -- none of them index the term pages).

Term ids are integers though, and the homepage at ``/tnpl/home`` shows
the most-recent ~20 ids together with the total count
(``Tìm thấy <b>16247</b> thuật ngữ``) and the 47-entry LinhVuc
taxonomy. So the only viable harvest strategy is:

1. fetch the homepage once (caches ``html/index.html``);
2. parse the LinhVuc taxonomy, total count, and the bootstrap ids;
3. compute a probe range ``[id_start, max(homepage_ids) + id_buffer]``
   capped by ``cfg.scraper.max_id``;
4. write ``jsonl/taxonomy.json`` with separate Vietnamese and English
   arrays -- ``lĩnh_vực: [{id, ten}]`` and ``area: [{id, name}]`` --
   plus the four ``tình_trạng`` values, and ``jsonl/listings.jsonl``
   (one stub row per probe id, marking which ids came from the
   homepage bootstrap).

The detail downloader then walks ``listings.jsonl`` and fetches every
id; missing/retracted ids return HTTP 200 with a soft-404 body that
the parser tags as ``fetch_status="not_found"``.

The harvester is single-threaded and one-shot; almost all wall-clock
time is in the detail stage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.common import PoliteSession, SiteLayout
from packages.common.http import session_from_scraper_cfg
from packages.datasites.thuvienphapluat_tnpl._shared import (
    LINH_VUC_VI_TO_EN,
    LISTING_JSONL_FIELDS,
    STATUS_VI_TO_EN,
)
from packages.datasites.thuvienphapluat_tnpl.components.parser import (
    parse_homepage_ids,
    parse_taxonomy,
    parse_total_count,
)

logger = logging.getLogger(__name__)


DEFAULT_INDEX_URL = "https://thuvienphapluat.vn/tnpl/home"
DEFAULT_DETAIL_URL_TEMPLATE = "https://thuvienphapluat.vn/tnpl/{id}/x?tab=0"


@dataclass
class HarvestState:
    """Accumulated harvest output before write."""

    taxonomy: dict[int, str] = field(default_factory=dict)
    total_count: int | None = None
    homepage_ids: list[int] = field(default_factory=list)
    max_id_seen: int = 0
    probe_start: int = 1
    probe_end: int = 0
    fetched_index: bool = False


class TnplHarvester:
    """Fetch the homepage, derive the probe range, write listings + taxonomy."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._index_url: str = str(
            cfg.scraper.get("index_url", DEFAULT_INDEX_URL)
        )
        self._id_start: int = int(cfg.scraper.get("id_start", 1))
        # Hard upper cap; null/None means "auto from homepage + id_buffer".
        self._cfg_max_id = cfg.scraper.get("max_id", None)
        self._id_buffer: int = int(cfg.scraper.get("id_buffer", 200))
        self._cache_index: bool = bool(cfg.scraper.get("cache_index", True))
        self.session: PoliteSession | None = None

    # ---- public entrypoint ----------------------------------------------

    def run(self) -> HarvestState:
        if self.session is None:
            self.session = session_from_scraper_cfg(self.cfg)

        state = HarvestState()

        index_html = self._fetch_index()
        state.fetched_index = bool(index_html)
        state.taxonomy = parse_taxonomy(index_html)
        state.total_count = parse_total_count(index_html)
        state.homepage_ids = parse_homepage_ids(index_html)
        state.max_id_seen = max(state.homepage_ids) if state.homepage_ids else 0

        # Compute probe range.
        if self._cfg_max_id is not None:
            probe_end = int(self._cfg_max_id)
        elif state.max_id_seen:
            probe_end = state.max_id_seen + self._id_buffer
        else:
            # Fallback: use total_count + buffer when the homepage gave
            # us nothing else (e.g. the ``<a class='tnpl'>`` set was
            # empty due to a rendering quirk).
            base = state.total_count or 0
            probe_end = base + self._id_buffer if base else 0

        if probe_end < self._id_start:
            raise RuntimeError(
                f"harvest produced an empty probe range: "
                f"id_start={self._id_start} probe_end={probe_end}; "
                f"check homepage parse + cfg.scraper.max_id",
            )

        state.probe_start = self._id_start
        state.probe_end = probe_end

        logger.info(
            "harvest: taxonomy=%d, total_count=%s, homepage_ids=%d, "
            "probe=[%d, %d] (size=%d)",
            len(state.taxonomy), state.total_count,
            len(state.homepage_ids),
            state.probe_start, state.probe_end,
            state.probe_end - state.probe_start + 1,
        )
        return state

    # ---- step 1: homepage -----------------------------------------------

    def _fetch_index(self) -> str:
        """GET the homepage, cache to ``html/index.html``, return the body."""
        assert self.session is not None
        cache = self.layout.html_dir / "index.html"
        if (
            self._cache_index
            and cache.exists()
            and cache.stat().st_size > 0
        ):
            return cache.read_text(encoding="utf-8")
        resp = self.session.get(self._index_url)
        if resp.status_code != 200:
            logger.warning(
                "index fetch failed: status=%d url=%s",
                resp.status_code, self._index_url,
            )
            return ""
        cache.write_text(resp.text, encoding="utf-8")
        return resp.text

    # ---- writers --------------------------------------------------------

    def write_outputs(self, state: HarvestState) -> tuple[Path, Path]:
        """Write bilingual taxonomy.json + listings.jsonl. Returns paths.

        The listings file is a stub-per-probe-id document the
        downloader walks. Carrying ``is_bootstrap`` lets the detail
        stage prioritise the homepage's recently-updated ids first if
        a resumed run wants to.
        """
        tax_path = self.layout.jsonl_dir / "taxonomy.json"
        captured_at = _utc_now_iso()
        # Emit separate VI and EN taxonomy arrays. This keeps the
        # source taxonomy faithful (`lĩnh_vực[].ten`) while giving
        # English consumers a first-class `area[].name` array keyed by
        # the same id.
        lv_entries: list[dict[str, Any]] = []
        area_entries: list[dict[str, Any]] = []
        unknown: list[str] = []
        for lv_id in sorted(state.taxonomy.keys()):
            name_vi = state.taxonomy[lv_id]
            name_en = LINH_VUC_VI_TO_EN.get(name_vi)
            if name_en is None:
                unknown.append(name_vi)
                name_en = name_vi  # fall through verbatim
            lv_entries.append({"id": lv_id, "ten": name_vi})
            area_entries.append({"id": lv_id, "name": name_en})
        if unknown:
            logger.warning(
                "%d LinhVuc names not in LINH_VUC_VI_TO_EN; "
                "translator will pass them through verbatim: %s",
                len(unknown), unknown,
            )

        tax_payload: dict[str, Any] = {
            "host": self.layout.host,
            "captured_at": captured_at,
            "source_url": self._index_url,
            "total_count": state.total_count,
            "max_id_seen": state.max_id_seen,
            "probe_range": {
                "start": state.probe_start,
                "end": state.probe_end,
            },
            "bootstrap_ids": list(state.homepage_ids),
            "lĩnh_vực": lv_entries,
            "area": area_entries,
            "tình_trạng": [
                {"vi": vi, "en": en} for vi, en in STATUS_VI_TO_EN.items()
            ],
        }
        tax_path.write_text(
            json.dumps(tax_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        lst_path = self.layout.jsonl_dir / "listings.jsonl"
        bootstrap_set = set(state.homepage_ids)
        with lst_path.open("w", encoding="utf-8") as f:
            for term_id in range(state.probe_start, state.probe_end + 1):
                row = {
                    "term_id": term_id,
                    "is_bootstrap": term_id in bootstrap_set,
                    "harvested_at": captured_at,
                }
                f.write(
                    json.dumps(
                        {k: row.get(k) for k in LISTING_JSONL_FIELDS},
                        ensure_ascii=False,
                    )
                )
                f.write("\n")
        logger.info(
            "harvest written: %s (%d rows), %s (%d topics)",
            lst_path, state.probe_end - state.probe_start + 1,
            tax_path, len(state.taxonomy),
        )
        return lst_path, tax_path


# ---- helpers -------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_DETAIL_URL_TEMPLATE",
    "DEFAULT_INDEX_URL",
    "HarvestState",
    "TnplHarvester",
]
