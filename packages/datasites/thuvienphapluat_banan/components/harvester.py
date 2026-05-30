"""ID-range enumeration harvester for thuvienphapluat_banan.

The ``/banan/tim-ban-an`` paginated search endpoint is fronted by a
Cloudflare Turnstile JS challenge (HTTP 403 for any non-browser
client), which makes the natural "walk the listing" pattern from
``pbgdpl`` / ``vbpl`` infeasible. The **slugless detail shortcut**
``/banan/ban-an/x-<id>``, however, is wide open — it 302's to the
canonical ``/banan/ban-an/<slug>-<id>`` URL with full HTML and no
Turnstile in front. This harvester therefore enumerates the integer
``ban_an_id`` space and writes one synthetic listing row per id, with
``url`` set to the slugless shortcut so the downstream detail stage
gets the canonical URL via the redirect.

Output:

* ``jsonl/listings.jsonl`` -- one row per ``ban_an_id`` in
  ``[id_start, max_id]``. Each row carries only the bookkeeping the
  detail stage needs (``ban_an_id``, ``url``, ``harvested_at``);
  title / court / doc_number / issue_date are left null and filled
  by the detail stage from the sidebar HTML. Sparse ids that 302 to
  ``pagenotfound.htm`` get tagged ``fetch_status="not_found"`` by
  the detail stage — no special handling here.
* ``jsonl/taxonomy.json`` -- best-effort snapshot of the ``/banan/``
  landing-page facet menu when reachable (200); written as a stub
  ``{"host": ..., "captured_at": ..., "note": "..."}`` when blocked.

The harvester is intentionally a one-shot enumeration: no HTTP calls
for the id list itself (the listing endpoint is dead and there is no
sitemap covering ``/banan/``). The ``id_start`` / ``max_id`` /
``id_buffer`` knobs match the sibling :mod:`thuvienphapluat_tnpl`
harvester so operators see the same surface.

The previous paginated walker shipped in 0c8b… (git history) is
retired; this module replaces it wholesale. The ``BananHarvester``
class name + entrypoint are preserved so :mod:`.scraper` doesn't
need to know which harvest backend is active.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from packages.common import PoliteSession, SiteLayout
from packages.common.http import session_from_scraper_cfg
from packages.datasites.thuvienphapluat_banan._shared import (
    LISTING_JSONL_FIELDS,
    listings_dir,
)

logger = logging.getLogger(__name__)


#: Slugless detail shortcut used for downstream detail GETs. The
#: portal 302's it to the canonical ``/banan/ban-an/<slug>-<id>``
#: URL; the downloader records the resolved URL as ``source_url``.
DEFAULT_DETAIL_URL_TEMPLATE = (
    "https://thuvienphapluat.vn/banan/ban-an/x-{id}"
)

#: Landing page used as the taxonomy probe (the only ``/banan/``
#: surface known to return 200 without Turnstile). When that 200 we
#: scrape the page's ``<select>`` dropdowns for the facet menu; when
#: it 403s we write a stub taxonomy and continue.
DEFAULT_TAXONOMY_URL = "https://thuvienphapluat.vn/banan/"


@dataclass
class HarvestState:
    """Accumulated harvest output before write."""

    listings: dict[int, dict[str, Any]] = field(default_factory=dict)
    taxonomy: dict[str, Any] = field(default_factory=dict)
    fetched_at: str = ""
    id_start: int = 0
    max_id: int = 0


class BananHarvester:
    """Integer-ID enumerator for the ``/banan/`` corpus."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._id_start: int = max(1, int(cfg.scraper.get("id_start", 1)))
        # ``max_id`` null/0/negative ⇒ derive from the landing page's
        # largest visible id + ``id_buffer``.
        self._max_id_cfg = cfg.scraper.get("max_id", None)
        self._id_buffer: int = max(0, int(cfg.scraper.get("id_buffer", 5000)))
        self._cache_taxonomy: bool = bool(
            cfg.scraper.get("cache_listings", True),
        )
        self._taxonomy_url: str = str(
            cfg.scraper.get("taxonomy_url", DEFAULT_TAXONOMY_URL),
        )
        self._detail_url_template: str = str(
            cfg.scraper.get("detail_url_template", DEFAULT_DETAIL_URL_TEMPLATE),
        )
        self._limit = cfg.get("limit", None)
        # WAF 403 cool-down knobs (only used for the optional taxonomy
        # probe; the slugless detail URL is open in our tests so the
        # downloader rarely hits 403, but the policy is shared).
        self._http_403_initial_delay_s: float = float(
            cfg.scraper.get("http_403_initial_delay_s", 60.0),
        )
        self._http_403_max_delay_s: float = float(
            cfg.scraper.get("http_403_max_delay_s", 600.0),
        )
        self._http_403_max_retries: int = int(
            cfg.scraper.get("http_403_max_retries", 5),
        )
        self._session: PoliteSession | None = None

    # ------------------------------------------------------ entrypoint

    def run(self) -> HarvestState:
        """Enumerate ids + best-effort taxonomy. Returns the state."""
        state = HarvestState(fetched_at=_utc_now_iso())

        # Step 1: taxonomy + landing-page id sniff (best effort).
        landing_html = self._fetch_landing()
        state.taxonomy = self._parse_taxonomy(landing_html)
        sniffed_max = _max_id_from_landing(landing_html)

        # Step 2: lock the id range.
        if self._max_id_cfg and int(self._max_id_cfg) > 0:
            state.max_id = int(self._max_id_cfg)
            logger.info(
                "harvest: max_id pinned from cfg = %d", state.max_id,
            )
        elif sniffed_max:
            state.max_id = sniffed_max + self._id_buffer
            logger.info(
                "harvest: max_id sniffed from landing = %d (+ buffer %d) = %d",
                sniffed_max, self._id_buffer, state.max_id,
            )
        else:
            # Hard floor (the highest id we saw on 2026-05 recon) +
            # generous buffer so a stale floor doesn't drop new judgments.
            fallback_floor = 381_970
            state.max_id = fallback_floor + max(self._id_buffer, 10_000)
            logger.warning(
                "harvest: landing-page id sniff failed; using fallback "
                "max_id = %d (= %d hard floor + %d buffer)",
                state.max_id, fallback_floor, max(self._id_buffer, 10_000),
            )
        state.id_start = self._id_start

        # Step 3: enumerate. Cap by --limit for smoke runs.
        candidate_ids = list(range(self._id_start, state.max_id + 1))
        if self._limit is not None and int(self._limit) > 0:
            candidate_ids = candidate_ids[: int(self._limit)]
        logger.info(
            "harvest: emitting %d candidate ids [%d .. %d]",
            len(candidate_ids), self._id_start, candidate_ids[-1] if candidate_ids else state.max_id,
        )
        fetched_at = state.fetched_at
        listings: dict[int, dict[str, Any]] = {}
        for ban_an_id in candidate_ids:
            url = self._detail_url_template.format(id=ban_an_id)
            listings[ban_an_id] = {
                "ban_an_id":    ban_an_id,
                "slug":         None,
                "url":          url,
                "title":        None,
                "summary":      None,
                "doc_number":   None,
                "court":        None,
                "issue_date":   None,
                "case_kind":    None,
                "procedure":    None,
                "page_number":  None,
                "harvested_at": fetched_at,
            }
        state.listings = listings
        logger.info(
            "harvest: enumeration complete; %d candidate ids in [%d .. %d]",
            len(state.listings), self._id_start, state.max_id,
        )
        return state

    # ------------------------------------------------------ landing + taxonomy

    def _fetch_landing(self) -> str:
        """GET ``/banan/`` (cached). Returns "" if blocked."""
        cache = self.layout.html_dir / "taxonomy.html"
        if (
            self._cache_taxonomy
            and cache.exists()
            and cache.stat().st_size > 0
        ):
            return cache.read_text(encoding="utf-8")
        if self._session is None:
            self._session = session_from_scraper_cfg(self.cfg)
        try:
            resp = self._get_with_403_retry(self._taxonomy_url)
        except Exception as exc:
            logger.warning("harvest: landing fetch crashed: %s", exc)
            return ""
        if resp.status_code != 200:
            logger.warning(
                "harvest: landing fetch returned %d (taxonomy will be stub)",
                resp.status_code,
            )
            return ""
        text = resp.text or ""
        if self._cache_taxonomy:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        return text

    def _parse_taxonomy(self, html: str) -> dict[str, Any]:
        if not html:
            return {"note": "landing unavailable; taxonomy not captured"}
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return {"note": "bs4 not installed; taxonomy not parsed"}

        soup = BeautifulSoup(html, "html.parser")
        out: dict[str, Any] = {}
        for sel in soup.find_all("select"):
            name = sel.get("name") or sel.get("id")
            if not name:
                continue
            options: list[dict[str, Any]] = []
            for opt in sel.find_all("option"):
                value = opt.get("value")
                text = (opt.get_text(strip=True) or "").strip()
                if not text and value is None:
                    continue
                options.append({"value": value, "label": text})
            if options:
                out[str(name)] = options
        return out

    def _get_with_403_retry(self, url: str) -> requests.Response:
        assert self._session is not None
        delay = self._http_403_initial_delay_s
        attempts = 0
        while True:
            resp = self._session.get(url)
            if resp.status_code != 403:
                return resp
            if attempts >= self._http_403_max_retries:
                return resp
            attempts += 1
            logger.warning(
                "harvest: 403 on %s; WAF cool-down %.0fs (attempt %d/%d)",
                url, delay, attempts, self._http_403_max_retries,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, self._http_403_max_delay_s)

    # ------------------------------------------------------ writers

    def write_outputs(self, state: HarvestState) -> tuple[Path, Path]:
        """Write listings.jsonl + taxonomy.json. Returns ``(lst, tax)``."""
        lst_path = self.layout.jsonl_dir / "listings.jsonl"
        with lst_path.open("w", encoding="utf-8") as f:
            for ban_an_id in sorted(state.listings):
                row = state.listings[ban_an_id]
                f.write(
                    json.dumps(
                        {k: row.get(k) for k in LISTING_JSONL_FIELDS},
                        ensure_ascii=False,
                    ),
                )
                f.write("\n")

        tax_path = self.layout.jsonl_dir / "taxonomy.json"
        tax_payload: dict[str, Any] = {
            "host": self.layout.host,
            "captured_at": state.fetched_at,
            "source_url": self._taxonomy_url,
            "id_start": state.id_start,
            "max_id":   state.max_id,
            "candidates": len(state.listings),
            **state.taxonomy,
        }
        tax_path.write_text(
            json.dumps(tax_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "harvest written: %s (%d rows), %s",
            lst_path, len(state.listings), tax_path,
        )
        return lst_path, tax_path


def listings_cache_dir(layout: SiteLayout) -> Path:
    """Deprecated alias kept for back-compat; the id-range harvester
    does not cache per-page listing HTML.
    """
    return listings_dir(layout)


# ---- helpers -------------------------------------------------------------


def _max_id_from_landing(html: str) -> int | None:
    """Sniff the largest ``ban_an_id`` referenced on the landing page."""
    if not html:
        return None
    import re
    ids = [
        int(m)
        for m in re.findall(
            r"/banan/ban-an/[^\"'>]*?-(\d+)(?:[?#\"'>])", html,
        )
    ]
    return max(ids) if ids else None


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_DETAIL_URL_TEMPLATE",
    "DEFAULT_TAXONOMY_URL",
    "BananHarvester",
    "HarvestState",
]
