"""Walk the ``/banan/tim-ban-an`` paginated search listing.

The thuvienphapluat ``/banan/`` portal exposes a search endpoint at
``/banan/tim-ban-an?type_q=0&sortType=1&page=N`` that paginates the
full ~319 K-judgment corpus 20 rows at a time. We walk it page-by-page
under a polite QPS envelope (the Cloudflare WAF in front of the portal
hands out flat 403s when bucket overflows; the shared 403 cool-down in
:meth:`PoliteSession.get` keeps the harvester alive through those
windows).

Output:

* ``html/listings/page-<N>.html`` -- per-page cache so a partial run
  resumes by skipping pages already on disk.
* ``jsonl/listings.jsonl``         -- one row per discovered judgment
  card (the union over every page; rows are de-duped by ``ban_an_id``).
* ``jsonl/taxonomy.json``          -- the closed-set faceting menu
  surfaced on the search form: courts (``AgentId``), provinces
  (``CityId``), trial levels (``AnLeType``), statuses (``Status``),
  sort modes (``sortType``), languages (``LanguageCode``). Captured
  verbatim from the first page's ``<select>`` dropdowns so the
  visualizer + dataset card can use the same vocabulary the source
  portal does.

The walk terminates when one of the following triggers fires:

1. ``cfg.scraper.max_pages`` is set and reached (soft cap for smoke runs).
2. A page returns < ``min_card_threshold`` listing cards (the portal's
   pagination overshoot — every page past the true end renders an
   empty results table).
3. The same ``ban_an_id`` set is observed two pages in a row (defensive
   shield against the WAF silently serving page 1's HTML for an
   out-of-range ``page=N``).
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
from packages.datasites.thuvienphapluat_banan.components.parser import (
    ListingEntry,
    parse_listing_page,
)

logger = logging.getLogger(__name__)


DEFAULT_LISTING_URL = (
    "https://thuvienphapluat.vn/banan/tim-ban-an"
    "?type_q=0&sortType=1&page={page}"
)
DEFAULT_TAXONOMY_URL = (
    "https://thuvienphapluat.vn/banan/tim-ban-an?LanguageCode=vi"
)
DEFAULT_PAGE_SIZE = 20         # the portal renders 20 cards per page
DEFAULT_MIN_CARDS = 1          # tolerate single-card final pages


@dataclass
class HarvestState:
    """Accumulated harvest output before write."""

    listings: dict[int, ListingEntry] = field(default_factory=dict)
    taxonomy: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pages_fetched: int = 0
    last_page: int = 0
    fetched_at: str = ""


class BananHarvester:
    """Paginated listing walker for ``/banan/tim-ban-an``."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._listing_url: str = str(
            cfg.scraper.get("listing_url_template", DEFAULT_LISTING_URL)
        )
        self._taxonomy_url: str = str(
            cfg.scraper.get("taxonomy_url", DEFAULT_TAXONOMY_URL)
        )
        self._start_page: int = int(cfg.scraper.get("start_page", 1))
        max_pages_cfg = cfg.scraper.get("max_pages", None)
        # ``None`` / ``0`` / negative ⇒ walk until natural end-of-pages.
        self._max_pages: int | None = (
            int(max_pages_cfg) if max_pages_cfg and int(max_pages_cfg) > 0 else None
        )
        self._cache_listings: bool = bool(
            cfg.scraper.get("cache_listings", True),
        )
        self._min_cards: int = max(
            1, int(cfg.scraper.get("min_card_threshold", DEFAULT_MIN_CARDS)),
        )
        # 403 cool-down knobs (mirrors thuvienphapluat_tnpl: WAF returns
        # flat 403 from Cloudflare when QPS bucket overflows).
        self._http_403_initial_delay_s: float = float(
            cfg.scraper.get("http_403_initial_delay_s", 60.0),
        )
        self._http_403_max_delay_s: float = float(
            cfg.scraper.get("http_403_max_delay_s", 600.0),
        )
        self._http_403_max_retries: int = int(
            cfg.scraper.get("http_403_max_retries", 5),
        )
        self._limit = cfg.get("limit", None)
        self._session: PoliteSession | None = None

    # ------------------------------------------------------ entrypoint

    def run(self) -> HarvestState:
        """Walk every page and return the accumulated state."""
        if self._session is None:
            self._session = session_from_scraper_cfg(self.cfg)

        state = HarvestState(fetched_at=_utc_now_iso())
        state.taxonomy = self._fetch_taxonomy()

        prior_ids: set[int] | None = None
        page = self._start_page
        while True:
            if self._max_pages is not None and (page - self._start_page) >= self._max_pages:
                logger.info(
                    "harvest: max_pages=%d reached at page=%d (stop)",
                    self._max_pages, page,
                )
                break
            url = self._listing_url.format(page=page)
            html = self._fetch_page(url, page=page)
            if not html:
                logger.warning("harvest: empty body at page=%d url=%s", page, url)
                break

            entries = parse_listing_page(html, page_url=url)
            if len(entries) < self._min_cards:
                logger.info(
                    "harvest: page=%d returned %d cards (< %d threshold); "
                    "treating as end-of-pages",
                    page, len(entries), self._min_cards,
                )
                break
            ids_on_page = {e.ban_an_id for e in entries}
            if prior_ids is not None and ids_on_page == prior_ids:
                logger.warning(
                    "harvest: page=%d returned same id-set as page=%d "
                    "(WAF page-1 fallback?); stopping walk",
                    page, page - 1,
                )
                break

            added = 0
            for e in entries:
                if e.ban_an_id in state.listings:
                    # Prefer the entry with the most metadata.
                    if not state.listings[e.ban_an_id].doc_number and e.doc_number:
                        state.listings[e.ban_an_id] = e
                else:
                    state.listings[e.ban_an_id] = e
                    added += 1

            state.pages_fetched += 1
            state.last_page = page
            logger.info(
                "harvest: page=%d cards=%d new=%d total=%d",
                page, len(entries), added, len(state.listings),
            )

            # Operator-facing short-circuit: stop walking once we have
            # at least ``--limit`` entries. Useful for smoke runs.
            if self._limit is not None and len(state.listings) >= int(self._limit):
                logger.info(
                    "harvest: --limit %s reached at page=%d (stop)",
                    self._limit, page,
                )
                break

            prior_ids = ids_on_page
            page += 1

        logger.info(
            "harvest: walked %d pages, collected %d unique judgment ids",
            state.pages_fetched, len(state.listings),
        )
        return state

    # ------------------------------------------------------ HTTP

    def _fetch_page(self, url: str, *, page: int) -> str:
        """GET one listing page. Cached under ``html/listings/page-<N>.html``."""
        cache = listings_dir(self.layout) / f"page-{page:05d}.html"
        if (
            self._cache_listings
            and cache.exists()
            and cache.stat().st_size > 0
        ):
            return cache.read_text(encoding="utf-8")
        assert self._session is not None
        resp = self._get_with_403_retry(url)
        if resp.status_code != 200:
            logger.warning(
                "harvest: page fetch failed status=%d url=%s",
                resp.status_code, url,
            )
            return ""
        text = resp.text or ""
        if self._cache_listings:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")
        return text

    def _fetch_taxonomy(self) -> dict[str, list[dict[str, Any]]]:
        """Snapshot the closed-set filter dropdowns on the search form."""
        # Lean on the first listing page since it carries the same form.
        cache = self.layout.html_dir / "taxonomy.html"
        html = ""
        if self._cache_listings and cache.exists() and cache.stat().st_size > 0:
            html = cache.read_text(encoding="utf-8")
        else:
            assert self._session is not None
            resp = self._get_with_403_retry(self._taxonomy_url)
            if resp.status_code == 200:
                html = resp.text or ""
                if self._cache_listings:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_text(html, encoding="utf-8")
            else:
                logger.warning(
                    "harvest: taxonomy fetch failed status=%d url=%s",
                    resp.status_code, self._taxonomy_url,
                )
        return _parse_taxonomy_html(html)

    def _get_with_403_retry(self, url: str) -> requests.Response:
        """Issue the GET, sleeping through any Cloudflare 403 cool-down.

        The shared :class:`PoliteSession` already retries 429 / 5xx
        with exponential backoff, but treats 403 as terminal. The
        thuvienphapluat.vn WAF returns 403 to soft-banned IPs for
        ~5-15 minutes at a time, so we add a flat-then-doubling
        cool-down capped at ``http_403_max_delay_s``. Mirrors the same
        logic in ``thuvienphapluat_tnpl/components/downloader.py``.
        """
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
        """Write ``listings.jsonl`` + ``taxonomy.json``. Returns ``(lst, tax)``."""
        lst_path = self.layout.jsonl_dir / "listings.jsonl"
        with lst_path.open("w", encoding="utf-8") as f:
            for ban_an_id in sorted(state.listings):
                e = state.listings[ban_an_id]
                row = {
                    "ban_an_id":    e.ban_an_id,
                    "slug":         e.slug,
                    "url":          e.url,
                    "title":        e.title,
                    "summary":      e.summary,
                    "doc_number":   e.doc_number,
                    "court":        e.court,
                    "issue_date":   e.issue_date,
                    "case_kind":    e.case_kind,
                    "procedure":    e.procedure,
                    "page_number":  None,    # caller-side enrichment
                    "harvested_at": state.fetched_at,
                }
                f.write(
                    json.dumps(
                        {k: row.get(k) for k in LISTING_JSONL_FIELDS},
                        ensure_ascii=False,
                    )
                )
                f.write("\n")

        tax_path = self.layout.jsonl_dir / "taxonomy.json"
        tax_payload: dict[str, Any] = {
            "host": self.layout.host,
            "captured_at": state.fetched_at,
            "source_url": self._taxonomy_url,
            "pages_fetched": state.pages_fetched,
            "last_page": state.last_page,
            "total_judgments": len(state.listings),
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


# ---- taxonomy parser -----------------------------------------------------


def _parse_taxonomy_html(html: str) -> dict[str, list[dict[str, Any]]]:
    """Snapshot every ``<select>`` on the search form."""
    if not html:
        return {}
    from bs4 import BeautifulSoup

    out: dict[str, list[dict[str, Any]]] = {}
    soup = BeautifulSoup(html, "html.parser")
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


# ---- helpers -------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_LISTING_URL",
    "DEFAULT_TAXONOMY_URL",
    "BananHarvester",
    "HarvestState",
]
