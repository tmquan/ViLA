"""dichvucong URLGenerator: enumerate search-result pages to fetch.

Subclasses :class:`nemo_curator.stages.text.download.base.URLGenerator`.
The national portal exposes its whole administrative-procedure corpus
through the paginated ``procedure_advanced_search_service_v2`` call, so
"enumerate the corpus" means "enumerate the non-empty search pages".

Two enumeration modes:

* **Unfiltered** (default) — walk ``pageIndex = start..`` with no agency
  filter; one URL per page. Simplest "everything" crawl.
* **Agency-sharded** — when ``cfg.scraper.shard_by_agency`` is set, pull
  the agency list (``procedure_get_list_agency_by_type_service_v2``) and
  walk pages per ``impl_agency_id``. Slower but gives a deterministic,
  resumable shard key and dodges any deep-pagination cap on the
  unfiltered query.

Page count is auto-detected by walking until a short/empty page
(``len(rows) < recordPerPage``), capped by ``cfg.scraper.max_pages``.
Only the (pickle-safe) cfg is stored; the :class:`PoliteSession` is
built lazily inside :meth:`generate_urls` (it owns an unpicklable lock).
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_curator.stages.text.download.base import URLGenerator

from packages.common.http import PoliteSession, session_from_scraper_cfg
from packages.datasites.dichvucong.components import api

logger = logging.getLogger(__name__)


class DichvucongURLGenerator(URLGenerator):
    """Emit one pseudo-URL per non-empty search-result page."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        sc = cfg.scraper
        self._rest_url: str = str(sc.get("rest_url", api.DEFAULT_REST_URL))
        self._referer: str = str(sc.get("referer", api.DEFAULT_REFERER))
        self._service: str = str(sc.get("search_service", api.SERVICE_SEARCH))
        self._record_per_page: int = int(sc.get("record_per_page", 50))
        self._start_page: int = int(sc.get("start_page", 1))
        self._max_pages: int = int(sc.get("max_pages", 2000))
        self._agency_type: int = int(sc.get("agency_type", 1))
        self._shard_by_agency: bool = bool(sc.get("shard_by_agency", False))
        self.session: PoliteSession | None = None

    def generate_urls(self) -> list[str]:
        if self.session is None:
            self.session = session_from_scraper_cfg(self.cfg)
        shards = self._agency_shards()
        urls: list[str] = []
        for impl_agency_id in shards:
            urls.extend(self._pages_for_agency(impl_agency_id))
        logger.info("dichvucong: enumerated %d search page(s)", len(urls))
        return urls

    # ------------------------------------------------------ internals

    def _agency_shards(self) -> list[int]:
        if not self._shard_by_agency:
            return [-1]  # -1 == no agency filter (everything)
        rows = api.call_ref(
            self.session,
            self._rest_url,
            api.SERVICE_AGENCIES,
            referer=self._referer,
            loaicoquan=self._agency_type,
        )
        ids = [int(r["ID"]) for r in rows if r.get("ID")]
        logger.info("dichvucong: sharding by %d agencies", len(ids))
        return ids or [-1]

    def _pages_for_agency(self, impl_agency_id: int) -> list[str]:
        out: list[str] = []
        page = self._start_page
        while page < self._start_page + self._max_pages:
            rows = api.search_page(
                self.session,
                self._rest_url,
                page_index=page,
                record_per_page=self._record_per_page,
                service=self._service,
                agency_type=self._agency_type,
                impl_agency_id=impl_agency_id,
                referer=self._referer,
            )
            if not rows:
                break
            out.append(
                api.encode_page_url(
                    self._rest_url,
                    agency_type=self._agency_type,
                    impl_agency_id=impl_agency_id,
                    page_index=page,
                )
            )
            if len(rows) < self._record_per_page:
                break  # last (short) page
            page += 1
        return out


__all__ = ["DichvucongURLGenerator"]
