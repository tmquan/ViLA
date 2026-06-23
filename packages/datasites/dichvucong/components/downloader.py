"""dichvucong DocumentDownloader: fetch + cache one search page as JSON.

Subclasses :class:`nemo_curator.stages.text.download.base.DocumentDownloader`.
Given a page pseudo-URL from :class:`DichvucongURLGenerator`, this:

1. decodes the ``(agency_type, impl_agency_id, page_index)`` locator,
2. POSTs ``procedure_advanced_search_service_v2`` for that page,
3. writes the JSON array to ``<download_dir>/<stem>.json`` via an
   atomic ``.tmp -> final`` rename, and
4. is **idempotent** — an existing non-empty ``<stem>.json`` short-
   circuits the fetch, so restarts and incremental re-crawls are cheap.

The base ``download`` is fully overridden (same pattern as luutru) so we
own the atomic write; ``_get_output_filename`` / ``_download_to_path``
are implemented only to satisfy the ABC and are never called.

Freshness: the per-page JSON is the raw, append-friendly capture. The
*record-level* freshness diff (new / amended / withdrawn procedures via
``QDCBID`` + content hash) is computed downstream by the Extractor +
state manifest — see ``wiki/DICHVUCONG.md`` §5.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.download.base import DocumentDownloader

from packages.common.http import PoliteSession, session_from_scraper_cfg
from packages.datasites.dichvucong.components import api

logger = logging.getLogger(__name__)


class DichvucongDocumentDownloader(DocumentDownloader):
    """Fetch one search-result page and cache it as a JSON array."""

    def __init__(self, cfg: Any, download_dir: str, *, verbose: bool = False) -> None:
        super().__init__(download_dir=download_dir, verbose=verbose)
        self.cfg = cfg
        sc = cfg.scraper
        self._rest_url = str(sc.get("rest_url", api.DEFAULT_REST_URL))
        self._referer = str(sc.get("referer", api.DEFAULT_REFERER))
        self._service = str(sc.get("search_service", api.SERVICE_SEARCH))
        self._record_per_page = int(sc.get("record_per_page", 50))
        self._num_workers = int(sc.get("num_workers", 2)) or None
        self.session: PoliteSession | None = None

    def download(self, url: str) -> str | None:
        loc = api.decode_page_url(url)
        stem = api.page_doc_name(loc)
        final_path = Path(self._download_dir) / f"{stem}.json"
        if final_path.exists() and final_path.stat().st_size > 0:
            if self._verbose:
                logger.info("page %s cached; skipping", stem)
            return str(final_path)

        if self.session is None:
            self.session = session_from_scraper_cfg(self.cfg)
        try:
            rows = api.search_page(
                self.session,
                self._rest_url,
                page_index=loc["page_index"],
                record_per_page=self._record_per_page,
                service=self._service,
                agency_type=loc["agency_type"],
                impl_agency_id=loc["impl_agency_id"],
                referer=self._referer,
            )
            tmp_path = str(final_path) + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False)
            os.replace(tmp_path, final_path)
            if self._verbose:
                logger.info("page %s -> %d rows", stem, len(rows))
            return str(final_path)
        except Exception as exc:  # log + skip one page, never crash the run
            logger.error("download failed for page %s: %s", stem, exc)
            return None

    def num_workers_per_node(self) -> int | None:
        return self._num_workers

    def _get_output_filename(self, url: str) -> str:
        return f"{api.page_doc_name(api.decode_page_url(url))}.json"

    def _download_to_path(  # pragma: no cover - bypassed by download()
        self, url: str, path: str
    ) -> tuple[bool, str | None]:
        raise NotImplementedError(
            "DichvucongDocumentDownloader.download() is overridden."
        )


__all__ = ["DichvucongDocumentDownloader"]
