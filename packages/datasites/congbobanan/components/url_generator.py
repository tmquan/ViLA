"""Integer-ID URL generator for congbobanan.toaan.gov.vn.

The portal addresses each decision by a dense integer primary key
(``case_id``) rather than by a scrapeable listing page, so the URL
generator is pure arithmetic: enumerate ``[cfg.scraper.start_id,
cfg.scraper.end_id]`` and format each integer into the detail-page
URL. No HTTP round trips are required to produce the URL stream.

Some IDs are ghost records (the server returns HTTP 200 with a
placeholder page that has no metadata panel). Filtering those out
happens on the downloader side; the URL generator emits every
candidate ID.

URL pattern
-----------

    https://congbobanan.toaan.gov.vn/2ta{case_id}t1cvn/chi-tiet-ban-an

See the reference scraper at https://github.com/tmquan/datascraper/blob/main/congbobanan/scraper.py
for the `/2ta{id}t1cvn/...` / `/3ta{id}t1cvn/...` / `/5ta{id}t1cvn/...`
URL shape.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from nemo_curator.stages.text.download.base import URLGenerator

logger = logging.getLogger(__name__)


DEFAULT_DETAIL_URL_TEMPLATE = (
    "https://congbobanan.toaan.gov.vn/2ta{case_id}t1cvn/chi-tiet-ban-an"
)
DEFAULT_PDF_URL_TEMPLATE = "https://congbobanan.toaan.gov.vn/3ta{case_id}t1cvn/"

#: Matches the ``{case_id}`` embedded in either the ``2ta<id>t1cvn``
#: (detail) or ``3ta<id>t1cvn`` / ``5ta<id>t1cvn`` (pdf / download)
#: URL families.
_URL_ID_RE = re.compile(r"/[235]ta(\d+)t1cvn(?:/|$)")


def doc_id_from_url(url: str) -> str | None:
    """Pull the numeric ``case_id`` slug out of a congbobanan URL.

    Returns the integer as a string so it can round-trip through
    filesystem paths and parquet columns without coercion.
    """
    m = _URL_ID_RE.search(url or "")
    return m.group(1) if m else None


class CongbobananURLGenerator(URLGenerator):
    """Enumerate congbobanan detail-page URLs from an integer ID range.

    Config driven entirely by ``cfg.scraper``:

    * ``start_id`` / ``end_id``: closed interval of case IDs to crawl.
    * ``detail_url_template``: override the ``/2ta{case_id}t1cvn/...``
      shape if the site ever mirrors to a different path.

    No network I/O happens in :meth:`generate_urls`; the downloader
    pays the HTTP cost per URL and short-circuits ghost IDs.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        # ``cfg.scraper.detail_url_template`` is ``""`` by default in
        # the schema (not ``None``), so ``.get(key, fallback)`` never
        # fires. Fall back on the empty-string-falsy ``or`` pattern.
        self._detail_template: str = (
            str(cfg.scraper.get("detail_url_template", ""))
            or DEFAULT_DETAIL_URL_TEMPLATE
        )
        self._start_id: int = int(cfg.scraper.get("start_id", 1))
        self._end_id: int = int(cfg.scraper.get("end_id", 0))
        if self._end_id < self._start_id:
            raise ValueError(
                f"cfg.scraper.end_id ({self._end_id}) must be >= start_id "
                f"({self._start_id})"
            )

    # ------------------------------------------------------ URLGenerator API

    def generate_urls(self) -> list[str]:
        """Return ``[start_id .. end_id]`` rendered through the template."""
        total = self._end_id - self._start_id + 1
        logger.info(
            "congbobanan URL generator: emitting %d IDs in [%d..%d]",
            total, self._start_id, self._end_id,
        )
        return [
            self._detail_template.format(case_id=i)
            for i in range(self._start_id, self._end_id + 1)
        ]


__all__ = [
    "CongbobananURLGenerator",
    "DEFAULT_DETAIL_URL_TEMPLATE",
    "DEFAULT_PDF_URL_TEMPLATE",
    "doc_id_from_url",
]
