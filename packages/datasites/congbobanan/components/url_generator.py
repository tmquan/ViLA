"""Integer-ID URL generator for congbobanan.toaan.gov.vn.

The portal addresses each decision by a dense integer primary key
(``case_id``) rather than by a scrapeable listing page, so the URL
generator is pure arithmetic: enumerate ``[start_id, end_id]`` and
format each integer into the detail-page URL. No HTTP round trips are
required to produce the URL stream.

Some IDs are ghost records (the server returns HTTP 200 with a
placeholder page that has no metadata panel). Filtering those out
happens on the downloader side; the URL generator emits every
candidate ID.

URL families (see the reference scraper at
https://github.com/tmquan/datascraper/blob/main/congbobanan/scraper.py)::

    detail  https://congbobanan.toaan.gov.vn/2ta{case_id}t1cvn/chi-tiet-ban-an
    binary  https://congbobanan.toaan.gov.vn/3ta{case_id}t1cvn/
    file    https://congbobanan.toaan.gov.vn/5ta{case_id}t1cvn/<filename>
"""

from __future__ import annotations

import re
from collections.abc import Iterator

from nemo_curator.stages.text.download.base import URLGenerator

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

    Plain constructor args (no Hydra/OmegaConf):

    * ``start_id`` / ``end_id``: closed interval of case IDs to crawl.
    * ``detail_template``: override the ``/2ta{case_id}t1cvn/...`` shape
      if the site ever mirrors to a different path.

    No network I/O happens in :meth:`generate_urls`; the downloader pays
    the HTTP cost per URL and short-circuits ghost IDs. Use
    :meth:`iter_urls` to stream large ranges (the full corpus is ~2.1M
    IDs) without materialising the whole list.
    """

    def __init__(
        self,
        start_id: int = 1,
        end_id: int = 0,
        *,
        detail_template: str = DEFAULT_DETAIL_URL_TEMPLATE,
    ) -> None:
        self.start_id = int(start_id)
        self.end_id = int(end_id)
        self.detail_template = detail_template or DEFAULT_DETAIL_URL_TEMPLATE
        if self.end_id < self.start_id:
            raise ValueError(
                f"end_id ({self.end_id}) must be >= start_id ({self.start_id})"
            )

    def iter_urls(self) -> Iterator[str]:
        """Stream ``[start_id .. end_id]`` rendered through the template."""
        for i in range(self.start_id, self.end_id + 1):
            yield self.detail_template.format(case_id=i)

    def generate_urls(self) -> list[str]:
        """Return ``[start_id .. end_id]`` rendered through the template."""
        return list(self.iter_urls())


__all__ = [
    "DEFAULT_DETAIL_URL_TEMPLATE",
    "DEFAULT_PDF_URL_TEMPLATE",
    "CongbobananURLGenerator",
    "doc_id_from_url",
]
