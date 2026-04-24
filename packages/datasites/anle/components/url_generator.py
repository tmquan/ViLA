"""Anle URLGenerator: walk the Oracle ADF listing and emit detail URLs.

Subclasses :class:`nemo_curator.stages.text.download.base.URLGenerator`.
Curator's contract is ``generate_urls() -> list[str]``, so listing-row
fields (title, date, summary, court) are not carried in the URL stream;
they are re-discovered by :class:`AnleDocumentExtractor` from the detail
HTML saved alongside the PDF.

Two modes, selected by ``cfg.scraper.paginated``:

* Static mode -- a single listing URL (or a handful of
  filter-variants via ``cfg.scraper.listing_pages``) is fetched and
  parsed once.
* Paginated mode -- Oracle ADF ``selectedPage=N`` walk with
  exponential probe + binary search to auto-detect the last page
  (nguonanle serves ~200 pages of 10 rows).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterator

from bs4 import BeautifulSoup
from nemo_curator.stages.text.download.base import URLGenerator

from packages.common.http import PoliteSession

logger = logging.getLogger(__name__)


def _session_from_cfg(cfg: Any) -> PoliteSession:
    """Build a :class:`PoliteSession` from ``cfg.scraper``.

    Constructed lazily (inside ``generate_urls`` / ``_download_to_path``)
    because :class:`PoliteSession` holds a :class:`threading.Lock` which
    Ray cannot pickle across worker boundaries.
    """
    proxy = cfg.scraper.get("proxy", None)
    return PoliteSession(
        qps=float(cfg.scraper.qps),
        user_agent=str(cfg.scraper.user_agent),
        proxy=str(proxy) if proxy else None,
        timeout=float(cfg.scraper.timeout_s),
        max_retries=int(cfg.scraper.max_retries),
        verify_tls=bool(cfg.scraper.verify_tls),
        download_max_retries=int(cfg.scraper.get("download_max_retries", 50)),
        download_retry_delay_s=float(cfg.scraper.get("download_retry_delay_s", 30.0)),
    )


DEFAULT_LISTING_URL = "https://anle.toaan.gov.vn/webcenter/portal/anle/anle"

_ADF_TABLE_RE = re.compile(
    r"<table\s+class='table\s+table-bordered[^']*'>(.+?)</table>",
    re.DOTALL,
)
_DDOCNAME_RE = re.compile(r"dDocName=([A-Za-z0-9_-]+)")
DEFAULT_LISTING_SELECTORS: list[str] = ["a[href*='dDocName=']"]


class AnleURLGenerator(URLGenerator):
    """Enumerate anle detail-page URLs from the portal's listing surface.

    Stores only the (pickle-safe) OmegaConf cfg. The :class:`PoliteSession`
    is built lazily inside :meth:`generate_urls` because it holds a
    :class:`threading.Lock` that Ray cannot serialise across workers.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._detail_template = str(
            cfg.scraper.get(
                "detail_url_template",
                "https://anle.toaan.gov.vn/webcenter/portal/anle/chitietanle"
                "?dDocName={doc_name}",
            )
        )
        self._listing_url: str = str(cfg.scraper.listing_url) or DEFAULT_LISTING_URL
        self._paginated: bool = bool(cfg.scraper.get("paginated", False))
        self._page_param: str = str(cfg.scraper.get("page_param", "selectedPage"))
        self._start_page: int = int(cfg.scraper.get("start_page", 1))
        self._max_pages_cfg = cfg.scraper.get("max_pages", None)
        self._page_detect_cap = int(cfg.scraper.get("page_detect_cap", 5000))
        self._page_detect_probes: list[int] = list(
            cfg.scraper.get("page_detect_probes", [])
            or [10, 50, 100, 200, 500, 1000, 2000, 5000]
        )
        self._extra_params: dict[str, str] = {
            str(k): str(v)
            for k, v in (cfg.scraper.get("extra_params", {}) or {}).items()
        }
        selectors = cfg.scraper.get("selectors", {}) or {}
        self._listing_selectors: list[str] = list(
            selectors.get("listing_item", DEFAULT_LISTING_SELECTORS)
        )
        # Built on first use inside generate_urls().
        self.session: PoliteSession | None = None

    # ------------------------------------------------------ URLGenerator API

    def generate_urls(self) -> list[str]:
        """Return the de-duplicated detail-page URL list for this site."""
        if self.session is None:
            self.session = _session_from_cfg(self.cfg)
        seen: set[str] = set()
        urls: list[str] = []
        iterator = (
            self._iter_paginated_docnames()
            if self._paginated
            else self._iter_static_docnames()
        )
        for doc_name in iterator:
            if doc_name in seen:
                continue
            seen.add(doc_name)
            urls.append(self._detail_template.format(doc_name=doc_name))
        return urls

    # ------------------------------------------------------ static mode

    def _iter_static_docnames(self) -> Iterator[str]:
        assert self.session is not None
        for page_url in self._iter_listing_pages():
            resp = self.session.get(page_url)
            if resp.status_code != 200:
                logger.warning(
                    "listing fetch failed: url=%s status=%d",
                    page_url, resp.status_code,
                )
                continue
            yield from self._parse_listing(resp.text)

    def _iter_listing_pages(self) -> Iterator[str]:
        pages = list(self.cfg.scraper.get("listing_pages", []) or [])
        if pages:
            yield from pages
            return
        yield self._listing_url

    # ------------------------------------------------------ paginated mode

    def _iter_paginated_docnames(self) -> Iterator[str]:
        assert self.session is not None
        end_page = int(self._max_pages_cfg or self._detect_last_page())
        logger.info("paginated crawl: pages %d..%d", self._start_page, end_page)
        for page in range(self._start_page, end_page + 1):
            url = self._page_url(page)
            resp = self.session.get(url)
            if resp.status_code != 200:
                logger.warning(
                    "listing fetch failed: page=%d url=%s status=%d",
                    page, url, resp.status_code,
                )
                continue
            rows = list(self._parse_listing_table(resp.text))
            if not rows:
                logger.info("page %d empty; continuing", page)
                continue
            logger.info("page %d ok, rows=%d", page, len(rows))
            yield from rows

    def _page_url(self, page: int) -> str:
        from urllib.parse import urlencode

        params = dict(self._extra_params)
        params[self._page_param] = str(page)
        sep = "&" if "?" in self._listing_url else "?"
        return f"{self._listing_url}{sep}{urlencode(params)}"

    def _detect_last_page(self) -> int:
        """Auto-detect the last non-empty page via exponential probe + bisect.

        Oracle ADF wraps back to page-1 content (or returns an empty
        table) beyond the real last page. We use the first table row's
        doc_name as the wrap signal: if page N's first row matches
        page 1's, N is past the end.
        """
        first_rows = list(
            self._parse_listing_table(
                self.session.get(self._page_url(self._start_page)).text
            )
        )
        if not first_rows:
            logger.warning("detect_last_page: first page empty; defaulting to 1")
            return self._start_page
        first_key = first_rows[0]

        def is_past_end(page: int) -> bool:
            resp = self.session.get(self._page_url(page))
            rows = list(self._parse_listing_table(resp.text))
            return (not rows) or rows[0] == first_key

        lo, hi = self._start_page, self._page_detect_cap
        for probe in self._page_detect_probes:
            if probe <= lo or probe > self._page_detect_cap:
                continue
            logger.info("detect_last_page: probing %d", probe)
            if is_past_end(probe):
                hi = probe
                break
            lo = probe
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if is_past_end(mid):
                hi = mid
            else:
                lo = mid
            logger.info("detect_last_page: bisect lo=%d hi=%d", lo, hi)
        logger.info("detect_last_page: last=%d", lo)
        return lo

    # ------------------------------------------------------ listing parse

    def _parse_listing_table(self, html: str) -> Iterator[str]:
        m = _ADF_TABLE_RE.search(html)
        if not m:
            return
        soup = BeautifulSoup(f"<table>{m.group(1)}</table>", "lxml")
        for row in soup.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            title_link = cells[1].find("a", href=re.compile(r"dDocName="))
            if not title_link:
                continue
            href = title_link.get("href", "")
            doc_name = extract_doc_name(href)
            if doc_name:
                yield doc_name

    def _parse_listing(self, html: str) -> Iterator[str]:
        table_rows = list(self._parse_listing_table(html))
        if table_rows:
            yield from table_rows
            return
        soup = BeautifulSoup(html, "html.parser")
        for selector in self._listing_selectors:
            for a in soup.select(selector):
                href = a.get("href")
                if not href:
                    continue
                doc_name = extract_doc_name(href)
                if doc_name:
                    yield doc_name
            break


# ----------------------------------------------------------------- helpers


def absolutize(base_url: str, href: str) -> str:
    if href.startswith(("http://", "https://")):
        return href
    from urllib.parse import urljoin

    return urljoin(base_url, href)


def extract_doc_name(href: str) -> str | None:
    m = _DDOCNAME_RE.search(href)
    if m:
        return m.group(1)
    tail = href.rstrip("/").rsplit("/", 1)[-1]
    tail = tail.split("?", 1)[0].split("#", 1)[0]
    return tail or None


def extract_doc_name_from_url(url: str) -> str | None:
    """Pull the ``dDocName`` slug out of an anle detail-page URL."""
    return extract_doc_name(url)


__all__ = [
    "AnleURLGenerator",
    "DEFAULT_LISTING_SELECTORS",
    "DEFAULT_LISTING_URL",
    "absolutize",
    "extract_doc_name",
    "extract_doc_name_from_url",
]
