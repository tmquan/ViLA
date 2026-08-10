"""Anle URLGenerator: walk the Oracle ADF listing and emit detail URLs.

Subclasses :class:`nemo_curator.stages.text.download.base.URLGenerator`.
Curator's contract is ``generate_urls() -> list[str]``; listing-row fields
(title, date, court) are not carried in the URL stream -- they are
re-discovered by :class:`AnleExtractor` from the detail HTML saved
alongside the binary.

Two modes:

* Static -- a single listing URL (or a handful of filter-variants via
  ``listing_pages``) is fetched and parsed once.
* Paginated -- Oracle ADF ``selectedPage=N`` walk with exponential probe
  + binary search to auto-detect the last page (nguonanle serves ~200
  pages of 10 rows).

Self-contained: uses ``make_session(verify=False)`` (anle.toaan.gov.vn's
cert is signed by a VN CA not in the Mozilla bundle) and sends
``Accept: */*`` because Oracle ADF returns a JS-loopback page for
browser-like Accept headers.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from urllib.parse import urlencode, urljoin

from bs4 import BeautifulSoup
from loguru import logger
from nemo_curator.stages.text.download.base import URLGenerator

from packages.datasites._curator.base import make_session

DEFAULT_LISTING_URL = "https://anle.toaan.gov.vn/webcenter/portal/anle/nguonanle"
DEFAULT_DETAIL_TEMPLATE = (
    "https://anle.toaan.gov.vn/webcenter/portal/anle/chitietnguonanle"
    "?dDocName={doc_name}"
)
#: Oracle ADF returns a JS loopback page unless Accept is ``*/*``.
ADF_HEADERS = {"Accept": "*/*", "User-Agent": "anle-scraper/1.0"}

_ADF_TABLE_RE = re.compile(
    r"<table\s+class='table\s+table-bordered[^']*'>(.+?)</table>",
    re.DOTALL,
)
_DDOCNAME_RE = re.compile(r"dDocName=([A-Za-z0-9_-]+)")
DEFAULT_LISTING_SELECTORS: list[str] = ["a[href*='dDocName=']"]


class AnleURLGenerator(URLGenerator):
    """Enumerate anle detail-page URLs from the portal's listing surface."""

    def __init__(
        self,
        listing_url: str = DEFAULT_LISTING_URL,
        *,
        detail_url_template: str = DEFAULT_DETAIL_TEMPLATE,
        paginated: bool = True,
        page_param: str = "selectedPage",
        start_page: int = 1,
        max_pages: int | None = None,
        page_detect_cap: int = 5000,
        page_detect_probes: list[int] | None = None,
        extra_params: dict[str, str] | None = None,
        listing_pages: list[str] | None = None,
        listing_selectors: list[str] | None = None,
        proxy: str | None = None,
        timeout: int = 30,
        pace: float = 0.5,
    ) -> None:
        self._listing_url = listing_url or DEFAULT_LISTING_URL
        self._detail_template = detail_url_template
        self._paginated = paginated
        self._page_param = page_param
        self._start_page = start_page
        self._max_pages = max_pages
        self._page_detect_cap = page_detect_cap
        self._page_detect_probes = list(
            page_detect_probes or [10, 50, 100, 200, 500, 1000, 2000, 5000]
        )
        self._extra_params = {str(k): str(v) for k, v in (extra_params or {}).items()}
        self._listing_pages = list(listing_pages or [])
        self._listing_selectors = list(
            listing_selectors or DEFAULT_LISTING_SELECTORS
        )
        self._proxy = proxy
        self._timeout = timeout
        self._pace = pace
        self._sess = None

    # -- pickle: drop the live session ------------------------------------ #
    def __getstate__(self) -> dict:
        st = self.__dict__.copy()
        st["_sess"] = None
        return st

    def __setstate__(self, st: dict) -> None:
        self.__dict__.update(st)
        self._sess = None

    def _session(self):
        if self._sess is None:
            s = make_session(proxy=self._proxy, verify=False)
            s.headers.update(ADF_HEADERS)
            self._sess = s
        return self._sess

    def _get(self, url: str):
        return self._session().get(url, timeout=self._timeout, allow_redirects=True)

    # -- URLGenerator API -------------------------------------------------- #
    def generate_urls(self) -> list[str]:
        """Return the de-duplicated detail-page URL list."""
        return list(self.iter_urls())

    def iter_urls(self) -> Iterator[str]:
        """Stream detail-page URLs (de-duplicated) as they are discovered."""
        seen: set[str] = set()
        names = (
            self._iter_paginated_docnames()
            if self._paginated
            else self._iter_static_docnames()
        )
        for doc_name in names:
            if doc_name in seen:
                continue
            seen.add(doc_name)
            yield self._detail_template.format(doc_name=doc_name)

    # -- static mode ------------------------------------------------------- #
    def _iter_static_docnames(self) -> Iterator[str]:
        for page_url in self._listing_pages or [self._listing_url]:
            resp = self._get(page_url)
            if resp.status_code != 200:
                logger.warning(f"listing fetch failed: {page_url} -> {resp.status_code}")
                continue
            yield from self._parse_listing(resp.text)
            time.sleep(self._pace)

    # -- paginated mode ---------------------------------------------------- #
    def _iter_paginated_docnames(self) -> Iterator[str]:
        end_page = int(self._max_pages or self._detect_last_page())
        logger.info(f"paginated crawl: pages {self._start_page}..{end_page}")
        for page in range(self._start_page, end_page + 1):
            resp = self._get(self._page_url(page))
            if resp.status_code != 200:
                logger.warning(f"listing fetch failed: page={page} -> {resp.status_code}")
                continue
            rows = list(self._parse_listing_table(resp.text))
            if not rows:
                logger.info(f"page {page} empty; continuing")
            else:
                logger.info(f"page {page} ok, rows={len(rows)}")
            yield from rows
            time.sleep(self._pace)

    def _page_url(self, page: int) -> str:
        params = dict(self._extra_params)
        params[self._page_param] = str(page)
        sep = "&" if "?" in self._listing_url else "?"
        return f"{self._listing_url}{sep}{urlencode(params)}"

    def _detect_last_page(self) -> int:
        """Auto-detect the last non-empty page via exponential probe + bisect.

        Oracle ADF wraps back to page-1 content (or an empty table) beyond
        the real last page. We use the first row's doc_name as the wrap
        signal: if page N's first row matches page 1's, N is past the end.
        """
        first_rows = list(
            self._parse_listing_table(self._get(self._page_url(self._start_page)).text)
        )
        if not first_rows:
            logger.warning("detect_last_page: first page empty; defaulting to start")
            return self._start_page
        first_key = first_rows[0]

        def is_past_end(page: int) -> bool:
            rows = list(self._parse_listing_table(self._get(self._page_url(page)).text))
            return (not rows) or rows[0] == first_key

        lo, hi = self._start_page, self._page_detect_cap
        for probe in self._page_detect_probes:
            if probe <= lo or probe > self._page_detect_cap:
                continue
            logger.info(f"detect_last_page: probing {probe}")
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
            logger.info(f"detect_last_page: bisect lo={lo} hi={hi}")
        logger.info(f"detect_last_page: last={lo}")
        return lo

    # -- listing parse ----------------------------------------------------- #
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
            doc_name = extract_doc_name(title_link.get("href", ""))
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
    return urljoin(base_url, href)


def extract_doc_name(href: str) -> str | None:
    """Pull the ``dDocName`` slug out of an anle href/URL."""
    m = _DDOCNAME_RE.search(href)
    if m:
        return m.group(1)
    tail = href.rstrip("/").rsplit("/", 1)[-1]
    tail = tail.split("?", 1)[0].split("#", 1)[0]
    return tail or None


#: Back-compat alias.
extract_doc_name_from_url = extract_doc_name


__all__ = [
    "ADF_HEADERS",
    "DEFAULT_DETAIL_TEMPLATE",
    "DEFAULT_LISTING_SELECTORS",
    "DEFAULT_LISTING_URL",
    "AnleURLGenerator",
    "absolutize",
    "extract_doc_name",
    "extract_doc_name_from_url",
]
