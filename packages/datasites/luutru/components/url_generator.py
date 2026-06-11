"""luutru URLGenerator: walk the vanban.aspx listing and emit detail URLs.

Subclasses :class:`nemo_curator.stages.text.download.base.URLGenerator`.
Curator's contract is ``generate_urls() -> list[str]``, so listing-row
fields (số hiệu, ngày ban hành, trích yếu) are not carried in the URL
stream; they are re-discovered by :class:`LuutruDocumentExtractor` from
the detail HTML the downloader caches alongside the PDF.

luutru.gov.vn exposes the document corpus through a GET-paginated
ASP.NET search surface::

    /vanban.aspx?type={all,qppl,cddh}&p=N&shvb=&htvb=&lvvb=&cqbh=&trynd=

Each listing page links ~10 detail pages of the form
``/xemchitietvanban.htm?id=<GUID>``. The pager exposes the last page
directly (``p=299`` for ``type=all``), so the last page is read from
the pager links; a wrap-detection fallback covers the rare case where
the pager is absent.

The server returns an IIS 500 stub to a bare request, so
:meth:`generate_urls` does a warm-up GET against
``cfg.scraper.warm_up_url`` (the home page) to seed the
``ASP.NET_SessionId`` cookie before walking the listing. The cookie
rides on the shared :class:`requests.Session` inside
:class:`PoliteSession`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlencode, urljoin

from bs4 import BeautifulSoup
from nemo_curator.stages.text.download.base import URLGenerator

from packages.common.http import PoliteSession, session_from_scraper_cfg

logger = logging.getLogger(__name__)


# Back-compat alias mirroring the anle module shape; both names point at
# the shared :func:`packages.common.http.session_from_scraper_cfg`.
_session_from_cfg = session_from_scraper_cfg


DEFAULT_LISTING_URL = "https://luutru.gov.vn/vanban.aspx"
DEFAULT_DETAIL_TEMPLATE = "https://luutru.gov.vn/xemchitietvanban.htm?id={doc_name}"
DEFAULT_WARM_UP_URL = "https://luutru.gov.vn/home.htm"

#: Detail-page links carry the document GUID in the ``id=`` query param.
_ID_RE = re.compile(r"[?&]id=([0-9a-fA-F-]{32,36})")
#: Pager links carry the page number in the ``p=`` query param.
_PAGE_RE = re.compile(r"[?&]p=(\d+)")
#: CSS selector matching the listing's detail-page anchors.
DEFAULT_LISTING_SELECTORS: list[str] = ["a[href*='xemchitietvanban.htm']"]


class LuutruURLGenerator(URLGenerator):
    """Enumerate luutru detail-page URLs from the vanban.aspx listing.

    Stores only the (pickle-safe) OmegaConf cfg. The
    :class:`PoliteSession` is built lazily inside :meth:`generate_urls`
    because it holds a :class:`threading.Lock` that Ray cannot
    serialise across workers.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self._detail_template: str = str(
            cfg.scraper.get("detail_url_template", DEFAULT_DETAIL_TEMPLATE)
        )
        self._listing_url: str = (
            str(cfg.scraper.get("listing_url", DEFAULT_LISTING_URL))
            or DEFAULT_LISTING_URL
        )
        self._warm_up_url: str = str(
            cfg.scraper.get("warm_up_url", DEFAULT_WARM_UP_URL)
        )
        self._page_param: str = str(cfg.scraper.get("page_param", "p"))
        self._start_page: int = int(cfg.scraper.get("start_page", 1))
        self._max_pages_cfg = cfg.scraper.get("max_pages", None)
        self._page_detect_cap = int(cfg.scraper.get("page_detect_cap", 5000))
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
        self._ensure_session()
        assert self.session is not None
        self._warm_up()

        seen: set[str] = set()
        urls: list[str] = []
        for doc_name in self._iter_docnames():
            if doc_name in seen:
                continue
            seen.add(doc_name)
            urls.append(self._detail_template.format(doc_name=doc_name))
        logger.info("luutru url generation: %d unique detail URLs", len(urls))
        return urls

    # ------------------------------------------------------ session + warm-up

    def _ensure_session(self) -> None:
        if self.session is None:
            self.session = _session_from_cfg(self.cfg)

    def _warm_up(self) -> None:
        """Seed the ASP.NET_SessionId cookie so vanban.aspx returns 200."""
        if not self._warm_up_url:
            return
        assert self.session is not None
        try:
            resp = self.session.get(self._warm_up_url)
            logger.info(
                "luutru warm-up GET %s -> %d", self._warm_up_url, resp.status_code,
            )
        except Exception as exc:  # warm-up is best-effort
            logger.warning("luutru warm-up failed (%s); continuing", exc)

    # ------------------------------------------------------ pagination walk

    def _iter_docnames(self) -> Iterator[str]:
        assert self.session is not None
        end_page = int(self._max_pages_cfg or self._detect_last_page())
        logger.info("luutru paginated crawl: pages %d..%d", self._start_page, end_page)
        for page in range(self._start_page, end_page + 1):
            url = self._page_url(page)
            resp = self.session.get(url)
            if resp.status_code != 200:
                logger.warning(
                    "listing fetch failed: page=%d url=%s status=%d",
                    page, url, resp.status_code,
                )
                continue
            rows = list(self._parse_listing(resp.text))
            if not rows:
                logger.info("page %d empty; continuing", page)
                continue
            logger.info("page %d ok, rows=%d", page, len(rows))
            yield from rows

    def _page_url(self, page: int) -> str:
        params = dict(self._extra_params)
        params[self._page_param] = str(page)
        sep = "&" if "?" in self._listing_url else "?"
        return f"{self._listing_url}{sep}{urlencode(params)}"

    def _detect_last_page(self) -> int:
        """Read the last page from the first listing page's pager.

        luutru's pager links every page (``...&p=N``) including the
        last, so the max ``p=`` value on page 1 is the page count. Falls
        back to ``start_page`` when the pager is missing.
        """
        assert self.session is not None
        resp = self.session.get(self._page_url(self._start_page))
        if resp.status_code != 200:
            logger.warning(
                "detect_last_page: first page status=%d; defaulting to 1",
                resp.status_code,
            )
            return self._start_page
        pages = [int(m) for m in _PAGE_RE.findall(resp.text)]
        last = max(pages) if pages else self._start_page
        last = min(last, self._page_detect_cap)
        logger.info("detect_last_page: last=%d", last)
        return last

    # ------------------------------------------------------ listing parse

    def _parse_listing(self, html: str) -> Iterator[str]:
        soup = BeautifulSoup(html, "html.parser")
        for selector in self._listing_selectors:
            found = False
            for a in soup.select(selector):
                href = a.get("href")
                if not href:
                    continue
                doc_name = extract_doc_name(str(href))
                if doc_name:
                    found = True
                    yield doc_name
            if found:
                return


# ----------------------------------------------------------------- helpers


def absolutize(base_url: str, href: str) -> str:
    if href.startswith(("http://", "https://")):
        return href
    return urljoin(base_url, href)


def extract_doc_name(href: str) -> str | None:
    """Pull the document GUID out of a ``...?id=<GUID>`` detail link."""
    m = _ID_RE.search(href)
    if m:
        return m.group(1)
    return None


def extract_doc_name_from_url(url: str) -> str | None:
    """Pull the document GUID out of a luutru detail-page URL."""
    return extract_doc_name(url)


__all__ = [
    "DEFAULT_DETAIL_TEMPLATE",
    "DEFAULT_LISTING_SELECTORS",
    "DEFAULT_LISTING_URL",
    "DEFAULT_WARM_UP_URL",
    "LuutruURLGenerator",
    "absolutize",
    "extract_doc_name",
    "extract_doc_name_from_url",
]
