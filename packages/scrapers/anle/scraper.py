"""Scraper for anle.toaan.gov.vn (Vietnamese Án lệ / precedent portal).

Stage 1 of the anle pipeline. Walks the precedent listing, fetches each
detail page, and downloads the attached PDF. Writes to:

    data/anle.toaan.gov.vn/
        pdf/<doc_id>.pdf           # streamed binary
        metadata/<doc_id>.json     # per-item header fields
        data.json                  # aggregated records (JSON array; small corpus)
        data.csv                   # aggregated records (CSV sidecar)
        progress.scrape.json       # resume checkpoint
        logs/scrape-<date>.jsonl   # operational log

HTML selectors are configurable (configs/anle.yaml scraper.selectors.*)
so an HTML change on the source does not require editing this file.

Run:
    python -m packages.scrapers.anle.scraper \
        --config-name anle --num-workers 4

    # With Hydra-style overrides (OmegaConf)
    python -m packages.scrapers.anle.scraper \
        --config-name anle --override scraper.qps=0.5 limit=5
"""

from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterator

from bs4 import BeautifulSoup

from packages.scrapers.common.base import SiteLayout, SiteScraperBase
from packages.scrapers.common.cli import apply_log_level, build_arg_parser, load_and_override
from packages.scrapers.common.config import resolve_config_path
from packages.scrapers.common.http import PoliteSession
from packages.scrapers.common.schemas import PipelineCfg

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).parent / "configs"

DEFAULT_LISTING_URL = (
    "https://anle.toaan.gov.vn/webcenter/portal/anle/anle"
)
DEFAULT_DETAIL_URL_TEMPLATE = (
    "https://anle.toaan.gov.vn/webcenter/portal/anle/chitietanle"
    "?dDocName={doc_name}"
)
# anle's detail page embeds a PDF.js viewer at /webcenter/pdfview whose
# FILE_URL JS variable points at this derived endpoint on the Oracle
# WebCenter Content server. The short form below works for every
# precedent source document (nguonanle); the longer
# `...//idcPrimaryFile&revision=latestreleased&rid=1` form also works
# and is equivalent -- the server resolves to the latest released rid
# either way.
DEFAULT_PDF_URL_TEMPLATE = (
    "https://anle.toaan.gov.vn/webcenter/ShowProperty"
    "?nodeId=/UCMServer/{doc_name}"
)

# Table regex used by Oracle ADF listing pages. The table is rendered
# server-side with single-quoted attributes inside a deeply nested ADF
# tree; lxml + CSS selectors can miss it, so we first pull the table
# by regex and then hand it to BeautifulSoup for row iteration.
_ADF_TABLE_RE = re.compile(
    r"<table\s+class='table\s+table-bordered[^']*'>(.+?)</table>",
    re.DOTALL,
)

# Defensive selectors. Every one is a sequence of CSS selectors tried in
# order; the first non-empty match wins. Keep permissive — anle's site
# uses WebCenter which can emit slightly different markup across releases.
#
# The default `listing_item` includes every content type anle publishes:
#   - chitietanle    formal Án lệ precedents (~80 docs)
#   - anleduthao     draft precedents under consideration
#   - chitietduthao  draft-precedent detail pages
#   - gioithieu      "introduction" / lead articles about a precedent
#   - chitiettin     news / commentary articles
#   - anlethegioi    foreign precedents (comparative material)
# Override `scraper.selectors.listing_item` to restrict to only formal
# precedents (`a[href*='chitietanle']`) when wanted.
DEFAULT_SELECTORS: dict[str, list[str]] = {
    "listing_item": [
        "a[href*='dDocName=']",
    ],
    "precedent_number": [
        "h1.al-title",
        "h1",
        ".al-header",
    ],
    "adopted_date": [
        ".al-adopted-date",
        "span.date",
    ],
    "applied_article": [
        ".al-applied-article",
        ".al-article",
    ],
    "principle_text": [
        ".al-principle",
        ".al-body",
        "article",
    ],
    "pdf_link": [
        "a[href$='.pdf']",
        "a[href*='.pdf']",
    ],
}


class AnleScraper(SiteScraperBase):
    """anle.toaan.gov.vn scraper (inherits SiteScraperBase)."""

    stage = "scrape"

    def __init__(
        self,
        cfg: Any,
        layout: SiteLayout,
        session: PoliteSession,
        *,
        limit: int | None = None,
        force: bool = False,
        resume: bool = True,
    ) -> None:
        super().__init__(
            layout=layout,
            session=session,
            num_workers=int(cfg.scraper.num_workers),
            limit=limit,
            force=force,
            resume=resume,
        )
        self._cfg = cfg
        self._listing_url: str = str(cfg.scraper.listing_url) or DEFAULT_LISTING_URL
        self._detail_template: str = (
            str(cfg.scraper.detail_url_template) or DEFAULT_DETAIL_URL_TEMPLATE
        )
        self._pdf_url_template: str = cfg.scraper.get(
            "pdf_url_template", DEFAULT_PDF_URL_TEMPLATE
        )
        selectors = cfg.scraper.get("selectors", None)
        self._selectors = (
            _merge_selectors(DEFAULT_SELECTORS, selectors)
            if selectors
            else DEFAULT_SELECTORS
        )
        self._records: list[dict[str, Any]] = []

        # Paginated-listing knobs (Oracle ADF selectedPage=N style).
        self._paginated: bool = bool(cfg.scraper.get("paginated", False))
        self._page_param: str = str(cfg.scraper.get("page_param", "selectedPage"))
        self._start_page: int = int(cfg.scraper.get("start_page", 1))
        self._max_pages_cfg: int | None = cfg.scraper.get("max_pages", None)
        self._page_detect_cap: int = int(cfg.scraper.get("page_detect_cap", 5000))
        self._page_detect_probes: list[int] = list(
            cfg.scraper.get("page_detect_probes", []) or [10, 50, 100, 200, 500, 1000, 2000, 5000]
        )
        self._extra_params: dict[str, str] = {
            str(k): str(v) for k, v in (cfg.scraper.get("extra_params", {}) or {}).items()
        }
        # Install extra HTTP headers on the shared session (e.g.
        # Accept: */* to bypass Oracle ADF's JS loopback page).
        extra_headers = cfg.scraper.get("extra_headers", {}) or {}
        if extra_headers:
            session._session.headers.update({str(k): str(v) for k, v in extra_headers.items()})
        self._fetch_detail: bool = bool(cfg.scraper.get("fetch_detail_page", True))
        self._fetch_head: bool = bool(cfg.scraper.get("fetch_head_before_download", True))

    # ----------------------------------------------------------- SiteScraperBase

    def iter_items(self) -> Iterator[dict[str, Any]]:
        """Yield {'doc_name': 'TAND-xxx', 'title': '...', 'html_url': '...'}.

        Two modes:
          - Static listing: yields all items from each URL in
            `scraper.listing_pages` (or `scraper.listing_url`).
          - Paginated listing: walks `listing_url?{page_param}=N&...`
            for N in [start_page, max_pages]. `max_pages` is
            auto-detected when unset by probing exponentially and
            bisecting on the first empty / wrapped page.
        """
        seen: set[str] = set()
        if self._paginated:
            iterator = self._iter_paginated_items()
        else:
            iterator = self._iter_static_items()
        for item in iterator:
            if item["doc_name"] in seen:
                continue
            seen.add(item["doc_name"])
            yield item

    def _iter_static_items(self) -> Iterator[dict[str, Any]]:
        for page_url in self._iter_listing_pages():
            resp = self.session.get(page_url)
            if resp.status_code != 200:
                self.log.warning(event="listing_fetch_failed",
                                 url=page_url, status=resp.status_code)
                continue
            yield from self._parse_listing(resp.text, page_url)

    def _iter_paginated_items(self) -> Iterator[dict[str, Any]]:
        end_page = self._max_pages_cfg or self._detect_last_page()
        logger.info("paginated crawl: pages %d..%d", self._start_page, end_page)
        for page in range(self._start_page, end_page + 1):
            url = self._page_url(page)
            resp = self.session.get(url)
            if resp.status_code != 200:
                self.log.warning(event="listing_fetch_failed",
                                 url=url, status=resp.status_code, page=page)
                continue
            rows = list(self._parse_listing_table(resp.text))
            if not rows:
                self.log.info(event="page_empty", page=page)
                continue
            self.log.info(event="page_ok", page=page, rows=len(rows))
            yield from rows

    def item_id(self, item: dict[str, Any]) -> str:
        return item["doc_name"]

    def is_item_complete(self, item_id: str) -> bool:
        pdf = self.layout.pdf_dir / f"{item_id}.pdf"
        meta = self.layout.metadata_dir / f"{item_id}.json"
        if not (pdf.exists() and meta.exists()):
            return False
        if pdf.stat().st_size == 0 or meta.stat().st_size == 0:
            return False
        try:
            json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return True

    def process_item(self, item: dict[str, Any]) -> dict[str, Any]:
        """Fetch detail page + PDF, write metadata + pdf, return record.

        PDF discovery: the detail page embeds a PDF.js viewer. Rather
        than scrape the viewer's JS, we build the PDF URL from the
        known `ShowProperty` template (configurable via
        `scraper.pdf_url_template`). A missing or broken override in
        the selectors table falls back to the template as well.
        """
        doc_name = item["doc_name"]
        detail_url = item.get("html_url") or self._detail_template.format(
            doc_name=doc_name
        )

        header: dict[str, Any] = {}
        if self._fetch_detail:
            detail_resp = self.session.get(detail_url)
            detail_resp.raise_for_status()
            header = self._parse_detail(detail_resp.text)

        pdf_url = (
            header.get("pdf_url")
            or item.get("pdf_url")
            or self._pdf_url_template.format(doc_name=doc_name)
        )

        # anle's WebCenter backend stores most precedents as PDF but a
        # handful of drafts / foreign precedents are DOCX. Probe the
        # content-type via HEAD first, then choose an extension.
        pdf_path = self._download_binary(pdf_url, doc_name)

        # Pull richer fields from the listing row when we skipped the
        # detail fetch (nguonanle pagination case).
        record = {
            "doc_name": doc_name,
            "precedent_number": header.get("precedent_number") or item.get("title"),
            "adopted_date": header.get("adopted_date") or item.get("date"),
            "applied_article": header.get("applied_article"),
            "principle_text": header.get("principle_text") or item.get("summary"),
            "court": header.get("court") or item.get("court"),
            "html_url": detail_url,
            "pdf_url": pdf_url,
            "pdf_path": str(pdf_path.relative_to(self.layout.output_root)),
            "source": self.layout.host,
        }

        meta_path = self.layout.metadata_dir / f"{doc_name}.json"
        meta_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._records.append(record)
        return record

    # ------------------------------------------------------------------- anle

    _MIME_TO_EXT: dict[str, str] = {
        "application/pdf": ".pdf",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    }

    def _download_binary(self, url: str, doc_name: str) -> Path:
        """Download a doc to disk. Accept PDF / DOC / DOCX based on MIME.

        When `scraper.fetch_head_before_download` is False we skip the
        HEAD probe and stream directly, trusting the default PDF
        extension -- reduces the per-item round trips by half on
        large corpora.
        """
        if self._fetch_head:
            try:
                head = self.session._session.head(
                    url, timeout=self.session._timeout, allow_redirects=True,
                    verify=self.session._session.verify,
                )
                content_type = head.headers.get("Content-Type", "").split(";")[0].strip()
            except Exception:
                content_type = "application/pdf"
            ext = self._MIME_TO_EXT.get(content_type, ".pdf")
            expected = content_type if content_type in self._MIME_TO_EXT else "application/pdf"
        else:
            ext = ".pdf"
            expected = "application/pdf"

        dest = self.layout.pdf_dir / f"{doc_name}{ext}"
        self.session.download(url, str(dest), expected_mime=expected)
        return dest

    def _iter_listing_pages(self) -> Iterator[str]:
        """Yield listing URLs for static (non-paginated) mode.

        A caller can override via scraper.listing_pages in the config to
        enumerate filter-variant URLs.
        """
        pages = list(self._cfg.scraper.get("listing_pages", []) or [])
        if pages:
            yield from pages
            return
        yield self._listing_url

    def _page_url(self, page: int) -> str:
        """Build a query-paginated listing URL for a 1-based page index."""
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
        page 1's, N is past the end. If the table is empty, same.
        """
        first_rows = list(self._parse_listing_table(
            self.session.get(self._page_url(self._start_page)).text
        ))
        if not first_rows:
            logger.warning("detect_last_page: first page empty; defaulting to 1")
            return self._start_page
        first_key = first_rows[0]["doc_name"]

        def is_past_end(page: int) -> bool:
            resp = self.session.get(self._page_url(page))
            rows = list(self._parse_listing_table(resp.text))
            return (not rows) or rows[0]["doc_name"] == first_key

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

    def _parse_listing_table(self, html: str) -> Iterator[dict[str, Any]]:
        """Parse the Oracle ADF listing <table class='table table-bordered'>.

        Columns (standard anle schema):
            [0] STT (row number)
            [1] Title/name (contains <a href='...?dDocName=...'>)
            [2] Date
            [3] Summary / extract (may contain a second <a>)
            [4] Court (optional)
        """
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
            doc_name = _extract_doc_name(href)
            if not doc_name:
                continue
            yield {
                "doc_name": doc_name,
                "title": title_link.get_text(strip=True),
                "date": cells[2].get_text(strip=True),
                "summary": cells[3].get_text(" ", strip=True),
                "court": cells[4].get_text(strip=True) if len(cells) >= 5 else "",
                "stt": cells[0].get_text(strip=True),
                "html_url": _absolute(self._listing_url, href),
            }

    def _parse_listing(self, html: str, base_url: str) -> Iterator[dict[str, Any]]:
        # Prefer the table path (Oracle ADF markup); fall back to
        # generic CSS selectors for portal pages without the standard
        # table shell (e.g. the formal /anle landing).
        table_rows = list(self._parse_listing_table(html))
        if table_rows:
            yield from table_rows
            return
        soup = BeautifulSoup(html, "html.parser")
        for selector in self._selectors["listing_item"]:
            for a in soup.select(selector):
                href = a.get("href")
                if not href:
                    continue
                url = _absolute(base_url, href)
                doc_name = _extract_doc_name(href)
                if not doc_name:
                    continue
                yield {
                    "doc_name": doc_name,
                    "title": (a.get_text() or "").strip(),
                    "html_url": url,
                }
            break

    def _parse_detail(self, html: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        header: dict[str, Any] = {}

        header["precedent_number"] = _first_text(soup, self._selectors["precedent_number"])
        header["adopted_date"] = _first_text(soup, self._selectors["adopted_date"])
        header["applied_article"] = _first_text(soup, self._selectors["applied_article"])
        header["principle_text"] = _first_text(soup, self._selectors["principle_text"])
        header["pdf_url"] = _first_href(soup, self._selectors["pdf_link"])
        if header["pdf_url"]:
            # Make absolute if needed; use the detail URL origin.
            header["pdf_url"] = _absolute(
                "https://anle.toaan.gov.vn/", header["pdf_url"]
            )
        return header

    # ------------------------------------------------------------- aggregation

    def finalize(self) -> None:
        """Write aggregated data.json / data.csv after run()."""
        data_json = self.layout.site_root / "data.json"
        data_csv = self.layout.site_root / "data.csv"

        existing: list[dict[str, Any]] = []
        if data_json.exists():
            try:
                existing = json.loads(data_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []
        merged = {r["doc_name"]: r for r in existing}
        for r in self._records:
            merged[r["doc_name"]] = r

        rows = sorted(merged.values(), key=lambda r: r["doc_name"])
        data_json.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if rows:
            fieldnames = list(rows[0].keys())
            with data_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow({k: _scalar(r.get(k, "")) for k in fieldnames})


# ----------------------------------------------------------------- helpers


def _absolute(base_url: str, href: str) -> str:
    if href.startswith(("http://", "https://")):
        return href
    from urllib.parse import urljoin

    return urljoin(base_url, href)


def _extract_doc_name(href: str) -> str | None:
    m = re.search(r"dDocName=([A-Za-z0-9_-]+)", href)
    if m:
        return m.group(1)
    # Fallback: use the last path segment without extension.
    tail = href.rstrip("/").rsplit("/", 1)[-1]
    tail = tail.split("?", 1)[0].split("#", 1)[0]
    return tail or None


def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for s in selectors:
        node = soup.select_one(s)
        if node is not None:
            text = (node.get_text(separator=" ") or "").strip()
            if text:
                return text
    return None


def _first_href(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for s in selectors:
        node = soup.select_one(s)
        if node is not None and node.get("href"):
            return node["href"]
    return None


def _merge_selectors(
    base: dict[str, list[str]],
    override: Any,
) -> dict[str, list[str]]:
    if not override:
        return dict(base)
    out = {k: list(v) for k, v in base.items()}
    for key, sels in override.items():
        out[key] = list(sels)
    return out


def _scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser(
        description="Scraper for anle.toaan.gov.vn (stage 1).",
        stage="scrape",
    )
    args = parser.parse_args(argv)
    apply_log_level(args.log_level)

    config_path = resolve_config_path(
        args.config, args.config_name, CONFIGS_DIR, default_name="anle"
    )
    cfg = load_and_override(
        config_path=config_path,
        overrides=args.override,
        schema_cls=PipelineCfg,
    )

    layout = SiteLayout(
        output_root=Path(args.output).expanduser().resolve(),
        host=str(cfg.host),
    )
    session = PoliteSession(
        qps=float(cfg.scraper.qps),
        user_agent=str(cfg.scraper.user_agent),
        proxy=args.proxy or (str(cfg.scraper.proxy) if cfg.scraper.proxy else None),
        timeout=float(cfg.scraper.timeout_s),
        max_retries=int(cfg.scraper.max_retries),
        verify_tls=bool(cfg.scraper.verify_tls),
    )

    scraper = AnleScraper(
        cfg=cfg,
        layout=layout,
        session=session,
        limit=args.limit,
        force=args.force,
        resume=not args.no_resume,
    )
    counts = scraper.run()
    scraper.finalize()
    session.close()

    logger.info("scrape done: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
