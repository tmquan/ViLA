"""SPA listing harvester for vbpl.vn (replaces the dead sitemap harvest).

Context
-------
As of 2026-07 vbpl.vn no longer publishes per-document sitemap URLs (its
``sitemap.xml`` shrank to a single static shard). The document listings
moved to a **Next.js React Server Components** UI whose data is streamed
(``?_rsc=…``) and **gated behind reCAPTCHA v3**. The legacy
:class:`packages.datasites.vbpl.components.harvester.VbplSitemapHarvester`
therefore finds zero documents.

This harvester rebuilds discovery the same way the detail stage already
fetches documents: drive a stealth-configured Chromium through the
reCAPTCHA v3 flow (so the server releases the listing), then scrape the
rendered document links straight from the DOM. It emits the identical
``jsonl/sitemap.jsonl`` schema (:data:`SITEMAP_JSONL_FIELDS`) the detail
stage consumes, so the rest of the vbpl pipeline is unchanged.

Approach: DOM-scrape, not API-parse
-----------------------------------
We extract ``a[href*="/van-ban/chi-tiet/"]`` anchors and parse the
canonical ``/van-ban/chi-tiet/<slug>--<id>`` pattern (already used by the
detail stage). Scraping the rendered DOM is robust to the exact RSC/API
response shape -- we only need the page to *render* rows, which is
exactly what passing reCAPTCHA unlocks.

Runtime requirement (honest caveat)
-----------------------------------
reCAPTCHA v3 scores headless browsers low; on a box with no display it
returns **0 rows** (the page shell renders but the gated data never
arrives). Run this where reCAPTCHA can pass -- a real display, or a
virtual one (``xvfb-run -a python -m packages.datasites.vbpl --pipeline
harvest_spa``), or with ``scraper.headless=false`` on a desktop. When the
first page yields 0 links we log a loud reCAPTCHA hint rather than
silently writing an empty sitemap.

Pagination (tunable)
--------------------
The page-turn mechanism is URL-param based by default
(``?<page_param>=<n>``, ``page_param="page"``). If a live run shows a
different scheme, override ``scraper.page_param`` /
``scraper.listing_templates`` -- no code change needed. We stop a scope
after ``scraper.stop_after_empty`` consecutive pages yield no *new* IDs
(or at ``scraper.max_pages``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from packages.common import SiteLayout
from packages.datasites.vbpl._shared import SCOPES, SITEMAP_JSONL_FIELDS
from packages.datasites.vbpl.components.detail import (
    _DEFAULT_CHROMIUM_ARGS,
    _STEALTH_INIT_SCRIPT,
    _AsyncTokenBucket,
    _autodetect_full_chromium,
)

logger = logging.getLogger(__name__)

SITE_ORIGIN = "https://vbpl.vn"
#: Default per-scope listing landing pages.
DEFAULT_LISTING_TEMPLATES: dict[str, str] = {
    "trung_uong": "https://vbpl.vn/van-ban/trung-uong",
    "dia_phuong": "https://vbpl.vn/van-ban/dia-phuong",
}
#: Canonical detail-URL shape shared with the detail stage.
_DETAIL_RE = re.compile(r"/van-ban/chi-tiet/(?P<slug>[^/?#]+?)--(?P<id>\d+)")
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class VbplListingHarvester:
    """Playwright DOM-scraper for vbpl.vn document listings."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        s = cfg.scraper
        self._qps: float = max(0.001, float(s.get("qps", 0.5)))
        self._headless: bool = bool(s.get("headless", True))
        self._nav_timeout_ms: int = int(float(s.get("nav_timeout_s", 60.0)) * 1000)
        self._render_wait_s: float = float(s.get("api_wait_s", 25.0))
        self._user_agent: str = str(s.get("user_agent", "")) or DEFAULT_USER_AGENT
        self._warmup_url: str = str(s.get("warmup_url", "")) or SITE_ORIGIN + "/"
        self._verify_tls: bool = bool(s.get("verify_tls", True))
        self._executable_path: str = str(s.get("executable_path", "") or "")
        self._stealth: bool = bool(s.get("stealth", True))
        self._page_param: str = str(s.get("page_param", "page"))
        self._start_page: int = int(s.get("start_page", 1))
        self._max_pages = s.get("max_pages", None)
        self._stop_after_empty: int = int(s.get("stop_after_empty", 3))
        self._detail_selector: str = (
            str(s.get("detail_link_selector", "") or "")
            or 'a[href*="/van-ban/chi-tiet/"]'
        )
        templates = dict(s.get("listing_templates", {}) or {})
        self._templates: dict[str, str] = {**DEFAULT_LISTING_TEMPLATES, **templates}
        scopes_cfg = list(s.get("scopes", []) or [])
        self._scopes = tuple(x for x in scopes_cfg if x in SCOPES) or SCOPES
        self._limit = cfg.get("limit", None)

    # ------------------------------------------------------ entrypoint

    def run(self) -> tuple[Path, int]:
        return asyncio.run(self._run_async())

    async def _run_async(self) -> tuple[Path, int]:
        from playwright.async_api import async_playwright

        out_path = self.layout.jsonl_dir / "sitemap.jsonl"
        seen = _load_seen(out_path)
        logger.info(
            "listing harvest start: scopes=%s already_known=%d headless=%s",
            list(self._scopes), len(seen), self._headless,
        )

        bucket = _AsyncTokenBucket(qps=self._qps)
        harvested_at = _utc_now_iso()
        total_new = 0

        with out_path.open("a", encoding="utf-8") as out_f:
            async with async_playwright() as p:
                browser = await self._launch(p)
                try:
                    ctx = await browser.new_context(
                        user_agent=self._user_agent,
                        ignore_https_errors=not self._verify_tls,
                        locale="vi-VN",
                        timezone_id="Asia/Ho_Chi_Minh",
                        viewport={"width": 1366, "height": 768},
                    )
                    ctx.set_default_timeout(self._nav_timeout_ms)
                    if self._stealth:
                        await ctx.add_init_script(_STEALTH_INIT_SCRIPT)
                    await self._warmup(ctx)
                    for scope in self._scopes:
                        n = await self._harvest_scope(
                            ctx, scope=scope, seen=seen,
                            out_f=out_f, bucket=bucket, harvested_at=harvested_at,
                        )
                        total_new += n
                        if self._limit is not None and len(seen) >= int(self._limit):
                            logger.info("limit=%s reached; stopping", self._limit)
                            break
                finally:
                    await browser.close()

        logger.info(
            "listing harvest done: new_rows=%d total_rows=%d -> %s",
            total_new, len(seen), out_path,
        )
        if total_new == 0 and not seen:
            logger.warning(
                "harvested 0 document links. vbpl's listing is reCAPTCHA-v3 "
                "gated; a headless browser scores below threshold and the "
                "server withholds the rows. Re-run with a display: "
                "`xvfb-run -a python -m packages.datasites.vbpl --pipeline "
                "harvest_spa` (or scraper.headless=false on a desktop)."
            )
        return out_path, len(seen)

    # ------------------------------------------------------ per scope

    async def _harvest_scope(
        self,
        ctx: Any,
        *,
        scope: str,
        seen: set[str],
        out_f: Any,
        bucket: _AsyncTokenBucket,
        harvested_at: str,
    ) -> int:
        base = self._templates.get(scope)
        if not base:
            logger.warning("no listing template for scope %s; skipping", scope)
            return 0
        page_num = self._start_page
        empty_streak = 0
        new_rows = 0
        while True:
            if self._max_pages is not None and page_num > int(self._max_pages):
                break
            url = _with_page(base, self._page_param, page_num)
            await bucket.acquire()
            hrefs = await self._scrape_page(ctx, url)
            fresh = 0
            for href in hrefs:
                m = _DETAIL_RE.search(href)
                if not m:
                    continue
                item_id = m.group("id")
                key = f"{scope}:{item_id}"
                if key in seen:
                    continue
                seen.add(key)
                out_f.write(json.dumps(
                    _row(item_id=item_id, scope=scope, slug=m.group("slug"),
                         url=urljoin(SITE_ORIGIN + "/", href.lstrip("/")),
                         harvested_at=harvested_at),
                    ensure_ascii=False,
                ) + "\n")
                out_f.flush()
                fresh += 1
            new_rows += fresh
            logger.info(
                "scope=%s page=%d links=%d new=%d (total_known=%d)",
                scope, page_num, len(hrefs), fresh, len(seen),
            )
            if fresh == 0:
                empty_streak += 1
                if empty_streak >= self._stop_after_empty:
                    logger.info(
                        "scope=%s: %d consecutive empty pages; stopping",
                        scope, empty_streak,
                    )
                    break
            else:
                empty_streak = 0
            page_num += 1
        return new_rows

    async def _scrape_page(self, ctx: Any, url: str) -> list[str]:
        """Navigate one listing page and return every detail href on it."""
        page = await ctx.new_page()
        try:
            try:
                await page.goto(
                    url, timeout=self._nav_timeout_ms, wait_until="domcontentloaded",
                )
            except Exception as exc:
                logger.warning("nav failed %s: %s", url, exc)
                return []
            # reCAPTCHA + RSC render the rows asynchronously; spin until
            # at least one detail anchor appears or the budget is spent.
            deadline = asyncio.get_event_loop().time() + self._render_wait_s
            hrefs: list[str] = []
            while asyncio.get_event_loop().time() < deadline:
                hrefs = await page.eval_on_selector_all(
                    self._detail_selector,
                    "els => els.map(e => e.getAttribute('href')).filter(Boolean)",
                )
                if hrefs:
                    break
                await asyncio.sleep(0.5)
            return hrefs
        finally:
            await page.close()

    # ------------------------------------------------------ browser setup

    async def _launch(self, p: Any) -> Any:
        args = list(_DEFAULT_CHROMIUM_ARGS)
        if not self._headless:
            args = [a for a in args if a != "--headless=new"]
        kwargs: dict[str, Any] = {"headless": self._headless, "args": args}
        full = self._executable_path or _autodetect_full_chromium()
        if full:
            kwargs["executable_path"] = full
        else:
            logger.warning(
                "full Chromium not found; using chrome-headless-shell "
                "(reCAPTCHA v3 almost always flags it). "
                "Run `playwright install chromium`.",
            )
        return await p.chromium.launch(**kwargs)

    async def _warmup(self, ctx: Any) -> None:
        try:
            page = await ctx.new_page()
            try:
                await page.goto(
                    self._warmup_url, timeout=self._nav_timeout_ms,
                    wait_until="domcontentloaded",
                )
                await asyncio.sleep(4)
            finally:
                await page.close()
        except Exception as exc:
            logger.warning("warmup failed (continuing): %s", exc)


# ---- helpers -------------------------------------------------------------


def _with_page(base: str, param: str, n: int) -> str:
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}{param}={n}"


def _row(*, item_id: str, scope: str, slug: str, url: str, harvested_at: str) -> dict:
    full = {
        "item_id": item_id,
        "scope": scope,
        "slug": slug,
        "url": url,
        "lastmod": None,
        "changefreq": None,
        "priority": None,
        "harvested_at": harvested_at,
    }
    return {k: full.get(k) for k in SITEMAP_JSONL_FIELDS}


def _load_seen(path: Path) -> set[str]:
    """Resume: ``{scope}:{item_id}`` keys already in sitemap.jsonl."""
    seen: set[str] = set()
    if not path.exists() or path.stat().st_size == 0:
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid, iid = row.get("scope"), row.get("item_id")
            if sid and iid:
                seen.add(f"{sid}:{iid}")
    return seen


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_LISTING_TEMPLATES",
    "VbplListingHarvester",
]
