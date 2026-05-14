"""Playwright-driven detail fetcher for vbpl.vn.

Reads the ``jsonl/sitemap.jsonl`` produced by
:class:`packages.datasites.vbpl.components.harvester.VbplSitemapHarvester`
and, for every row, drives a headless Chromium tab against the
``/van-ban/chi-tiet/<slug>--<id>`` URL. The site's reCAPTCHA v3 -> Bearer
token flow runs exactly as for a real visitor; we just install a
``page.on("response", ...)`` listener that captures every JSON
response under ``/api/qtdc/public/doc/...`` for the page.

What lands on disk per ItemID:

* ``html/<scope>/<id>.html``    -- the rendered page snapshot (post-load).
* ``html/<scope>/<id>.api.json`` -- the captured API responses (a
  ``{"<api_url>": <payload>, ...}`` map). Kept verbatim so a future
  extractor can re-parse without re-running the slow browser fetch.
* ``pdf/<scope>/<id>.{pdf,doc,docx}`` -- the original document file
  when the API exposes one (downloaded with the captured Bearer).
* one row in ``jsonl/docs.jsonl``.

Concurrency is one persistent :class:`BrowserContext` per worker
(``cfg.scraper.num_workers``). All workers share a global async token
bucket pacing total navigations at ``cfg.scraper.qps`` QPS so adding
workers smoothes out per-tab latency without breaching the rate limit.

Resume: any ItemID whose ``html/<scope>/<id>.html`` exists is
short-circuited (no browser navigation, no row written). The detail
JSONL is appended-to on resume so old rows are preserved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from packages.common import SiteLayout
from packages.datasites.vbpl._shared import (
    DETAIL_JSONL_FIELDS,
    scope_html_dir,
    scope_pdf_dir,
)
from packages.datasites.vbpl.components.parser import (
    DetailRecord,
    FilePath,
    detail_record_from_api_json,
)

logger = logging.getLogger(__name__)


DEFAULT_WARMUP_URL = "https://vbpl.vn/"
DEFAULT_API_URL_SUBSTR = "/api/qtdc/public/doc/"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# Initial-stealth JS pasted into every page before any site code runs.
# Masks the most-fingerprintable webdriver tells so reCAPTCHA v3 scores
# the headless tab closer to a regular Chrome session. Not bulletproof
# but cheap, and the difference between "no API call ever fires" and
# "API call fires reliably" on vbpl.vn.
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {
  get: () => ['vi-VN', 'vi', 'en-US', 'en'],
});
Object.defineProperty(navigator, 'plugins', {
  get: () => [
    {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer'},
    {name: 'Chrome PDF Viewer',
     filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai'},
    {name: 'Native Client', filename: 'internal-nacl-plugin'},
  ],
});
window.chrome = window.chrome || {runtime: {}};
const _q = navigator.permissions && navigator.permissions.query;
if (_q) {
  navigator.permissions.query = (p) =>
    p && p.name === 'notifications'
      ? Promise.resolve({state: Notification.permission})
      : _q.call(navigator.permissions, p);
}
"""

# Default Chromium launch flags that make reCAPTCHA v3 happier.
# `--headless=new` opts into the modern headless mode (real-browser
# rendering pipeline) rather than the old `chrome-headless-shell`
# behavior; the rest knock down a few of the most obvious automation
# tells.
_DEFAULT_CHROMIUM_ARGS: tuple[str, ...] = (
    "--headless=new",
    "--disable-blink-features=AutomationControlled",
    "--disable-features=IsolateOrigins,site-per-process",
    "--no-sandbox",
    "--disable-dev-shm-usage",
)

# Filesystem subpaths under each ``chromium-N`` Playwright cache dir
# where the full Chromium binary may live. Recent Playwright builds
# use ``chrome-linux64/chrome``; older ones used ``chrome-linux/chrome``.
# We probe both shapes when auto-detecting.
_FULL_CHROMIUM_SUBPATHS: tuple[str, ...] = (
    "chrome-linux64/chrome",
    "chrome-linux/chrome",
    "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    "chrome-win/chrome.exe",
)


class VbplDetailDownloader:
    """Per-ItemID Playwright fetcher.

    The class is sync-facing (``run() -> Path``) so the shared
    :func:`packages.common.runner.run_crawler_site` driver can call it
    like every other crawler. Internally it spins up an
    :func:`asyncio.run` boundary because Playwright's async API is the
    only one that exposes per-response callbacks.
    """

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._num_workers: int = max(1, int(cfg.scraper.get("num_workers", 2)))
        self._qps: float = max(0.001, float(cfg.scraper.get("qps", 0.5)))
        self._headless: bool = bool(cfg.scraper.get("headless", True))
        self._browser: str = str(cfg.scraper.get("browser", "chromium"))
        self._nav_timeout_ms: int = int(
            float(cfg.scraper.get("nav_timeout_s", 60.0)) * 1000,
        )
        self._user_agent: str = (
            str(cfg.scraper.get("user_agent", "")) or DEFAULT_USER_AGENT
        )
        self._warmup_url: str = (
            str(cfg.scraper.get("warmup_url", "")) or DEFAULT_WARMUP_URL
        )
        self._api_substr: str = (
            str(cfg.scraper.get("api_url_substr", "")) or DEFAULT_API_URL_SUBSTR
        )
        self._download_files: bool = bool(
            cfg.scraper.get("download_files", True),
        )
        self._proxy: str | None = cfg.scraper.get("proxy", None) or None
        self._verify_tls: bool = bool(cfg.scraper.get("verify_tls", True))
        # How long to wait for the first /api/qtdc/... XHR after
        # DOMContentLoaded. networkidle alone is unreliable on the
        # vbpl Next.js shell because reCAPTCHA polls keep the network
        # busy; we spin on the captured-list length up to this cap.
        self._api_wait_s: float = float(cfg.scraper.get("api_wait_s", 25.0))
        # Optional override of the full-Chromium binary path. When
        # empty we auto-detect by probing :data:`_FULL_CHROMIUM_HINTS`.
        # If both the autodetect and the explicit override come up
        # empty, the launch falls back to Playwright's default
        # chrome-headless-shell (which reCAPTCHA almost always flags).
        self._executable_path: str = str(
            cfg.scraper.get("executable_path", "") or "",
        )
        # Whether to install the per-context stealth init script.
        # On by default; disable for diagnostic runs where you want
        # to confirm a fingerprinting failure is the actual cause.
        self._stealth: bool = bool(cfg.scraper.get("stealth", True))
        self._limit = cfg.get("limit", None)
        self._run_id = _make_run_id()

    # ------------------------------------------------------ entrypoint

    def run(self) -> Path:
        """Drive every ItemID through Playwright; return the docs.jsonl path.

        Ordering: ``--limit`` is applied **before** the skip-if-exists
        filter. That way re-running with the same ``--limit N`` covers
        the same first ``N`` rows of ``sitemap.jsonl`` and is fully
        idempotent (every row short-circuits on the html cache). To
        progress past the boundary on each run, drop the limit.
        """
        sitemap_path = self.layout.jsonl_dir / "sitemap.jsonl"
        if not sitemap_path.exists():
            raise FileNotFoundError(
                f"{sitemap_path} missing; run --pipeline harvest first.",
            )
        rows = list(_iter_jsonl(sitemap_path))
        total_in_sitemap = len(rows)
        if self._limit is not None:
            rows = rows[: int(self._limit)]
        rows_to_fetch = [r for r in rows if _needs_fetch(r, self.layout)]
        skipped = len(rows) - len(rows_to_fetch)
        logger.info(
            "detail run: %d/%d sitemap rows in scope; skip-existing=%d; "
            "to fetch=%d; workers=%d qps=%.3f run_id=%s",
            len(rows), total_in_sitemap, skipped, len(rows_to_fetch),
            self._num_workers, self._qps, self._run_id,
        )
        if not rows_to_fetch:
            logger.info("nothing to fetch; --limit slice fully covered on disk")
            return self.layout.jsonl_dir / "docs.jsonl"

        return asyncio.run(self._run_async(rows_to_fetch))

    # ------------------------------------------------------ async core

    async def _run_async(self, rows: list[dict[str, Any]]) -> Path:
        # Lazy-imported so the rest of the package (harvest, parsers,
        # __main__ smoke tests) is usable without playwright installed.
        from playwright.async_api import async_playwright

        out_path = self.layout.jsonl_dir / "docs.jsonl"
        # Append on resume so prior rows survive partial reruns.
        write_lock = asyncio.Lock()
        bucket = _AsyncTokenBucket(qps=self._qps)

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for r in rows:
            queue.put_nowait(r)

        ok = err = 0
        # Open file in append mode so re-runs don't truncate. The
        # writer is sync inside an async lock; vbpl is bottlenecked
        # by reCAPTCHA + nav latency, not disk.
        with out_path.open("a", encoding="utf-8") as out_f:
            async with async_playwright() as p:
                browser_type = getattr(p, self._browser)
                launch_kwargs: dict[str, Any] = {"headless": self._headless}
                if self._proxy:
                    launch_kwargs["proxy"] = {"server": self._proxy}
                if self._browser == "chromium":
                    launch_kwargs["args"] = list(_DEFAULT_CHROMIUM_ARGS)
                    full_chromium = (
                        self._executable_path or _autodetect_full_chromium()
                    )
                    if full_chromium:
                        launch_kwargs["executable_path"] = full_chromium
                        logger.info(
                            "using full Chromium at %s (reCAPTCHA-friendlier "
                            "than chrome-headless-shell)", full_chromium,
                        )
                    else:
                        logger.warning(
                            "full Chromium binary not found; falling back to "
                            "chrome-headless-shell. reCAPTCHA v3 may score "
                            "this run below the threshold; install full "
                            "Chromium with `playwright install chromium` "
                            "or set scraper.executable_path explicitly.",
                        )
                browser = await browser_type.launch(**launch_kwargs)
                try:
                    workers = [
                        asyncio.create_task(
                            self._worker_loop(
                                browser=browser,
                                wid=i,
                                queue=queue,
                                bucket=bucket,
                                out_f=out_f,
                                write_lock=write_lock,
                                tally=_Tally(),
                            ),
                        )
                        for i in range(self._num_workers)
                    ]
                    tallies = await asyncio.gather(*workers)
                finally:
                    await browser.close()

        for t in tallies:
            ok += t.ok
            err += t.err
        logger.info(
            "detail run done: ok=%d err=%d -> %s", ok, err, out_path,
        )
        self._write_manifest(out_path=out_path, ok=ok, err=err)
        return out_path

    async def _worker_loop(
        self,
        *,
        browser: Any,
        wid: int,
        queue: asyncio.Queue[dict[str, Any]],
        bucket: "_AsyncTokenBucket",
        out_f: Any,
        write_lock: asyncio.Lock,
        tally: "_Tally",
    ) -> "_Tally":
        """One worker: warm up Bearer once, then drain the queue."""
        ctx_kwargs: dict[str, Any] = {
            "user_agent": self._user_agent,
            "ignore_https_errors": not self._verify_tls,
            # vi-VN matches the site's localisation and pairs with
            # the navigator.languages override in the stealth script
            # so reCAPTCHA sees a coherent fingerprint.
            "locale": "vi-VN",
            "timezone_id": "Asia/Ho_Chi_Minh",
            "viewport": {"width": 1366, "height": 768},
        }
        ctx = await browser.new_context(**ctx_kwargs)
        ctx.set_default_timeout(self._nav_timeout_ms)
        if self._stealth:
            await ctx.add_init_script(_STEALTH_INIT_SCRIPT)
        try:
            bearer_box: dict[str, str | None] = {"value": None}
            await self._warmup(ctx, bearer_box=bearer_box)
            while True:
                try:
                    row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return tally
                try:
                    await bucket.acquire()
                    await self._process_one(
                        ctx=ctx,
                        row=row,
                        out_f=out_f,
                        write_lock=write_lock,
                        bearer_box=bearer_box,
                    )
                    tally.ok += 1
                except Exception as exc:  # noqa: BLE001 - surface in JSONL
                    tally.err += 1
                    logger.exception(
                        "worker=%d item=%s crashed", wid, row.get("item_id"),
                    )
                    await self._write_failed(
                        row=row,
                        status=f"crash:{type(exc).__name__}",
                        error=repr(exc),
                        out_f=out_f,
                        write_lock=write_lock,
                    )
                finally:
                    n = tally.ok + tally.err
                    if n % 25 == 0:
                        logger.info(
                            "worker=%d progress: ok=%d err=%d",
                            wid, tally.ok, tally.err,
                        )
        finally:
            await ctx.close()

    async def _warmup(
        self,
        ctx: Any,
        *,
        bearer_box: dict[str, str | None],
    ) -> None:
        """Visit the site root once so reCAPTCHA mints the Bearer token.

        We don't strictly need to capture the token here -- the
        per-item navigation will trigger reCAPTCHA again as needed --
        but warming up the context primes the Google captcha cookies
        so subsequent navigations don't pay the same cold-start tax.
        """
        try:
            page = await ctx.new_page()
            try:
                page.on("response", _make_response_listener(
                    api_substr=self._api_substr,
                    captured=[],
                    bearer_box=bearer_box,
                ))
                await page.goto(
                    self._warmup_url,
                    timeout=self._nav_timeout_ms,
                    wait_until="domcontentloaded",
                )
                try:
                    await page.wait_for_load_state(
                        "networkidle",
                        timeout=self._nav_timeout_ms,
                    )
                except Exception:  # networkidle never settles on some pages
                    pass
            finally:
                await page.close()
        except Exception as exc:  # noqa: BLE001 - warm-up is best-effort
            logger.warning("warmup failed (continuing): %s", exc)

    async def _process_one(
        self,
        *,
        ctx: Any,
        row: dict[str, Any],
        out_f: Any,
        write_lock: asyncio.Lock,
        bearer_box: dict[str, str | None],
    ) -> None:
        """Fetch one detail page and write its row."""
        item_id = str(row["item_id"])
        scope = str(row["scope"])
        url = str(row["url"])

        # Belt-and-braces skip: the queue was filtered upstream but a
        # parallel worker may have just finished the same id during a
        # restart corner case. Idempotent + cheap.
        html_path = scope_html_dir(self.layout, scope) / f"{item_id}.html"
        if html_path.exists() and html_path.stat().st_size > 0:
            return

        captured: list[tuple[str, Any]] = []
        page = await ctx.new_page()
        try:
            page.on("response", _make_response_listener(
                api_substr=self._api_substr,
                captured=captured,
                bearer_box=bearer_box,
            ))
            try:
                await page.goto(
                    url,
                    timeout=self._nav_timeout_ms,
                    wait_until="domcontentloaded",
                )
            except Exception as exc:  # noqa: BLE001 - log + record
                await self._write_failed(
                    row=row, status="nav_failed", error=repr(exc),
                    out_f=out_f, write_lock=write_lock,
                )
                return
            # Spin until at least one /api/qtdc/... response was
            # captured OR the api_wait budget is spent. networkidle
            # alone is unreliable on this site because reCAPTCHA's
            # background polls keep the network busy.
            deadline = asyncio.get_event_loop().time() + self._api_wait_s
            while not captured:
                if asyncio.get_event_loop().time() >= deadline:
                    break
                await asyncio.sleep(0.5)
            # One more short settle so any sibling API responses
            # (related-file, preview-by-target) land before we
            # snapshot.
            try:
                await page.wait_for_load_state(
                    "networkidle",
                    timeout=5000,
                )
            except Exception:
                pass

            page_html = await page.content()
        finally:
            await page.close()

        # Persist raw artefacts before parsing so a parser bug doesn't
        # cost us the slow browser fetch.
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(page_html or "", encoding="utf-8")

        api_path = html_path.with_suffix(".api.json")
        api_path.write_text(
            json.dumps(
                {u: payload for u, payload in captured},
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

        rec = detail_record_from_api_json(
            item_id=item_id,
            scope=scope,
            source_url=url,
            api_responses=captured,
        )

        if self._download_files and rec.file_paths:
            await self._download_files_for(
                ctx=ctx,
                rec=rec,
                bearer=bearer_box.get("value"),
                api_substr=self._api_substr,
            )

        status = "ok" if (rec.body_text or rec.body_html or rec.title) else "empty"
        out_row = _record_to_jsonl_dict(
            rec=rec,
            row=row,
            status=status,
            error=None,
            run_id=self._run_id,
            html_path=html_path,
            host=self.layout.host,
        )
        async with write_lock:
            out_f.write(json.dumps(
                {k: out_row.get(k) for k in DETAIL_JSONL_FIELDS},
                ensure_ascii=False,
                default=_json_default,
            ))
            out_f.write("\n")
            out_f.flush()

    async def _download_files_for(
        self,
        *,
        ctx: Any,
        rec: DetailRecord,
        bearer: str | None,
        api_substr: str,
    ) -> None:
        """Fetch every attachment URL using the captured browser context.

        ``ctx.request`` shares cookies + the user agent with the
        page, but the API uses Authorization headers (not cookies)
        so we re-attach the captured Bearer explicitly. URLs in the
        API response are often relative to the gateway origin; we
        absolutise against the first captured api_url.
        """
        if not rec.file_paths:
            return
        api_origin = _origin_from_api_url(
            next(iter(rec.raw_api_json.keys()), None),
        )
        headers: dict[str, str] = {}
        if bearer:
            headers["authorization"] = bearer
        out_dir = scope_pdf_dir(self.layout, rec.scope)
        for i, fp in enumerate(rec.file_paths):
            full_url = _absolutise(fp.file_url, api_origin)
            ext = (fp.file_type or "bin").lower()
            base = f"{rec.item_id}" if i == 0 else f"{rec.item_id}-{i}"
            dest = out_dir / f"{base}.{ext}"
            if dest.exists() and dest.stat().st_size > 0:
                fp.local_path = str(dest.resolve())
                continue
            try:
                resp = await ctx.request.get(full_url, headers=headers)
                if resp.status != 200:
                    logger.warning(
                        "file fetch %d on %s (item=%d)",
                        resp.status, full_url, rec.item_id,
                    )
                    continue
                body = await resp.body()
                if not body:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(body)
                fp.local_path = str(dest.resolve())
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "file fetch crash on %s (item=%d): %s",
                    full_url, rec.item_id, exc,
                )

    async def _write_failed(
        self,
        *,
        row: dict[str, Any],
        status: str,
        error: str | None,
        out_f: Any,
        write_lock: asyncio.Lock,
    ) -> None:
        item_id = str(row["item_id"])
        scope = str(row["scope"])
        html_path = scope_html_dir(self.layout, scope) / f"{item_id}.html"
        rec = DetailRecord(
            item_id=item_id,
            scope=scope,
            source_url=str(row.get("url") or ""),
        )
        out_row = _record_to_jsonl_dict(
            rec=rec,
            row=row,
            status=status,
            error=error,
            run_id=self._run_id,
            html_path=html_path,
            host=self.layout.host,
        )
        async with write_lock:
            out_f.write(json.dumps(
                {k: out_row.get(k) for k in DETAIL_JSONL_FIELDS},
                ensure_ascii=False,
                default=_json_default,
            ))
            out_f.write("\n")
            out_f.flush()

    # ------------------------------------------------------ manifest

    def _write_manifest(self, *, out_path: Path, ok: int, err: int) -> None:
        path = self.layout.jsonl_dir / "manifest.json"
        payload = {
            "host": self.layout.host,
            "run_id": self._run_id,
            "completed_at": _utc_now_iso(),
            "items_ok": ok,
            "items_err": err,
            "docs_jsonl": str(out_path.resolve()),
            "sitemap_jsonl": str(
                (self.layout.jsonl_dir / "sitemap.jsonl").resolve(),
            ),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---- helpers -------------------------------------------------------------


class _Tally:
    __slots__ = ("ok", "err")

    def __init__(self) -> None:
        self.ok = 0
        self.err = 0


class _AsyncTokenBucket:
    """Process-global async rate limiter shared across workers."""

    def __init__(self, qps: float) -> None:
        if qps <= 0:
            raise ValueError("qps must be > 0")
        self._period = 1.0 / qps
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_at = max(now, self._next_at) + self._period


def _make_response_listener(
    *,
    api_substr: str,
    captured: list[tuple[str, Any]],
    bearer_box: dict[str, str | None],
) -> Any:
    """Build a closure suitable for ``page.on("response", ...)``.

    Side effects: appends ``(url, parsed_json)`` to ``captured`` for
    every JSON response under ``api_substr``, and stashes the request's
    ``Authorization`` header (if any) into ``bearer_box["value"]``.
    """

    async def _listener(resp: Any) -> None:
        try:
            url = str(resp.url)
        except Exception:  # noqa: BLE001
            return
        if api_substr not in url:
            return
        try:
            req = resp.request
            auth = None
            try:
                auth = await req.header_value("authorization")
            except Exception:  # noqa: BLE001
                auth = None
            if auth:
                bearer_box["value"] = auth
        except Exception:  # noqa: BLE001
            pass
        try:
            payload = await resp.json()
        except Exception:  # noqa: BLE001
            return
        captured.append((url, payload))

    return _listener


def _record_to_jsonl_dict(
    *,
    rec: DetailRecord,
    row: dict[str, Any],
    status: str,
    error: str | None,
    run_id: str,
    html_path: Path,
    host: str,
) -> dict[str, Any]:
    body_text = rec.body_text or ""
    return {
        "item_id": str(rec.item_id),
        "scope": rec.scope,
        "source": host,
        "source_url": rec.source_url,
        "api_url": rec.api_url,
        "scraped_at": _utc_now_iso(),
        "scrape_run_id": run_id,
        "doc_type": rec.doc_type,
        "so_hieu": rec.so_hieu,
        "ngay_ban_hanh": rec.ngay_ban_hanh,
        "co_quan_ban_hanh": rec.co_quan_ban_hanh,
        "trich_yeu": rec.trich_yeu,
        "title": rec.title or row.get("slug") or "",
        "body_html": rec.body_html,
        "body_text": body_text,
        "body_char_len": len(body_text),
        "body_text_hash": _sha256(body_text) if body_text else None,
        "file_paths": [asdict(fp) for fp in rec.file_paths],
        "html_path": str(html_path.resolve()),
        "fetch_status": status,
        "fetch_error": error,
    }


def _needs_fetch(row: dict[str, Any], layout: SiteLayout) -> bool:
    """Skip rows whose ``html/<scope>/<id>.html`` is already on disk."""
    try:
        scope = str(row["scope"])
        item_id = str(row["item_id"])
    except (KeyError, TypeError):
        return True
    if not item_id:
        return True
    cache = scope_html_dir(layout, scope) / f"{item_id}.html"
    return not (cache.exists() and cache.stat().st_size > 0)


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _autodetect_full_chromium() -> str:
    """Return the path to the full Chromium binary if one is present.

    ``playwright install chromium`` drops both the headless-shell and
    the full browser into ``~/.cache/ms-playwright/``. The default
    :meth:`launch(headless=True)` selects the headless-shell, which
    is markedly easier to fingerprint as a bot. We prefer the full
    binary -- combined with ``--headless=new`` it presents a
    real-Chrome rendering pipeline.

    Walks every ``chromium-*`` cache dir (newest first) and probes
    each known sub-layout. Returns ``""`` if nothing matches; the
    caller falls back to chrome-headless-shell with a warning.
    """
    cache_root = Path("~/.cache/ms-playwright").expanduser()
    if not cache_root.exists():
        return ""
    candidates: list[Path] = sorted(
        cache_root.glob("chromium-*"),
        key=lambda p: _version_key(p.name),
        reverse=True,
    )
    for child in candidates:
        for sub in _FULL_CHROMIUM_SUBPATHS:
            cand = child / sub
            if cand.exists():
                return str(cand)
    return ""


def _version_key(name: str) -> tuple[int, ...]:
    """Sort key for ``chromium-<n>`` cache dirs. Largest n wins."""
    try:
        return (int(name.rsplit("-", 1)[1]),)
    except (IndexError, ValueError):
        return (0,)


def _origin_from_api_url(api_url: str | None) -> str | None:
    if not api_url:
        return None
    try:
        # e.g. https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/...
        scheme_end = api_url.index("://")
        path_start = api_url.index("/", scheme_end + 3)
        return api_url[:path_start]
    except ValueError:
        return None


def _absolutise(url: str, base_origin: str | None) -> str:
    if url.startswith(("http://", "https://")):
        return url
    if base_origin:
        return urljoin(base_origin + "/", url.lstrip("/"))
    return url


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(o: Any) -> Any:
    if isinstance(o, FilePath):
        return asdict(o)
    raise TypeError(f"unencodable: {type(o)!r}")


__all__ = [
    "DEFAULT_API_URL_SUBSTR",
    "DEFAULT_USER_AGENT",
    "DEFAULT_WARMUP_URL",
    "VbplDetailDownloader",
]
