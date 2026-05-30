"""Detail-page downloader for the thuvienphapluat_banan corpus.

Reads ``jsonl/listings.jsonl`` produced by
:class:`packages.datasites.thuvienphapluat_banan.components.harvester.BananHarvester`
and, for every row, fetches the canonical detail page via the
``/banan/ban-an/x-<id>`` shortcut URL (the portal redirects it to the
slug-prefixed canonical URL, which is exactly what we want — no need
to embed the slug into the cache key).

On-disk artefacts per ``ban_an_id``:

* ``html/items/<ban_an_id>.html``  -- raw detail HTML (atomic write).
* one row appended to ``jsonl/docs.jsonl``        -- parsed sidebar +
  body, mirrors :data:`packages.datasites.thuvienphapluat_banan._shared.DETAIL_JSONL_FIELDS`.

Concurrency is a :class:`ThreadPoolExecutor` (``cfg.scraper.num_workers``)
sharing one rate-limited :class:`PoliteSession`; the bucket serialises
requests under the configured QPS so more workers hide per-request
latency without breaking the polite envelope. The 403 cool-down mirrors
the tnpl / harvester logic.

Resume model: any ``ban_an_id`` whose ``html/items/<id>.html`` is on
disk **and** whose row already lives in ``docs.jsonl`` is short-
circuited. The merged result is written atomically via ``.tmp`` rename
so a kill mid-fetch leaves the prior file untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from packages.common import PoliteSession, SiteLayout
from packages.common.http import session_from_scraper_cfg
from packages.datasites.thuvienphapluat_banan._shared import (
    DETAIL_JSONL_FIELDS,
    items_dir,
)
from packages.datasites.thuvienphapluat_banan.components.parser import (
    DetailRecord,
    parse_detail_page,
)

logger = logging.getLogger(__name__)


#: Slugless shortcut URL — the portal 302's to the canonical
#: ``/banan/ban-an/<slug>-<id>``. We use this so the cache + log
#: surfaces are keyed by integer id only and re-runs survive any slug
#: drift on the source portal.
DEFAULT_DETAIL_URL_TEMPLATE = (
    "https://thuvienphapluat.vn/banan/ban-an/x-{id}"
)


class BananDetailDownloader:
    """Fetch + parse one judgment detail per listings.jsonl row."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._url_template: str = str(
            cfg.scraper.get("detail_url_template", DEFAULT_DETAIL_URL_TEMPLATE)
        )
        self._num_workers: int = max(1, int(cfg.scraper.get("num_workers", 4)))
        self._cache_details: bool = bool(
            cfg.scraper.get("cache_details", True),
        )
        # See harvester.py for the cool-down policy notes.
        self._http_403_initial_delay_s: float = float(
            cfg.scraper.get("http_403_initial_delay_s", 60.0),
        )
        self._http_403_max_delay_s: float = float(
            cfg.scraper.get("http_403_max_delay_s", 600.0),
        )
        self._http_403_max_retries: int = int(
            cfg.scraper.get("http_403_max_retries", 5),
        )
        self._skip_finished_statuses: tuple[str, ...] = tuple(
            str(s) for s in (cfg.scraper.get("skip_finished_statuses") or [])
        )
        self._limit = cfg.get("limit", None)
        self._run_id = _make_run_id()
        self._session: PoliteSession | None = None

    # ---- public entrypoint ---------------------------------------------

    def run(self) -> Path:
        """Walk listings.jsonl + write docs.jsonl. Returns the output path."""
        listings_path = self.layout.jsonl_dir / "listings.jsonl"
        if not listings_path.exists():
            raise FileNotFoundError(
                f"{listings_path} missing; run --pipeline harvest first.",
            )
        items = list(_iter_jsonl(listings_path))

        out_path = self.layout.jsonl_dir / "docs.jsonl"
        existing_rows = _load_existing_docs(out_path)

        if self._skip_finished_statuses and existing_rows:
            before = len(items)
            items = [
                it for it in items
                if not _is_finished(
                    int(it["ban_an_id"]),
                    existing_rows,
                    self._skip_finished_statuses,
                )
            ]
            logger.info(
                "detail skip policy %s: carrying over %d finished rows, "
                "%d ids queued for fetch",
                list(self._skip_finished_statuses),
                before - len(items), len(items),
            )

        if self._limit is not None:
            items = items[: int(self._limit)]
        logger.info(
            "detail run: %d ids, workers=%d, run_id=%s",
            len(items), self._num_workers, self._run_id,
        )

        self._session = session_from_scraper_cfg(self.cfg)

        write_lock = threading.Lock()
        new_rows: dict[int, dict[str, Any]] = {}

        def _process(row: dict[str, Any]) -> dict[str, Any]:
            try:
                return self._fetch_and_parse(row)
            except Exception as exc:
                logger.exception(
                    "detail processing crashed: ban_an_id=%s",
                    row.get("ban_an_id"),
                )
                return _failed_record(
                    row, status=f"crash:{type(exc).__name__}",
                    error=repr(exc), run_id=self._run_id,
                    layout=self.layout,
                )

        ok = nf = err = 0
        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            futures = [pool.submit(_process, r) for r in items]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                with write_lock:
                    new_rows[int(rec["ban_an_id"])] = rec
                status = rec.get("fetch_status")
                if status == "ok":
                    ok += 1
                elif status == "not_found":
                    nf += 1
                else:
                    err += 1
                if i % 200 == 0:
                    logger.info(
                        "detail progress: %d/%d ok=%d not_found=%d err=%d",
                        i, len(items), ok, nf, err,
                    )
        logger.info(
            "detail fetch done: ok=%d not_found=%d err=%d (this run only)",
            ok, nf, err,
        )

        # Merge new rows over carried-over rows.
        merged: dict[int, dict[str, Any]] = {**existing_rows, **new_rows}
        _atomic_write_docs_jsonl(out_path, merged)

        counts = _status_counts(merged)
        logger.info(
            "docs.jsonl merged: total=%d ok=%d not_found=%d err=%d -> %s",
            len(merged), counts["ok"], counts["not_found"], counts["err"],
            out_path,
        )
        self._write_manifest(items_total=len(merged), **counts)
        return out_path

    # ---- per-item ------------------------------------------------------

    def _fetch_and_parse(self, row: dict[str, Any]) -> dict[str, Any]:
        ban_an_id = int(row["ban_an_id"])
        url_from_listing = str(row.get("url") or "")
        # We always fetch via the slugless shortcut so the cache key
        # stays integer-only; the listing URL is recorded as the
        # canonical ``source_url`` once the redirect resolves.
        fetch_url = self._url_template.format(id=ban_an_id)

        cache = items_dir(self.layout) / f"{ban_an_id}.html"
        html = ""
        final_url = url_from_listing
        if (
            self._cache_details
            and cache.exists()
            and cache.stat().st_size > 0
        ):
            html = cache.read_text(encoding="utf-8")
        else:
            assert self._session is not None
            resp = self._get_with_403_retry(fetch_url)
            if resp.status_code != 200:
                return _failed_record(
                    row, status=f"http_{resp.status_code}",
                    error=None, run_id=self._run_id,
                    layout=self.layout,
                )
            # Cloudflare redirects an unknown id to ``pagenotfound.htm``;
            # detect that and short-circuit as ``not_found`` so the row
            # is auditable without re-reading the cached HTML.
            final_url = str(resp.url) if resp.url else fetch_url
            if "pagenotfound" in final_url.lower():
                return _failed_record(
                    row, status="not_found",
                    error=None, run_id=self._run_id,
                    layout=self.layout,
                )
            html = resp.text or ""
            if self._cache_details:
                cache.parent.mkdir(parents=True, exist_ok=True)
                # Atomic write so a kill mid-write doesn't leave a
                # half-written cache file behind.
                tmp = cache.with_suffix(cache.suffix + ".tmp")
                tmp.write_text(html, encoding="utf-8")
                tmp.replace(cache)

        # We use the listing-row URL as the canonical source_url when
        # we have it (avoids encoding the slugless shortcut into
        # downstream parquet); fall back to the redirect-resolved URL
        # otherwise.
        source_url = url_from_listing or final_url
        parsed = parse_detail_page(html, source_url=source_url)
        if parsed is None:
            return _failed_record(
                row, status="empty_fragment", error=None,
                run_id=self._run_id, layout=self.layout,
            )
        return self._record_from_parsed(row=row, html_path=cache, parsed=parsed)

    def _record_from_parsed(
        self,
        *,
        row: dict[str, Any],
        html_path: Path,
        parsed: DetailRecord,
    ) -> dict[str, Any]:
        ban_an_id = int(row["ban_an_id"])
        body_text = (parsed.body_text or "").strip()
        slug = parsed.slug or row.get("slug") or None
        # Year / case_kind / procedure: prefer the listing-row hint
        # (parsed from the same doc_number) but fall back to the
        # detail-stage computation when the harvester didn't enrich.
        doc_number = parsed.doc_number or row.get("doc_number")
        case_kind, procedure = _split_case_kind_procedure(doc_number) if doc_number else (
            row.get("case_kind"), row.get("procedure"),
        )
        year = _year_from(doc_number=doc_number, issue_date=parsed.issue_date)
        return {
            "ban_an_id":      ban_an_id,
            "scope":          "banan",
            "source":         self.layout.host,
            "source_url":     parsed.source_url or row.get("url"),
            "slug":           slug,
            "scraped_at":     _utc_now_iso(),
            "scrape_run_id":  self._run_id,
            "title":          parsed.title or row.get("title"),
            "court":          parsed.court or row.get("court"),
            "doc_number":     doc_number,
            "trial_level":    parsed.trial_level,
            "legal_area":     parsed.legal_area,
            "case_kind":      case_kind,
            "procedure":      procedure,
            "year":           year,
            "issue_date_raw": parsed.issue_date_raw or row.get("issue_date"),
            "issue_date":     parsed.issue_date,
            "keywords":       list(parsed.keywords or []),
            "related_doc_ids": list(parsed.related_doc_ids or []),
            "body_html":      parsed.body_html,   # nulled by hf_export before publish
            "body_text":      body_text,
            "body_char_len":  len(body_text),
            "body_text_hash": _sha256(body_text) if body_text else None,
            "html_path":      str(html_path.resolve()),
            "fetch_status":   "ok",
            "fetch_error":    None,
        }

    # ---- HTTP ----------------------------------------------------------

    def _get_with_403_retry(self, url: str) -> requests.Response:
        """Issue the GET, sleeping through any Cloudflare 403 cool-down."""
        assert self._session is not None
        delay = self._http_403_initial_delay_s
        attempts = 0
        while True:
            resp = self._session.get(url, allow_redirects=True)
            if resp.status_code != 403:
                return resp
            if attempts >= self._http_403_max_retries:
                return resp
            attempts += 1
            logger.warning(
                "detail: 403 on %s; WAF cool-down %.0fs (attempt %d/%d)",
                url, delay, attempts, self._http_403_max_retries,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, self._http_403_max_delay_s)

    # ---- manifest ------------------------------------------------------

    def _write_manifest(
        self, *, items_total: int, ok: int, not_found: int, err: int,
    ) -> None:
        path = self.layout.jsonl_dir / "manifest.json"
        payload = {
            "host": self.layout.host,
            "run_id": self._run_id,
            "completed_at": _utc_now_iso(),
            "items_total": items_total,
            "items_ok": ok,
            "items_not_found": not_found,
            "items_err": err,
            "docs_jsonl": str((self.layout.jsonl_dir / "docs.jsonl").resolve()),
            "listings_jsonl": str(
                (self.layout.jsonl_dir / "listings.jsonl").resolve(),
            ),
            "taxonomy_json": str(
                (self.layout.jsonl_dir / "taxonomy.json").resolve(),
            ),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ----------------------------------------------------------------------
# Helpers (kept module-level so the thread-pool exception path doesn't
# drag the downloader's session over via closure capture).
# ----------------------------------------------------------------------


def _failed_record(
    row: dict[str, Any],
    *,
    status: str,
    error: str | None,
    run_id: str,
    layout: SiteLayout,
) -> dict[str, Any]:
    ban_an_id = int(row["ban_an_id"])
    return {
        "ban_an_id":      ban_an_id,
        "scope":          "banan",
        "source":         layout.host,
        "source_url":     row.get("url"),
        "slug":           row.get("slug"),
        "scraped_at":     _utc_now_iso(),
        "scrape_run_id":  run_id,
        "title":          row.get("title"),
        "court":          row.get("court"),
        "doc_number":     row.get("doc_number"),
        "trial_level":    None,
        "legal_area":     None,
        "case_kind":      row.get("case_kind"),
        "procedure":      row.get("procedure"),
        "year":           _year_from(doc_number=row.get("doc_number"), issue_date=None),
        "issue_date_raw": row.get("issue_date"),
        "issue_date":     None,
        "keywords":       [],
        "related_doc_ids": [],
        "body_html":      None,
        "body_text":      "",
        "body_char_len":  0,
        "body_text_hash": None,
        "html_path":      str((items_dir(layout) / f"{ban_an_id}.html").resolve()),
        "fetch_status":   status,
        "fetch_error":    error,
    }


def _load_existing_docs(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    out: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                bid = int(r["ban_an_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            out[bid] = r
    return out


def _is_finished(
    ban_an_id: int,
    existing: dict[int, dict[str, Any]],
    finished_prefixes: tuple[str, ...],
) -> bool:
    prev = existing.get(ban_an_id)
    if not prev:
        return False
    st = str(prev.get("fetch_status") or "")
    return any(st.startswith(p) for p in finished_prefixes)


def _atomic_write_docs_jsonl(
    out_path: Path,
    rows: dict[int, dict[str, Any]],
) -> None:
    """Write merged rows to ``out_path.tmp`` then rename. Crash-safe."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        for bid in sorted(rows):
            rec = rows[bid]
            f.write(
                json.dumps(
                    {k: rec.get(k) for k in DETAIL_JSONL_FIELDS},
                    ensure_ascii=False,
                )
            )
            f.write("\n")
    tmp_path.replace(out_path)


def _status_counts(rows: dict[int, dict[str, Any]]) -> dict[str, int]:
    ok = nf = err = 0
    for rec in rows.values():
        st = rec.get("fetch_status")
        if st == "ok":
            ok += 1
        elif st == "not_found":
            nf += 1
        else:
            err += 1
    return {"ok": ok, "not_found": nf, "err": err}


def _split_case_kind_procedure(doc_number: str | None) -> tuple[str | None, str | None]:
    if not doc_number:
        return None, None
    import re
    m = re.match(
        r"^\s*\d+/\d{4}/(?P<suffix>[A-Z\u0110\u0111][A-Z\u0110\u0111-]+)\s*$",
        doc_number,
    )
    if not m:
        return None, None
    suffix = m.group("suffix")
    if "-" in suffix:
        return suffix.split("-", 1)  # type: ignore[return-value]
    return suffix, None


def _year_from(*, doc_number: str | None, issue_date: str | None) -> int | None:
    """Best-effort year extraction: ISO date first, else doc_number ``/YYYY/``."""
    if issue_date:
        try:
            return int(issue_date[:4])
        except (ValueError, TypeError):
            pass
    if doc_number:
        import re
        m = re.search(r"/(\d{4})/", doc_number)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, TypeError):
                pass
    return None


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "DEFAULT_DETAIL_URL_TEMPLATE",
    "BananDetailDownloader",
]
