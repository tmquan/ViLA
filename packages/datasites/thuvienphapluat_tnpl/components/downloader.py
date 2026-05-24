"""Detail-page downloader for the tnpl term corpus.

Given ``listings.jsonl`` (one stub row per probe id), this module
fetches ``https://thuvienphapluat.vn/tnpl/{id}/x?tab=0`` for every
id, parses the detail HTML, and writes one row per id to
``jsonl/terms.jsonl``.

On-disk cache (``html/items/<id>.html``) makes the fetcher idempotent
so a partial run resumes cheaply. Soft-404 ids (HTTP 200 with
``Không tìm thấy thuật ngữ này``) are still written to the JSONL with
``fetch_status="not_found"`` and zeroed content -- the dataset can be
audited from the JSONL alone without re-walking the cache.

Concurrency is a thread pool (``cfg.scraper.num_workers``) sharing one
rate-limited :class:`PoliteSession`; the bucket serialises requests
under the configured QPS so more workers hide per-request latency
without breaking the polite envelope.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from packages.common import PoliteSession, SiteLayout
from packages.common.http import session_from_scraper_cfg
from packages.datasites.thuvienphapluat_tnpl._shared import (
    DETAIL_JSONL_FIELDS,
    items_dir,
)
from packages.datasites.thuvienphapluat_tnpl.components.harvester import (
    DEFAULT_DETAIL_URL_TEMPLATE,
)
from packages.datasites.thuvienphapluat_tnpl.components.parser import (
    DetailRecord,
    parse_detail_fragment,
)

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"\S+", re.UNICODE)
#: Two ways the source advertises "this id does not exist": the
#: slugless URL emits ``Không tìm thấy ngữ thuật này``; the slugged
#: variant emits ``Không tìm thấy thuật ngữ này`` (sic, the two words
#: are swapped). We match both spellings.
_NOT_FOUND_SENTINELS: tuple[str, ...] = (
    "Không tìm thấy thuật ngữ này",
    "Không tìm thấy ngữ thuật này",
)
#: When the URL slug is present but the id is invalid the server falls
#: back to the homepage listing block instead of the proper not-found
#: page. The valid-detail surface anchors on ``<div id="Tab1"`` and
#: never contains ``divTNPL`` (the homepage list container). We use
#: the presence of ``divTNPL`` without ``#Tab1`` as the homepage-
#: fallback soft-404 signal.
_TAB1_MARKER = '<div id="Tab1"'
_HOMEPAGE_LIST_MARKER = "divTNPL"


class TnplDetailDownloader:
    """Fetch + parse one term detail per probe id, accumulate to terms.jsonl."""

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
        self._retry_statuses: tuple[str, ...] = tuple(
            str(s) for s in (cfg.scraper.get("retry_statuses") or [])
        )
        self._skip_finished_statuses: tuple[str, ...] = tuple(
            str(s) for s in (cfg.scraper.get("skip_finished_statuses") or [])
        )
        # HTTP 403 retry policy. The shared PoliteSession backs off on
        # 429/5xx but NOT on 403, because most sites use 403 as a
        # terminal authz failure. thuvienphapluat.vn instead returns
        # 403 from its Cloudflare WAF when our IP gets rate-bucketed --
        # those bans typically lift after 5-15 minutes, so a long flat
        # cool-down here turns the WAF wall into a transient stall
        # instead of thousands of permanently-tagged 403 rows.
        self._http_403_initial_delay_s: float = float(
            cfg.scraper.get("http_403_initial_delay_s", 60.0),
        )
        self._http_403_max_delay_s: float = float(
            cfg.scraper.get("http_403_max_delay_s", 600.0),
        )
        self._http_403_max_retries: int = int(
            cfg.scraper.get("http_403_max_retries", 5),
        )
        self._limit = cfg.get("limit", None)
        self._run_id = _make_run_id()
        self._session: PoliteSession | None = None

        # Build LinhVuc name -> id lookup from taxonomy.json so the
        # row's ``lĩnh_vực_id`` is resolved without a homepage re-fetch.
        self._lv_name_to_id: dict[str, int] = self._load_lv_name_to_id()

    # ---- public entrypoint ----------------------------------------------

    def run(self) -> Path:
        """Walk listings.jsonl + write terms.jsonl. Returns the output path.

        When ``scraper.skip_finished_statuses`` is set the run is
        incremental: existing rows in ``terms.jsonl`` whose status
        matches the policy are carried over verbatim, ids whose prior
        status does NOT match are re-fetched, and the merged result is
        written atomically via a ``.tmp`` rename at the end. This
        makes resumes cheap (skip the 13k cached rows entirely) and
        crash-safe (a kill mid-fetch leaves the prior terms.jsonl
        untouched).
        """
        listings_path = self.layout.jsonl_dir / "listings.jsonl"
        if not listings_path.exists():
            raise FileNotFoundError(
                f"{listings_path} missing; run --pipeline harvest first.",
            )
        items = list(_iter_listing_rows(listings_path))

        out_path = self.layout.jsonl_dir / "terms.jsonl"
        existing_rows = _load_existing_terms(out_path)

        if self._skip_finished_statuses and existing_rows:
            before = len(items)
            items = [
                it for it in items
                if not _is_finished(
                    int(it["term_id"]),
                    existing_rows,
                    self._skip_finished_statuses,
                )
            ]
            logger.info(
                "skip policy %s: carrying over %d finished rows, "
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

        if self._retry_statuses:
            invalidated = self._invalidate_cache_for_retry(items)
            logger.info(
                "retry policy %s: invalidated %d cached HTML files",
                list(self._retry_statuses), invalidated,
            )

        self._session = session_from_scraper_cfg(self.cfg)

        write_lock = threading.Lock()
        new_rows: dict[int, dict[str, Any]] = {}

        def _process(row: dict[str, Any]) -> dict[str, Any]:
            try:
                return self._fetch_and_parse(row)
            except Exception as exc:
                logger.exception(
                    "detail processing crashed: term_id=%s",
                    row.get("term_id"),
                )
                return _failed_record(
                    row, status=f"crash:{type(exc).__name__}",
                    error=repr(exc), run_id=self._run_id,
                    layout=self.layout, url=self._url_for(row["term_id"]),
                )

        ok = nf = err = 0
        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            futures = [pool.submit(_process, r) for r in items]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                with write_lock:
                    new_rows[int(rec["term_id"])] = rec
                status = rec.get("fetch_status")
                if status == "ok":
                    ok += 1
                elif status == "not_found":
                    nf += 1
                else:
                    err += 1
                if i % 500 == 0:
                    logger.info(
                        "detail progress: %d/%d ok=%d not_found=%d err=%d",
                        i, len(items), ok, nf, err,
                    )
        logger.info(
            "detail fetch done: ok=%d not_found=%d err=%d (this run only)",
            ok, nf, err,
        )

        # Merge: new rows win over carried-over rows for the same id.
        merged: dict[int, dict[str, Any]] = {**existing_rows, **new_rows}
        _atomic_write_terms_jsonl(out_path, merged)

        final_counts = _status_counts(merged)
        logger.info(
            "terms.jsonl merged: total=%d ok=%d not_found=%d err=%d -> %s",
            len(merged), final_counts["ok"], final_counts["not_found"],
            final_counts["err"], out_path,
        )
        self._write_manifest(
            items_total=len(merged),
            ok=final_counts["ok"],
            not_found=final_counts["not_found"],
            err=final_counts["err"],
        )
        return out_path

    # ---- per-item -------------------------------------------------------

    def _fetch_and_parse(self, row: dict[str, Any]) -> dict[str, Any]:
        term_id = int(row["term_id"])
        url = self._url_for(term_id)
        cache = items_dir(self.layout) / f"{term_id}.html"

        html = ""
        if (
            self._cache_details
            and cache.exists()
            and cache.stat().st_size > 0
        ):
            html = cache.read_text(encoding="utf-8")
        else:
            assert self._session is not None
            resp = self._get_with_403_retry(url)
            if resp.status_code != 200:
                return _failed_record(
                    row, status=f"http_{resp.status_code}",
                    error=None, run_id=self._run_id,
                    layout=self.layout, url=url,
                )
            html = resp.text or ""
            if self._cache_details:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(html, encoding="utf-8")

        # Soft-404 short-circuit. Two surfaces of the same "id does
        # not exist" condition: an explicit Vietnamese sentinel (with
        # one of two word-orders the source uses), or a silent
        # fall-through to the homepage list block (no #Tab1 block,
        # but the homepage list container is present). Tagging both
        # as ``not_found`` keeps the dataset auditable from the JSONL
        # alone; ``empty_fragment`` is reserved for unexpected cases
        # where #Tab1 is present but renders empty.
        if any(s in html for s in _NOT_FOUND_SENTINELS):
            return _failed_record(
                row, status="not_found", error=None,
                run_id=self._run_id, layout=self.layout, url=url,
            )
        if _TAB1_MARKER not in html and _HOMEPAGE_LIST_MARKER in html:
            return _failed_record(
                row, status="not_found", error=None,
                run_id=self._run_id, layout=self.layout, url=url,
            )

        parsed = parse_detail_fragment(html)
        if parsed is None:
            return _failed_record(
                row, status="empty_fragment", error=None,
                run_id=self._run_id, layout=self.layout, url=url,
            )
        return self._record_from_parsed(row=row, url=url, parsed=parsed)

    def _get_with_403_retry(self, url: str) -> requests.Response:
        """Issue the GET, sleeping through any Cloudflare 403 cool-down.

        ``PoliteSession`` already handles 429 / 5xx with exponential
        backoff, but 403 is treated as terminal. Cloudflare's WAF on
        thuvienphapluat.vn returns 403 to soft-banned IPs for ~5-15
        minutes at a time, so we add a flat-then-doubling cool-down
        capped at ``http_403_max_delay_s``. Returns the final
        response (still 403 if the wall outlasts our patience).
        """
        assert self._session is not None
        delay = self._http_403_initial_delay_s
        attempts = 0
        while True:
            resp = self._session.get(url)
            if resp.status_code != 403:
                return resp
            if attempts >= self._http_403_max_retries:
                return resp
            attempts += 1
            logger.warning(
                "403 on %s; WAF cool-down %.0fs (attempt %d/%d)",
                url, delay, attempts, self._http_403_max_retries,
            )
            time.sleep(delay)
            delay = min(delay * 2.0, self._http_403_max_delay_s)

    def _record_from_parsed(
        self,
        *,
        row: dict[str, Any],
        url: str,
        parsed: DetailRecord,
    ) -> dict[str, Any]:
        term_id = int(row["term_id"])
        cache_path = items_dir(self.layout) / f"{term_id}.html"
        defn_text = parsed.định_nghĩa or ""
        slug = _slug_from_url(url) or ""
        return {
            "term_id": term_id,
            "source": self.layout.host,
            "source_url": url,
            "slug": slug,
            "scraped_at": _utc_now_iso(),
            "scrape_run_id": self._run_id,
            "term_name_vi":                parsed.tên_thuật_ngữ,
            "term_name_en_native":         parsed.tên_thuật_ngữ_gốc_tiếng_anh,
            "definition_vi":               defn_text,
            "area_name_vi":                parsed.lĩnh_vực,
            "area_id":                     self._lv_name_to_id.get(parsed.lĩnh_vực or ""),
            "status_vi":                   parsed.tình_trạng,
            "updated_by_vi":               parsed.cập_nhật_bởi,
            "updated_at_raw":              parsed.cập_nhật_lúc_gốc,
            "updated_at":                  parsed.cập_nhật_lúc,
            "related_term_ids":            list(parsed.thuật_ngữ_liên_quan_ids),
            "related_term_names_vi":       list(parsed.thuật_ngữ_liên_quan),
            "definition_char_len":         len(defn_text),
            "definition_word_count":       len(_WORD_RE.findall(defn_text)),
            "definition_hash":             _sha256(defn_text) if defn_text else None,
            "html_path":                   str(cache_path.resolve()),
            "fetch_status":                "ok",
            "fetch_error":                 None,
        }

    # ---- helpers --------------------------------------------------------

    def _url_for(self, term_id: int) -> str:
        return self._url_template.format(id=term_id)

    def _invalidate_cache_for_retry(
        self, items: list[dict[str, Any]],
    ) -> int:
        """Delete cached HTML for ids whose prior status matches retry_statuses.

        Reads the previous ``terms.jsonl`` (if present) to build an
        ``id -> fetch_status`` map, then deletes the on-disk HTML
        cache file for every id whose status starts with one of the
        configured ``retry_statuses`` prefixes. The fetch loop then
        treats those ids as never-fetched and re-pulls them fresh.
        """
        prior_path = self.layout.jsonl_dir / "terms.jsonl"
        if not prior_path.exists() or prior_path.stat().st_size == 0:
            return 0
        prior_status: dict[int, str] = {}
        with prior_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    tid = int(r.get("term_id"))
                except (TypeError, ValueError):
                    continue
                st = r.get("fetch_status") or ""
                if tid and st:
                    prior_status[tid] = st
        if not prior_status:
            return 0
        item_ids = {int(r["term_id"]) for r in items}
        n = 0
        for tid, st in prior_status.items():
            if tid not in item_ids:
                continue
            if not any(st.startswith(prefix) for prefix in self._retry_statuses):
                continue
            cache = items_dir(self.layout) / f"{tid}.html"
            try:
                cache.unlink(missing_ok=True)
                n += 1
            except OSError as exc:
                logger.warning(
                    "could not unlink cache for term_id=%d: %s", tid, exc,
                )
        return n

    def _load_lv_name_to_id(self) -> dict[str, int]:
        path = self.layout.jsonl_dir / "taxonomy.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        out: dict[str, int] = {}
        for entry in payload.get("lĩnh_vực", []) or []:
            name = entry.get("ten") or entry.get("name")
            lv_id = entry.get("id")
            if isinstance(name, str) and isinstance(lv_id, int):
                out[name] = lv_id
        return out

    # ---- manifest -------------------------------------------------------

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
            "terms_jsonl": str((self.layout.jsonl_dir / "terms.jsonl").resolve()),
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
# Failed-record builder (kept outside the class so it's pickleable for
# the thread pool's exception path without dragging the downloader's
# session over).
# ----------------------------------------------------------------------


def _failed_record(
    row: dict[str, Any],
    *,
    status: str,
    error: str | None,
    run_id: str,
    layout: SiteLayout,
    url: str,
) -> dict[str, Any]:
    term_id = int(row["term_id"])
    return {
        "term_id":                     term_id,
        "source":                      layout.host,
        "source_url":                  url,
        "slug":                        _slug_from_url(url) or "",
        "scraped_at":                  _utc_now_iso(),
        "scrape_run_id":               run_id,
        "term_name_vi":                "",
        "term_name_en_native":         None,
        "definition_vi":               "",
        "area_name_vi":                None,
        "area_id":                     None,
        "status_vi":                   None,
        "updated_by_vi":               None,
        "updated_at_raw":              None,
        "updated_at":                  None,
        "related_term_ids":            [],
        "related_term_names_vi":       [],
        "definition_char_len":         0,
        "definition_word_count":       0,
        "definition_hash":             None,
        "html_path":                   str(
            (items_dir(layout) / f"{term_id}.html").resolve(),
        ),
        "fetch_status":                status,
        "fetch_error":                 error,
    }


# ---- helpers -------------------------------------------------------------


def _load_existing_terms(path: Path) -> dict[int, dict[str, Any]]:
    """Read ``terms.jsonl`` into ``{term_id: row}`` (skip malformed lines)."""
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
                tid = int(r["term_id"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            out[tid] = r
    return out


def _is_finished(
    term_id: int,
    existing: dict[int, dict[str, Any]],
    finished_prefixes: tuple[str, ...],
) -> bool:
    prev = existing.get(term_id)
    if not prev:
        return False
    st = str(prev.get("fetch_status") or "")
    return any(st.startswith(p) for p in finished_prefixes)


def _atomic_write_terms_jsonl(
    out_path: Path,
    rows: dict[int, dict[str, Any]],
) -> None:
    """Write merged rows to ``out_path.tmp`` then rename. Crash-safe."""
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as f:
        for tid in sorted(rows):
            rec = rows[tid]
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


def _slug_from_url(url: str) -> str | None:
    """Pull the slug between ``/tnpl/{id}/`` and ``?`` from the URL."""
    m = re.match(r".*?/tnpl/\d+/([^?]+)", url or "")
    return m.group(1) if m else None


def _iter_listing_rows(path: Path):
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


__all__ = ["TnplDetailDownloader"]
