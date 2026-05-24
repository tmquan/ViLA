"""Detail-page downloader for pbgdpl Q&A.

Given the ``listings.jsonl`` produced by :class:`PbgdplHarvester`, this
module fetches each ItemID's detail HTML fragment from
``/SMPT_Publishing_UC/HoiDapPL/frmDSCauHoi.aspx?ItemID=<id>``, parses
it, and writes one consolidated record per question to
``jsonl/qa.jsonl``.

The on-disk HTML cache (``html/items/<item_id>.html``) makes the
fetcher idempotent: re-running skips IDs that already have a non-empty
cached fragment, which means a partial run is cheap to resume after a
network blip. Records that fail to parse are still written to the
JSONL with ``fetch_status`` set so analysts can audit gaps.

Concurrency is provided by a small thread pool (``cfg.scraper.num_workers``)
sharing one :class:`PoliteSession`. The session's :class:`TokenBucket`
serialises requests under the configured QPS so adding more workers
hides per-request latency without breaking the rate-limit envelope.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from packages.common import PoliteSession, SiteLayout
from packages.common.http import session_from_scraper_cfg
from packages.datasites.pbgdpl._shared import (
    DETAIL_JSONL_FIELDS,
    items_dir,
)
from packages.datasites.pbgdpl.components.harvester import DEFAULT_LISTING_URL
from packages.datasites.pbgdpl.components.parser import (
    DetailRecord,
    parse_detail_fragment,
)

logger = logging.getLogger(__name__)


_WORD_RE = re.compile(r"\S+", re.UNICODE)


class PbgdplDetailDownloader:
    """Fetch + parse one Q&A detail per ItemID, accumulate into qa.jsonl."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._listing_url: str = str(
            cfg.scraper.get("listing_url", DEFAULT_LISTING_URL)
        )
        self._num_workers: int = max(1, int(cfg.scraper.get("num_workers", 4)))
        self._cache_details: bool = bool(
            cfg.scraper.get("cache_details", True),
        )
        self._limit = cfg.get("limit", None)
        self._run_id = _make_run_id()
        self._session: PoliteSession | None = None

    # ---- public entrypoint ----------------------------------------------

    def run(self) -> Path:
        """Walk listings.jsonl + write qa.jsonl. Returns the output path."""
        listings_path = self.layout.jsonl_dir / "listings.jsonl"
        if not listings_path.exists():
            raise FileNotFoundError(
                f"{listings_path} missing; run --pipeline harvest first.",
            )
        items = list(_iter_listing_rows(listings_path))
        if self._limit is not None:
            items = items[: int(self._limit)]
        logger.info(
            "detail run: %d items, workers=%d, run_id=%s",
            len(items), self._num_workers, self._run_id,
        )

        self._session = session_from_scraper_cfg(self.cfg)

        out_path = self.layout.jsonl_dir / "qa.jsonl"
        write_lock = threading.Lock()

        def _process(row: dict[str, Any]) -> dict[str, Any] | None:
            try:
                return self._fetch_and_parse(row)
            except Exception as exc:
                logger.exception(
                    "detail processing crashed: item_id=%s",
                    row.get("item_id"),
                )
                return _failed_record(
                    row, status=f"crash:{type(exc).__name__}",
                    error=repr(exc), run_id=self._run_id, layout=self.layout,
                )

        ok = err = 0
        with out_path.open("w", encoding="utf-8") as out_f, ThreadPoolExecutor(
            max_workers=self._num_workers,
        ) as pool:
            futures = [pool.submit(_process, r) for r in items]
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                if rec is None:
                    err += 1
                    continue
                with write_lock:
                    out_f.write(
                        json.dumps(
                            {k: rec.get(k) for k in DETAIL_JSONL_FIELDS},
                            ensure_ascii=False,
                        )
                    )
                    out_f.write("\n")
                if rec.get("fetch_status") == "ok":
                    ok += 1
                else:
                    err += 1
                if i % 100 == 0:
                    logger.info(
                        "detail progress: %d/%d ok=%d err=%d",
                        i, len(items), ok, err,
                    )
        logger.info("detail run done: ok=%d err=%d -> %s", ok, err, out_path)
        self._write_manifest(items_total=len(items), ok=ok, err=err)
        return out_path

    # ---- per-item -------------------------------------------------------

    def _fetch_and_parse(self, row: dict[str, Any]) -> dict[str, Any]:
        """Return a fully-populated detail-row dict."""
        item_id = int(row["item_id"])
        url = f"{self._listing_url}?{urlencode({'ItemID': str(item_id)})}"
        cache = items_dir(self.layout) / f"{item_id}.html"

        html = ""
        if (
            self._cache_details
            and cache.exists()
            and cache.stat().st_size > 0
        ):
            html = cache.read_text(encoding="utf-8")
        else:
            assert self._session is not None
            resp = self._session.get(url)
            if resp.status_code != 200:
                return _failed_record(
                    row, status=f"http_{resp.status_code}",
                    error=None, run_id=self._run_id, layout=self.layout,
                )
            html = resp.text or ""
            if self._cache_details:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(html, encoding="utf-8")

        parsed = parse_detail_fragment(html)
        if parsed is None:
            return _failed_record(
                row, status="empty_fragment",
                error=None, run_id=self._run_id, layout=self.layout,
                source_url=url,
            )
        return _record_from_parsed(
            row=row,
            url=url,
            parsed=parsed,
            run_id=self._run_id,
            layout=self.layout,
        )

    # ---- manifest -------------------------------------------------------

    def _write_manifest(self, *, items_total: int, ok: int, err: int) -> None:
        path = self.layout.jsonl_dir / "manifest.json"
        payload = {
            "host": self.layout.host,
            "run_id": self._run_id,
            "completed_at": _utc_now_iso(),
            "items_total": items_total,
            "items_ok": ok,
            "items_err": err,
            "qa_jsonl": str((self.layout.jsonl_dir / "qa.jsonl").resolve()),
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


# ---- record assembly -----------------------------------------------------


def _record_from_parsed(
    *,
    row: dict[str, Any],
    url: str,
    parsed: DetailRecord,
    run_id: str,
    layout: SiteLayout,
) -> dict[str, Any]:
    item_id = int(row["item_id"])
    cache_path = items_dir(layout) / f"{item_id}.html"
    answer_text = parsed.answer_text or ""
    question_text = parsed.question_text or ""
    return {
        "item_id": item_id,
        "source": layout.host,
        "source_url": url,
        "scraped_at": _utc_now_iso(),
        "scrape_run_id": run_id,
        "listing_page": row.get("listing_page"),
        "listing_position": row.get("listing_position"),
        "is_featured": bool(row.get("is_featured")),
        "title_listing": row.get("title_listing") or "",
        "question_summary_listing": row.get("question_summary_listing") or "",
        "lv_ids": list(row.get("lv_ids") or []),
        "lv_names": list(row.get("lv_names") or []),
        "title": parsed.title,
        "question_html": parsed.question_html,
        "question_text": question_text,
        "answer_html": parsed.answer_html,
        "answer_text": answer_text,
        "date_sent_raw": parsed.date_sent_raw,
        "date_sent": parsed.date_sent,
        "sender_name": parsed.sender_name,
        "disclaimer": parsed.disclaimer,
        "question_char_len": len(question_text),
        "answer_char_len": len(answer_text),
        "question_word_count": len(_WORD_RE.findall(question_text)),
        "answer_word_count": len(_WORD_RE.findall(answer_text)),
        "answer_text_hash": _sha256(answer_text) if answer_text else None,
        "html_path": str(cache_path.resolve()),
        "fetch_status": "ok",
        "fetch_error": None,
    }


def _failed_record(
    row: dict[str, Any],
    *,
    status: str,
    error: str | None,
    run_id: str,
    layout: SiteLayout,
    source_url: str | None = None,
) -> dict[str, Any]:
    item_id = int(row["item_id"])
    return {
        "item_id": item_id,
        "source": layout.host,
        "source_url": source_url or "",
        "scraped_at": _utc_now_iso(),
        "scrape_run_id": run_id,
        "listing_page": row.get("listing_page"),
        "listing_position": row.get("listing_position"),
        "is_featured": bool(row.get("is_featured")),
        "title_listing": row.get("title_listing") or "",
        "question_summary_listing": row.get("question_summary_listing") or "",
        "lv_ids": list(row.get("lv_ids") or []),
        "lv_names": list(row.get("lv_names") or []),
        "title": "",
        "question_html": "",
        "question_text": "",
        "answer_html": "",
        "answer_text": "",
        "date_sent_raw": None,
        "date_sent": None,
        "sender_name": None,
        "disclaimer": None,
        "question_char_len": 0,
        "answer_char_len": 0,
        "question_word_count": 0,
        "answer_word_count": 0,
        "answer_text_hash": None,
        "html_path": str((items_dir(layout) / f"{item_id}.html").resolve()),
        "fetch_status": status,
        "fetch_error": error,
    }


# ---- helpers -------------------------------------------------------------


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


__all__ = ["PbgdplDetailDownloader"]
