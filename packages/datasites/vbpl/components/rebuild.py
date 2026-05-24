"""Offline rebuild of ``jsonl/docs.jsonl`` from cached API JSON.

The detail stage drives Playwright against vbpl.vn to capture authenticated
``/api/qtdc/public/doc/...`` responses into ``html/<scope>/<id>.api.json``
sidecars. Those captures are durable (the slow part of the pipeline -- one
~90-hour browser run for 158K docs). When the *post-detail* mapping
(``detail_record_from_api_json``) is updated -- e.g. to pull richer
metadata like ``issue_date`` / ``issuing_authority`` / ``legal_type``
that earlier versions missed -- we want to re-derive ``docs.jsonl``
without paying the browser cost again.

That's what this component does. It walks every cached
``html/<scope>/<id>.api.json`` on disk, replays
:func:`detail_record_from_api_json` on the captured responses, and writes
a fresh ``jsonl/docs.jsonl`` in the canonical ``DETAIL_JSONL_FIELDS``
shape. The output is bit-for-bit equivalent to what a fresh ``detail``
run would produce against the same cached pages, minus the
``scraped_at`` / ``scrape_run_id`` columns (which we backfill from the
sitemap row).

The original ``docs.jsonl`` is left in place under
``jsonl/docs.jsonl.bak-<timestamp>`` for safety; the rebuild writes to
``jsonl/docs.jsonl.tmp`` and renames atomically.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.common import SiteLayout
from packages.datasites.vbpl._shared import (
    DETAIL_JSONL_FIELDS,
    SCOPES,
    build_layout,
    scope_html_dir,
)
from packages.datasites.vbpl.components.parser import (
    detail_record_from_api_json,
)

logger = logging.getLogger(__name__)


API_JSON_SUFFIX = ".api.json"


class VbplDetailRebuilder:
    """Rebuild ``jsonl/docs.jsonl`` from cached ``html/<scope>/*.api.json``.

    No browser, no network. Iterates the on-disk API JSON cache,
    re-applies :func:`detail_record_from_api_json`, and writes a fresh
    ``docs.jsonl``. Useful when the API → row mapping is updated and we
    need every downstream stage (parse, extract, embed, reduce) to see
    the richer record shape without paying the ~90-hour Playwright cost.
    """

    def __init__(self, cfg: Any, layout: SiteLayout | None = None) -> None:
        self.cfg = cfg
        self.layout = layout or build_layout(cfg)
        self._run_id = _make_run_id()
        self._host = str(self.cfg.host)

    # ------------------------------------------------------------------ run

    def run(self) -> Path:
        """Rebuild ``docs.jsonl`` from cached api.json. Returns its path."""
        out_path = self.layout.jsonl_dir / "docs.jsonl"
        tmp_path = out_path.with_suffix(".jsonl.tmp")

        sitemap_index = self._load_sitemap_index()

        ok = 0
        empty = 0
        skipped_no_json = 0
        with tmp_path.open("w", encoding="utf-8") as out_f:
            for scope in SCOPES:
                scope_dir = scope_html_dir(self.layout, scope)
                if not scope_dir.exists():
                    logger.info(
                        "rebuild_docs: %s/ not present; skipping", scope_dir,
                    )
                    continue
                json_files = sorted(scope_dir.glob(f"*{API_JSON_SUFFIX}"))
                logger.info(
                    "rebuild_docs: %s -> %d cached api.json files",
                    scope, len(json_files),
                )
                for i, p in enumerate(json_files, 1):
                    item_id = p.name.removesuffix(API_JSON_SUFFIX)
                    if not item_id:
                        skipped_no_json += 1
                        continue
                    sm_row = sitemap_index.get((scope, item_id), {})
                    row = _row_from_api_json(
                        api_json_path=p,
                        item_id=item_id,
                        scope=scope,
                        host=self._host,
                        sitemap_row=sm_row,
                        run_id=self._run_id,
                        layout=self.layout,
                    )
                    if row is None:
                        empty += 1
                        continue
                    out_f.write(
                        json.dumps(
                            {k: row.get(k) for k in DETAIL_JSONL_FIELDS},
                            ensure_ascii=False,
                            default=_json_default,
                        )
                    )
                    out_f.write("\n")
                    ok += 1
                    if i % 5000 == 0:
                        logger.info(
                            "rebuild_docs: %s progress %d/%d ok=%d empty=%d",
                            scope, i, len(json_files), ok, empty,
                        )

        if out_path.exists():
            backup = out_path.with_suffix(
                f".jsonl.bak-{_make_run_id()}"
            )
            shutil.move(str(out_path), str(backup))
            logger.info("rebuild_docs: existing docs.jsonl backed up to %s",
                        backup)
        shutil.move(str(tmp_path), str(out_path))

        self._write_manifest(ok=ok, empty=empty, skipped_no_json=skipped_no_json)
        logger.info(
            "rebuild_docs: done ok=%d empty=%d skipped=%d -> %s",
            ok, empty, skipped_no_json, out_path,
        )
        return out_path

    # ------------------------------------------------------------- helpers

    def _load_sitemap_index(self) -> dict[tuple[str, str], dict[str, Any]]:
        """Index ``sitemap.jsonl`` by ``(scope, item_id)`` for source_url lookup.

        Returns ``{}`` if the sitemap is missing -- the rebuild still
        works, just without ``source_url`` recovery (the api_url alone
        is enough for downstream parse/extract).
        """
        sm_path = self.layout.jsonl_dir / "sitemap.jsonl"
        index: dict[tuple[str, str], dict[str, Any]] = {}
        if not sm_path.exists():
            logger.warning(
                "rebuild_docs: sitemap.jsonl not found at %s; "
                "source_url will fall back to api_url",
                sm_path,
            )
            return index
        with sm_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                iid = str(row.get("item_id") or "")
                scope = str(row.get("scope") or "")
                if not iid or not scope:
                    continue
                index[(scope, iid)] = row
        logger.info("rebuild_docs: loaded %d sitemap entries", len(index))
        return index

    def _write_manifest(
        self, *, ok: int, empty: int, skipped_no_json: int,
    ) -> None:
        path = self.layout.jsonl_dir / "rebuild_docs_manifest.json"
        payload = {
            "host": self._host,
            "run_id": self._run_id,
            "completed_at": _utc_now_iso(),
            "rows_ok": ok,
            "rows_empty_api": empty,
            "skipped_no_item_id": skipped_no_json,
            "source": "html/<scope>/<id>.api.json",
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---- internal helpers ------------------------------------------------------


def _row_from_api_json(
    *,
    api_json_path: Path,
    item_id: str,
    scope: str,
    host: str,
    sitemap_row: dict[str, Any],
    run_id: str,
    layout: SiteLayout,
) -> dict[str, Any] | None:
    """Load one cached api.json, apply the parser, build a docs.jsonl row."""
    try:
        raw = json.loads(api_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("rebuild_docs: bad api.json %s: %s", api_json_path, exc)
        return None
    if not isinstance(raw, dict) or not raw:
        return None

    source_url = (
        str(sitemap_row.get("url"))
        if sitemap_row.get("url") else None
    )

    rec = detail_record_from_api_json(
        item_id=item_id,
        scope=scope,
        source_url=source_url or "",
        api_responses=list(raw.items()),
    )

    html_path = api_json_path.with_suffix("").with_suffix(".html")
    body_text = rec.body_text or ""
    status = "ok" if (body_text or rec.body_html or rec.title) else "empty"

    return {
        "item_id": str(rec.item_id),
        "scope": rec.scope,
        "source": host,
        "source_url": rec.source_url or None,
        "api_url": rec.api_url,
        "scraped_at": sitemap_row.get("harvested_at") or _utc_now_iso(),
        "scrape_run_id": sitemap_row.get("scrape_run_id") or run_id,
        "doc_type": rec.doc_type,
        "legal_type": rec.legal_type,
        "legal_area": rec.legal_area,
        "doc_number": rec.doc_number,
        "issue_date": rec.issue_date,
        "issuing_authority": rec.issuing_authority,
        "summary": rec.summary,
        "title": rec.title or sitemap_row.get("slug") or "",
        "body_html": rec.body_html,
        "body_text": body_text,
        "body_char_len": len(body_text),
        "body_text_hash": _sha256(body_text) if body_text else None,
        "file_paths": [asdict(fp) for fp in rec.file_paths],
        "html_path": str(html_path.resolve()),
        "fetch_status": status,
        "fetch_error": None,
    }


def _make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _sha256(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(f"unserialisable {type(o).__name__}")


__all__ = ["VbplDetailRebuilder"]
