"""Stage-3 parser for thuvienphapluat_banan: ``docs.jsonl`` -> markdown.

Reads :file:`jsonl/docs.jsonl` produced by
:class:`packages.datasites.thuvienphapluat_banan.components.downloader.BananDetailDownloader`
and emits one ``md/<ban_an_id>.md`` per judgment plus a sibling
``<ban_an_id>.meta.json`` so the downstream Curator extract stage can
rehydrate the sidebar metadata without re-loading every input.

The body source priority is intentionally simple compared to vbpl
(which juggles a PDF / API-body / Next.js-shell fallback chain):

1. ``body_html`` field from ``docs.jsonl`` (the active tab's HTML) →
   converted to markdown via :mod:`markdownify`. This covers ~100% of
   thuvienphapluat_banan rows since the portal renders the full
   judgment text inline (no PDF attachment, no SPA).
2. ``body_text`` field as the plain-text fallback if ``body_html`` is
   empty/malformed (e.g. WAF served a stripped page).

The parser is in-process (no Ray) to stay symmetric with the harvest
+ detail stages. Concurrency comes from a :class:`ThreadPoolExecutor`
sharing the same :mod:`markdownify` import; the per-doc work is
~100 µs CPU-bound, so 4-8 workers saturate the disk reads.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.common import SiteLayout

logger = logging.getLogger(__name__)


class BananDocumentParser:
    """In-process HTML → markdown writer over ``docs.jsonl``."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._num_workers: int = max(
            1, int(cfg.parser.get("num_workers", 4)),
        )
        self._min_body_chars: int = int(
            cfg.parser.get("min_body_chars", 50),
        )
        self._force: bool = bool(
            cfg.parser.get("force", False)
            or cfg.parser.get("refresh_meta", False)
        )
        self._limit = cfg.get("limit", None)
        self._run_id = _make_run_id()
        # markdownify is shipped as a soft dep; the lock + cached import
        # avoid threadlocal re-imports on every call.
        self._import_lock = threading.Lock()
        self._markdownify = None  # type: ignore[var-annotated]

    # ---- public entrypoint ---------------------------------------------

    def run(self) -> Path:
        """Walk docs.jsonl, write per-doc markdown. Returns ``md_dir``."""
        docs_path = self.layout.jsonl_dir / "docs.jsonl"
        if not docs_path.exists():
            raise FileNotFoundError(
                f"{docs_path} missing; run --pipeline detail first.",
            )
        rows = list(_iter_jsonl(docs_path))
        if self._limit is not None:
            rows = rows[: int(self._limit)]
        if self._force:
            rows_to_parse = list(rows)
            skipped = 0
        else:
            rows_to_parse = [r for r in rows if _needs_parse(r, self.layout)]
            skipped = len(rows) - len(rows_to_parse)
        logger.info(
            "parse run: %d docs in scope; skip-existing=%d; "
            "to parse=%d; workers=%d; force=%s; run_id=%s",
            len(rows), skipped, len(rows_to_parse),
            self._num_workers, self._force, self._run_id,
        )
        if not rows_to_parse:
            logger.info("nothing to parse; --limit slice fully covered on disk")
            return self.layout.md_dir

        ok = err = 0
        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            futures = [pool.submit(self._parse_one, r) for r in rows_to_parse]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    success = fut.result()
                except Exception:
                    logger.exception("parse worker crashed")
                    err += 1
                    continue
                if success:
                    ok += 1
                else:
                    err += 1
                if i % 200 == 0:
                    logger.info(
                        "parse progress: %d/%d ok=%d err=%d",
                        i, len(rows_to_parse), ok, err,
                    )
        logger.info(
            "parse run done: ok=%d err=%d -> %s",
            ok, err, self.layout.md_dir,
        )
        self._write_manifest(parsed=ok, errors=err)
        return self.layout.md_dir

    # ---- per-item ------------------------------------------------------

    def _parse_one(self, row: dict[str, Any]) -> bool:
        try:
            ban_an_id = int(row["ban_an_id"])
        except (KeyError, TypeError, ValueError):
            logger.warning("skipping malformed row: %r", row)
            return False

        markdown, source_kind = self._parse_body(row)
        markdown = (markdown or "").strip()

        md_path = self.layout.md_dir / f"{ban_an_id}.md"
        meta_path = md_path.with_suffix(".meta.json")
        md_path.parent.mkdir(parents=True, exist_ok=True)

        md_path.write_text(markdown, encoding="utf-8")
        meta = self._build_meta(
            row=row,
            ban_an_id=ban_an_id,
            md_path=md_path,
            markdown=markdown,
            source_kind=source_kind,
        )
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return bool(markdown) and len(markdown) >= self._min_body_chars

    def _parse_body(self, row: dict[str, Any]) -> tuple[str, str]:
        body_html = (row.get("body_html") or "").strip()
        if body_html:
            md = self._html_to_markdown(body_html)
            if md.strip():
                return md, "body_html"

        body_text = (row.get("body_text") or "").strip()
        if body_text:
            return body_text, "body_text"

        return "", "empty"

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to markdown; collapse blank-line runs > 2."""
        if not html:
            return ""
        md_fn = self._get_markdownify()
        if md_fn is None:
            # Last-ditch: strip tags via bs4.
            try:
                from bs4 import BeautifulSoup
                return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            except ImportError:
                return html
        try:
            md = md_fn(
                html,
                heading_style="ATX",
                strip=["script", "style"],
            )
        except Exception as exc:
            logger.warning("markdownify failed: %s", exc)
            try:
                from bs4 import BeautifulSoup
                return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
            except ImportError:
                return ""

        out_lines: list[str] = []
        blanks = 0
        for line in md.splitlines():
            if not line.strip():
                blanks += 1
                if blanks <= 2:
                    out_lines.append("")
                continue
            blanks = 0
            out_lines.append(line.rstrip())
        return "\n".join(out_lines).strip()

    def _get_markdownify(self):
        if self._markdownify is not None:
            return self._markdownify
        with self._import_lock:
            if self._markdownify is None:
                try:
                    from markdownify import markdownify as _md
                    self._markdownify = _md
                except ImportError:
                    self._markdownify = None
        return self._markdownify

    # ---- writers -------------------------------------------------------

    def _build_meta(
        self,
        *,
        row: dict[str, Any],
        ban_an_id: int,
        md_path: Path,
        markdown: str,
        source_kind: str,
    ) -> dict[str, Any]:
        """Sidecar meta written next to ``<ban_an_id>.md``."""
        return {
            "doc_name":       str(ban_an_id),
            "ban_an_id":      ban_an_id,
            "scope":          "banan",
            "source":         str(self.cfg.host),
            "source_url":     row.get("source_url"),
            "title":          row.get("title"),
            "court":          row.get("court"),
            "doc_number":     row.get("doc_number"),
            "trial_level":    row.get("trial_level"),
            "legal_area":     row.get("legal_area"),
            "case_kind":      row.get("case_kind"),
            "procedure":      row.get("procedure"),
            "year":           row.get("year"),
            "issue_date":     row.get("issue_date"),
            "keywords":       row.get("keywords") or [],
            "related_doc_ids": row.get("related_doc_ids") or [],
            "html_path":      row.get("html_path"),
            "md_path":        str(md_path.resolve()),
            "char_len":       len(markdown),
            "body_source":    source_kind,
            "parser_model":   "local/markdownify",
            "parser_runtime": "local",
            "num_pages":      None,
            "confidence":     None,
            "parsed_at":      _utc_now_iso(),
            "scrape_run_id":  row.get("scrape_run_id"),
            "parse_run_id":   self._run_id,
        }

    def _write_manifest(self, *, parsed: int, errors: int) -> None:
        path = self.layout.jsonl_dir / "parse_manifest.json"
        payload = {
            "host": str(self.cfg.host),
            "run_id": self._run_id,
            "completed_at": _utc_now_iso(),
            "parser_runtime": "local/markdownify",
            "items_ok": parsed,
            "items_err": errors,
            "md_dir": str(self.layout.md_dir.resolve()),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---- helpers -------------------------------------------------------------


def _needs_parse(row: dict[str, Any], layout: SiteLayout) -> bool:
    """Skip rows whose ``md/<ban_an_id>.md`` is already on disk."""
    try:
        ban_an_id = int(row["ban_an_id"])
    except (KeyError, TypeError, ValueError):
        return True
    md_path = layout.md_dir / f"{ban_an_id}.md"
    return not md_path.exists()


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _make_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = ["BananDocumentParser"]
