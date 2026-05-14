"""Stage-2 parser for vbpl: downloaded artifacts -> markdown on disk.

Reads the on-disk artefacts produced by
:class:`packages.datasites.vbpl.components.detail.VbplDetailDownloader`
(``jsonl/docs.jsonl`` + ``pdf/<scope>/<id>.{pdf,doc,docx}`` +
``html/<scope>/<id>.html`` + ``html/<scope>/<id>.api.json``) and emits
one ``md/<scope>/<id>.md`` per document plus a sibling
``<id>.meta.json`` so the downstream extractor can rehydrate the
provenance + sitemap fields without re-loading every input.

For each item the parser picks a body source in priority order:

1. **Downloaded file** (``pdf_path``) -- run through
   :class:`packages.parser.pypdf.PypdfParser` (or
   :class:`HybridParser` when ``cfg.parser.runtime`` is ``hybrid`` /
   ``nim`` and ``NVIDIA_API_KEY`` is set). Handles ``.pdf``, ``.docx``,
   ``.doc`` (legacy Word via antiword/catdoc/soffice fallbacks) and
   any other binary the downloader fetched.
2. **Captured API body HTML** (``body_html`` field on the
   ``docs.jsonl`` row) -- converted to markdown via
   :mod:`markdownify`. This is the path used when reCAPTCHA passed
   and the SPA delivered the body inline but without an attachment.
3. **Rendered Next.js shell** (``html/<scope>/<id>.html``) -- a last-
   ditch fallback that produces little useful text on most rows
   because the SPA leaves the body to client-side fetches; recorded
   only so downstream consumers can audit the gap.

The parser is in-process (no Ray) to stay symmetric with the harvest
+ detail stages. Concurrency comes from a :class:`ThreadPoolExecutor`
sharing one :class:`ParserAlgorithm`; pypdf releases the GIL during
PDF parsing and the ``.doc`` subprocess paths are I/O-bound, so
adding workers scales close to linearly until disk reads saturate.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.common import SiteLayout
from packages.datasites.vbpl._shared import (
    SCOPES,
    scope_html_dir,
    scope_md_dir,
)
from packages.parser.base import ParserAlgorithm
from packages.parser.pypdf import PypdfParser

logger = logging.getLogger(__name__)


class VbplDocumentParser:
    """In-process parser: docs.jsonl + on-disk artefacts -> markdown files."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._num_workers: int = max(
            1, int(cfg.parser.get("num_workers", 4)),
        )
        self._runtime: str = str(cfg.parser.get("runtime", "local")).lower()
        self._min_local_chars: int = int(
            cfg.parser.get("min_local_chars", 50),
        )
        self._limit = cfg.get("limit", None)
        self._run_id = _make_run_id()
        # Built lazily inside :meth:`run` so a missing ``NVIDIA_API_KEY``
        # only fails when the user actually requested ``runtime=nim``.
        self._algo: ParserAlgorithm | None = None
        # Thread-local guard around the pdf / docx / doc backends.
        # PypdfParser is technically thread-safe but the underlying
        # pypdf reader keeps a per-call cursor; simpler to lock.
        self._algo_lock = threading.Lock()

    # ------------------------------------------------------ public

    def run(self) -> Path:
        """Walk docs.jsonl + cached files, write per-item markdown.

        Returns the markdown root dir. Per-item outputs:

        * ``md/<scope>/<item_id>.md``
        * ``md/<scope>/<item_id>.meta.json``
        """
        docs_path = self.layout.jsonl_dir / "docs.jsonl"
        if not docs_path.exists():
            raise FileNotFoundError(
                f"{docs_path} missing; run --pipeline detail first.",
            )
        rows = list(_iter_jsonl(docs_path))
        if self._limit is not None:
            rows = rows[: int(self._limit)]
        rows_to_parse = [r for r in rows if _needs_parse(r, self.layout)]
        skipped = len(rows) - len(rows_to_parse)
        logger.info(
            "parse run: %d/%d docs in scope; skip-existing=%d; "
            "to parse=%d; runtime=%s; workers=%d; run_id=%s",
            len(rows), len(rows), skipped, len(rows_to_parse),
            self._runtime, self._num_workers, self._run_id,
        )
        if not rows_to_parse:
            logger.info("nothing to parse; --limit slice fully covered on disk")
            return self.layout.md_dir

        self._algo = self._build_algo()

        ok = err = 0
        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            futures = [pool.submit(self._parse_one, r) for r in rows_to_parse]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    success = fut.result()
                except Exception:  # noqa: BLE001 - logged, counted
                    logger.exception("parse worker crashed")
                    err += 1
                    continue
                if success:
                    ok += 1
                else:
                    err += 1
                if i % 100 == 0:
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

    # ------------------------------------------------------ per-item

    def _parse_one(self, row: dict[str, Any]) -> bool:
        """Parse one ``docs.jsonl`` row. Returns True iff non-empty markdown."""
        try:
            item_id = str(row["item_id"])
            scope = str(row["scope"])
        except (KeyError, TypeError):
            logger.warning("skipping malformed row: %r", row)
            return False
        if scope not in SCOPES:
            logger.warning(
                "skipping row with unknown scope: item=%s scope=%s",
                item_id, scope,
            )
            return False

        markdown, source_kind, parser_model, num_pages, confidence = (
            self._parse_body(row, item_id, scope)
        )
        markdown = (markdown or "").strip()

        md_path = scope_md_dir(self.layout, scope) / f"{item_id}.md"
        meta_path = md_path.with_suffix(".meta.json")
        md_path.parent.mkdir(parents=True, exist_ok=True)

        md_path.write_text(markdown, encoding="utf-8")
        meta = self._build_meta(
            row=row,
            item_id=item_id,
            scope=scope,
            md_path=md_path,
            markdown=markdown,
            source_kind=source_kind,
            parser_model=parser_model,
            num_pages=num_pages,
            confidence=confidence,
        )
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return bool(markdown)

    def _parse_body(
        self,
        row: dict[str, Any],
        item_id: str,
        scope: str,
    ) -> tuple[str, str, str, int | None, float | None]:
        """Pick a body source and parse it. Returns (md, kind, model, npages, conf)."""
        # 1. Prefer a downloaded file when ``download_files=true`` ran
        #    successfully on the detail stage.
        file_paths = row.get("file_paths") or []
        if isinstance(file_paths, list):
            for fp in file_paths:
                local = (
                    fp.get("local_path") if isinstance(fp, dict) else None
                )
                if local and os.path.exists(local):
                    md, model, n_pages, conf = self._parse_file(local)
                    if md:
                        return md, "file", model, n_pages, conf

        # 2. Fall back to the body_html captured from the API.
        body_html = str(row.get("body_html") or "").strip()
        if body_html:
            md = _html_to_markdown(body_html)
            if md.strip():
                return md, "body_html", "local/markdownify", None, None

        # 3. Last resort: parse the rendered Next.js shell. This
        #    produces little signal on most vbpl pages because the
        #    SPA leaves the body to a client-side fetch; we still try
        #    so downstream consumers can audit the gap.
        shell_path = scope_html_dir(self.layout, scope) / f"{item_id}.html"
        if shell_path.exists():
            try:
                shell_html = shell_path.read_text(encoding="utf-8")
            except OSError:
                shell_html = ""
            if shell_html:
                shell_md = _html_to_markdown(shell_html)
                if len(shell_md) >= self._min_local_chars:
                    return shell_md, "shell_html", "local/markdownify", None, None

        return "", "empty", "", None, None

    def _parse_file(
        self, path: str,
    ) -> tuple[str, str, int | None, float | None]:
        """Run the configured ParserAlgorithm over a local file."""
        try:
            data = Path(path).read_bytes()
        except OSError as exc:
            logger.warning("file read failed for %s: %s", path, exc)
            return "", "", None, None
        if not data:
            return "", "", None, None
        assert self._algo is not None
        try:
            with self._algo_lock:
                result = self._algo.parse(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("parse failed for %s: %s", path, exc)
            return "", "", None, None
        md = str(result.get("markdown") or "").strip()
        n_pages = len(result.get("pages") or []) or None
        conf = result.get("confidence")
        model = str(getattr(self._algo, "model_id", "") or self._runtime)
        return md, model, n_pages, conf

    def _build_algo(self) -> ParserAlgorithm:
        """Construct the configured parser. Lazy NIM client to keep the cold-start cheap."""
        if self._runtime == "local":
            return PypdfParser()
        if self._runtime in ("nim", "hybrid"):
            try:
                from packages.parser.nemotron import NemoretrieverParser
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NIM parser unavailable (%s); downgrading to local",
                    exc,
                )
                return PypdfParser()
            api_key = (
                os.environ.get("NVIDIA_API_KEY")
                or os.environ.get("NVIDIA_NIM_API_KEY")
            )
            if not api_key:
                logger.warning(
                    "NVIDIA_API_KEY not set; downgrading parser.runtime=%s "
                    "to local",
                    self._runtime,
                )
                return PypdfParser()
            base_url = str(self.cfg.parser.nim_base_url)
            if base_url.startswith("${") and base_url.endswith("}"):
                from packages.parser.nemotron import (
                    DEFAULT_BASE_URL as _DEF_URL,
                )
                base_url = _DEF_URL
            nim = NemoretrieverParser(
                api_key=api_key,
                base_url=base_url,
                model=str(self.cfg.parser.model_id),
                timeout=float(self.cfg.parser.timeout_s),
                dpi=int(self.cfg.parser.get("nim_dpi", 150)),
                tool=str(self.cfg.parser.get("nim_tool", "markdown_bbox")),
            )
            if self._runtime == "nim":
                return nim
            from packages.parser.hybrid import HybridParser
            return HybridParser(
                local=PypdfParser(),
                nim=nim,
                min_chars=self._min_local_chars,
            )
        logger.warning(
            "unknown parser.runtime=%r; using local pypdf",
            self._runtime,
        )
        return PypdfParser()

    # ------------------------------------------------------ writers

    def _build_meta(
        self,
        *,
        row: dict[str, Any],
        item_id: str,
        scope: str,
        md_path: Path,
        markdown: str,
        source_kind: str,
        parser_model: str,
        num_pages: int | None,
        confidence: float | None,
    ) -> dict[str, Any]:
        """Sidecar meta written next to ``<id>.md``.

        Carries every field the extractor needs to emit a complete
        JSONL row without re-reading ``docs.jsonl`` per item.
        """
        return {
            "doc_name": item_id,
            "item_id": item_id,
            "scope": scope,
            "source": str(self.cfg.host),
            "source_url": row.get("source_url"),
            "api_url": row.get("api_url"),
            "doc_type": row.get("doc_type"),
            "so_hieu": row.get("so_hieu"),
            "ngay_ban_hanh": row.get("ngay_ban_hanh"),
            "co_quan_ban_hanh": row.get("co_quan_ban_hanh"),
            "trich_yeu": row.get("trich_yeu"),
            "title": row.get("title"),
            "file_paths": row.get("file_paths") or [],
            "html_path": row.get("html_path"),
            "md_path": str(md_path.resolve()),
            "char_len": len(markdown),
            "body_source": source_kind,
            "parser_model": parser_model,
            "parser_runtime": self._runtime,
            "num_pages": num_pages,
            "confidence": confidence,
            "parsed_at": _utc_now_iso(),
            "scrape_run_id": row.get("scrape_run_id"),
            "parse_run_id": self._run_id,
        }

    def _write_manifest(self, *, parsed: int, errors: int) -> None:
        path = self.layout.jsonl_dir / "parse_manifest.json"
        payload = {
            "host": str(self.cfg.host),
            "run_id": self._run_id,
            "completed_at": _utc_now_iso(),
            "parser_runtime": self._runtime,
            "items_ok": parsed,
            "items_err": errors,
            "md_dir": str(self.layout.md_dir.resolve()),
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---- helpers -------------------------------------------------------------


def _html_to_markdown(html: str) -> str:
    """Convert HTML to markdown. Falls back to bs4 text extraction."""
    if not html:
        return ""
    try:
        from markdownify import markdownify
    except ImportError:
        # markdownify is the documented dep; fall back to bs4 plain
        # text so a missing install doesn't kill the run.
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return html
        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        return text
    try:
        md = markdownify(
            html,
            heading_style="ATX",
            strip=["script", "style"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("markdownify failed: %s", exc)
        try:
            from bs4 import BeautifulSoup
            return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
        except ImportError:
            return ""
    # markdownify can leave runs of blank lines; collapse > 2 to 2.
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


def _needs_parse(row: dict[str, Any], layout: SiteLayout) -> bool:
    """Skip rows whose ``md/<scope>/<id>.md`` is already on disk."""
    try:
        scope = str(row["scope"])
        item_id = str(row["item_id"])
    except (KeyError, TypeError):
        return True
    if not item_id or scope not in SCOPES:
        return True
    md_path = scope_md_dir(layout, scope) / f"{item_id}.md"
    return not md_path.exists()


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = ["VbplDocumentParser"]
