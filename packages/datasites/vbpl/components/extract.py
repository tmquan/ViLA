"""LEGACY in-process extractor for vbpl (retired May 2026).

.. deprecated:: replaced by the Curator pipeline in
   :mod:`packages.datasites.vbpl.extract` (wiki/DATASITES.md §3.5 +
   ``cfg.extractor.normalizers``). This module is no longer wired
   into ``--pipeline extract`` / ``--pipeline all`` -- the CLI now
   builds a :class:`nemo_curator.pipeline.Pipeline` that runs the
   declarative ``NormalizerChainStage`` -> ``LegalExtractStage`` ->
   ``JsonlPerDocWriter`` chain and post-coalesces the per-doc JSONL
   into ``parquet/extract/extract-NNNNN-of-KKKKK.parquet`` shards.

   Kept in-tree as a reference implementation of the per-record
   pipeline (helpful for debugging a single document outside Ray):
   the public class :class:`VbplDocumentExtractor` can still be
   instantiated by ad-hoc scripts. New code SHOULD use the Curator
   factory; this module will be deleted in a follow-up sweep once
   no internal tooling depends on it.

Reads the per-document outputs of
:class:`packages.datasites.vbpl.components.parse.VbplDocumentParser`
(``md/<scope>/<id>.md`` + sibling ``<id>.meta.json``) and emits one
``jsonl/extract.jsonl`` row per document with the canonical fields
documented in :data:`packages.datasites.vbpl._shared.EXTRACTOR_JSONL_FIELDS`.

Three layers run on every document, mirroring the
:class:`packages.extractor.stage.LegalExtractStage` contract used by
anle / congbobanan but in-process (no Curator / Ray):

1. :func:`packages.extractor.normalization.normalize_text` -- NFC +
   Vietnamese tone-mark canonicalization (post-1984 orthography) +
   PDF whitespace cleanup. Toggle with
   ``cfg.extractor.run_text_normalization``.
2. :class:`packages.extractor.generic.GenericExtractor` -- regex /
   dictionary NER + statute linker. Toggle with
   ``cfg.extractor.run_generic_layer``.
3. :class:`packages.extractor.structure.LegalStructureExtractor` --
   hierarchical document representation (DocumentMeta + Section +
   Paragraph + Sentence). Toggle with
   ``cfg.extractor.run_structure_layer``.

The Vietnamese precedent normalizer (layer 2 in the legal extract
stage) is **off by default** for vbpl because the corpus is
statutes / regulations / circulars, not án lệ -- the fields it
emits (``precedent_number``, ``adopted_date``, ``applied_article``,
``principle_text``) don't apply. Override
``cfg.extractor.run_site_layer=true`` if you have your own use for
those columns; the value will be carried through unchanged.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from packages.common import SiteLayout
from packages.datasites.vbpl._shared import (
    EXTRACTOR_JSONL_FIELDS,
    SCOPES,
    scope_md_dir,
)
from packages.extractor.generic import GenericExtractor
from packages.extractor.normalization import normalize_text
from packages.extractor.structure import LegalStructureExtractor

logger = logging.getLogger(__name__)


class VbplDocumentExtractor:
    """In-process extractor: ``md/<scope>/*.md`` -> ``jsonl/extract.jsonl``."""

    def __init__(self, cfg: Any, layout: SiteLayout) -> None:
        self.cfg = cfg
        self.layout = layout
        self._num_workers: int = max(
            1, int(cfg.extractor.get("num_workers", 4)),
        )
        self._run_id = _make_run_id()
        self._limit = cfg.get("limit", None)

        # Toggles. ``run_site_layer`` defaults to False for vbpl since
        # the precedent normalizer targets án lệ, not statutes.
        self._run_normalize: bool = bool(
            cfg.extractor.get("run_text_normalization", True),
        )
        self._run_generic: bool = bool(
            cfg.extractor.get("run_generic_layer", True),
        )
        self._run_structure: bool = bool(
            cfg.extractor.get("run_structure_layer", True),
        )

        # Shared algorithms; both implementations are stateless across
        # calls (read-only regex tables) so one instance per process
        # is enough. Setup is cheap.
        self._generic: GenericExtractor | None = None
        self._structure: LegalStructureExtractor | None = None
        self._setup_lock = threading.Lock()

    # ------------------------------------------------------ public

    def run(self) -> Path:
        """Walk md/<scope>/*.md, write one jsonl/extract.jsonl row per item."""
        md_paths = list(_iter_md_paths(self.layout.md_dir))
        if self._limit is not None:
            md_paths = md_paths[: int(self._limit)]
        out_path = self.layout.jsonl_dir / "extract.jsonl"
        if not md_paths:
            logger.warning(
                "no markdown found at %s; run --pipeline parse first.",
                self.layout.md_dir,
            )
            out_path.write_text("", encoding="utf-8")
            return out_path

        logger.info(
            "extract run: %d markdown files; workers=%d run_id=%s "
            "(normalize=%s generic=%s structure=%s)",
            len(md_paths), self._num_workers, self._run_id,
            self._run_normalize, self._run_generic, self._run_structure,
        )
        self._setup()

        write_lock = threading.Lock()
        ok = err = 0
        with out_path.open("w", encoding="utf-8") as out_f, ThreadPoolExecutor(
            max_workers=self._num_workers,
        ) as pool:
            futures = [
                pool.submit(self._extract_one, mp) for mp in md_paths
            ]
            for i, fut in enumerate(as_completed(futures), 1):
                try:
                    row = fut.result()
                except Exception:  # noqa: BLE001
                    logger.exception("extract worker crashed")
                    err += 1
                    continue
                if row is None:
                    err += 1
                    continue
                with write_lock:
                    out_f.write(json.dumps(
                        {k: row.get(k) for k in EXTRACTOR_JSONL_FIELDS},
                        ensure_ascii=False,
                        default=_json_default,
                    ))
                    out_f.write("\n")
                ok += 1
                if i % 200 == 0:
                    logger.info(
                        "extract progress: %d/%d ok=%d err=%d",
                        i, len(md_paths), ok, err,
                    )
        logger.info(
            "extract run done: ok=%d err=%d -> %s", ok, err, out_path,
        )
        self._write_manifest(rows=ok, errors=err, out_path=out_path)
        return out_path

    # ------------------------------------------------------ per-item

    def _extract_one(self, md_path: Path) -> dict[str, Any] | None:
        """Read md + sibling meta.json, run the three layers, return one row."""
        try:
            markdown = md_path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("md read failed %s: %s", md_path, exc)
            return None
        meta_path = md_path.with_suffix(".meta.json")
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                logger.warning("meta read failed %s: %s", meta_path, exc)
        if self._run_normalize and markdown:
            markdown = normalize_text(markdown)

        item_id = str(meta.get("item_id") or md_path.stem)
        scope = str(meta.get("scope") or _scope_from_md_path(md_path) or "")

        # Layer 1: generic extractor (regex NER + statute linker).
        # Always runs to populate `text_hash` + `char_len` even when
        # the entity / relation lists are empty per cfg.
        assert self._generic is not None
        generic = self._generic.extract(doc_id=item_id, markdown=markdown)
        if self._run_generic:
            extracted_payload = generic.to_jsonable()
        else:
            extracted_payload = {
                "doc_id": item_id,
                "text_hash": generic.text_hash,
                "char_len": generic.char_len,
                "entities": [],
                "relations": [],
                "statute_refs": [],
            }

        # Layer 2: legal structure extractor (sections / paragraphs /
        # sentences). Picks up sidebar metadata via scraper_metadata so
        # the doc-meta header survives even when the parser output is
        # noisy.
        structure_payload: dict[str, Any] | None = None
        if self._run_structure:
            assert self._structure is not None
            scraper_meta = _scraper_meta_from(meta)
            structure = self._structure.extract(
                doc_id=item_id,
                markdown=markdown,
                scraper_metadata=scraper_meta,
            )
            structure_payload = structure.to_jsonable()

        return {
            "doc_name": item_id,
            "item_id": item_id,
            "scope": scope,
            "source": str(self.cfg.host),
            "source_url": meta.get("source_url"),
            "api_url": meta.get("api_url"),
            "html_path": meta.get("html_path"),
            "md_path": str(md_path.resolve()),
            "file_paths": meta.get("file_paths") or [],
            "markdown": markdown,
            "num_pages": meta.get("num_pages"),
            "confidence": meta.get("confidence"),
            "parser_model": meta.get("parser_model"),
            "parser_runtime": meta.get("parser_runtime"),
            "body_source": meta.get("body_source"),
            "parsed_at": meta.get("parsed_at"),
            "text_hash": generic.text_hash,
            "char_len": generic.char_len,
            "extracted": extracted_payload,
            "structure": structure_payload,
            "title": meta.get("title"),
            "doc_type": meta.get("doc_type"),
            "legal_type": meta.get("legal_type"),
            "legal_area": meta.get("legal_area"),
            "doc_number": meta.get("doc_number"),
            "issue_date": meta.get("issue_date"),
            "issuing_authority": meta.get("issuing_authority"),
            "summary": meta.get("summary"),
            "scrape_run_id": meta.get("scrape_run_id"),
            "parse_run_id": meta.get("parse_run_id"),
            "extract_run_id": self._run_id,
            "extracted_at": _utc_now_iso(),
        }

    # ------------------------------------------------------ setup / writers

    def _setup(self) -> None:
        with self._setup_lock:
            if self._generic is None:
                self._generic = GenericExtractor()
            if self._structure is None and self._run_structure:
                self._structure = LegalStructureExtractor()

    def _write_manifest(
        self, *, rows: int, errors: int, out_path: Path,
    ) -> None:
        path = self.layout.jsonl_dir / "extract_manifest.json"
        payload = {
            "host": str(self.cfg.host),
            "run_id": self._run_id,
            "completed_at": _utc_now_iso(),
            "rows_ok": rows,
            "rows_err": errors,
            "extract_jsonl": str(out_path.resolve()),
            "layers": {
                "normalize": self._run_normalize,
                "generic": self._run_generic,
                "structure": self._run_structure,
            },
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


# ---- helpers -------------------------------------------------------------


def _iter_md_paths(md_root: Path):
    """Yield every per-scope ``<id>.md`` under ``md/<scope>/``."""
    for scope in SCOPES:
        scope_dir = md_root / scope
        if not scope_dir.exists():
            continue
        for p in sorted(scope_dir.iterdir()):
            if p.suffix == ".md" and p.is_file():
                yield p


def _scope_from_md_path(p: Path) -> str | None:
    """Recover ``trung_uong`` / ``dia_phuong`` from ``md/<scope>/<id>.md``."""
    parent = p.parent.name
    return parent if parent in SCOPES else None


def _scraper_meta_from(meta: dict[str, Any]) -> dict[str, Any]:
    """Subset of meta that the LegalStructureExtractor consults.

    The structure extractor accepts arbitrary keys but reads
    ``title`` / ``doc_type`` / ``adopted_date`` / etc. for header-meta
    hints. We forward both the canonical keys and vbpl-specific
    aliases so the extractor's lookup logic finds something useful.
    """
    out: dict[str, Any] = {}
    for k in (
        "title",
        "doc_type",
        "legal_type",
        "legal_area",
        "doc_number",
        "issue_date",
        "issuing_authority",
        "summary",
        "source_url",
    ):
        v = meta.get(k)
        if v is not None:
            out[k] = v
    # Aliases the upstream extractor knows about (anle precedent
    # vocabulary). Cheap to forward; ignored if the layer doesn't
    # use them.
    if meta.get("issue_date"):
        out.setdefault("adopted_date", meta["issue_date"])
    return out


def _make_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_default(o: Any) -> Any:
    if hasattr(o, "to_jsonable"):
        return o.to_jsonable()
    if hasattr(o, "__dict__"):
        return o.__dict__
    raise TypeError(f"unencodable: {type(o)!r}")


__all__ = ["VbplDocumentExtractor"]
