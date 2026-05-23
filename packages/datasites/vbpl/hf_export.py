"""Materialise the vbpl corpus as a HuggingFace-ready dataset folder.

Reads the extractor raw per-doc JSONL tier
(``data/<host>/jsonl/<doc_name>.jsonl``, one record per file, per
wiki/DATASITES.md §3.5) and the optional reducer parquets from
``data/<host>/parquet/reduced/*``, writing a self-contained ``hf/``
tree that can be uploaded with
:mod:`packages.datasites.vbpl.push_to_hf`::

    data/vbpl.vn/hf/
        README.md            # Vietnamese / English dataset card
        documents.parquet    # one row per document, with structure
        manifest.json        # corpus roll-up consumed by the card
        embedding-<facet>-<dim>.png   # 8 PNG scatter plots (4 facets x 2 dims)

Schema
------

The parquet is a flat table over the corpus with three families of
columns:

* **Identification + meta** -- ``doc_name`` (= ``item_id``), ``scope``
  (``trung_uong`` / ``dia_phuong``), ``source_url``, ``api_url``,
  ``title``, ``doc_type``, ``doc_number`` (document number),
  ``issue_date`` (issue date), ``year``, ``issuing_body``
  (issuing agency), ``summary`` (summary). All flat, queryable
  without parsing JSON.
* **Body + stats** -- ``markdown`` (NFC-normalised, Vietnamese tone
  canonicalised), ``num_pages``, ``num_sections``, ``num_paragraphs``,
  ``num_sentences``, ``char_len``, ``text_hash``, ``parser_model``,
  ``parser_runtime``, ``body_source``, ``parsed_at``.
* **Hierarchy + entities** -- ``structure_json`` (DocumentMeta +
  Section + Paragraph + Sentence) and ``extracted_json`` (entities,
  relations, statute_refs) carried as JSON strings.

The ``structure_json`` / ``extracted_json`` columns are JSON strings
rather than native pyarrow structs because the inner lists have
unbounded length and slightly varying field sets per document; JSON
strings round-trip cleanly through pandas + parquet without forcing
a Procrustean nested schema.

Empty rows (``markdown==""`` -- typically docs whose detail fetch
landed ``fetch_status="empty"`` because reCAPTCHA blocked the body
call) are dropped from the parquet so the public corpus only carries
documents that actually have content. The full row count including
empties is preserved in ``manifest.json`` for audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pyarrow as pa

from packages.common.hf import iter_jsonl
from packages.datasites.vbpl.codes import (
    CANONICAL_CODE_TO_NAME,
    CANONICAL_CODE_TO_SLUG,
    SLUG_TO_CANONICAL_CODE,
    UNCATEGORISED_AREA,
    canonical_code,
    code_from_slug,
    doc_type_slug,
    legal_area_label,
    legal_type_name,
)

logger = logging.getLogger(__name__)

DEFAULT_JSONL_DIR = Path("data/vbpl.vn/jsonl")
DEFAULT_REDUCED_DIR = Path("data/vbpl.vn/parquet/reduced")
DEFAULT_OUT_DIR = Path("data/vbpl.vn/hf")

#: Stage / sidecar filenames that live under the raw per-doc JSONL
#: directory but are NOT document rows. The extractor pipeline
#: shipped some of these historically (single-file ``extract.jsonl``,
#: manifests, prior stages' jsonl) and they must be skipped when
#: enumerating per-doc inputs.
_NON_DOC_JSONL_NAMES = frozenset({
    "sitemap.jsonl",
    "docs.jsonl",
    "extract.jsonl",
    "extract_manifest.json",
    "parse_manifest.json",
    "manifest.json",
})
DEFAULT_LICENSE = "cc-by-4.0"
DEFAULT_REPO_OWNER = "tmquan"
DEFAULT_REPO_NAME = "vbpl-vn"

#: Maximum rows per parquet shard. vbpl rows are *fatter* than the
#: cross-corpus default (10 K) because each one carries the full
#: ``structure_json`` (sections + paragraphs + sentences with char
#: spans) and ``extracted_json`` (entities + statute_refs) on top
#: of the markdown body itself (max 2.4 MB for a single doc). At
#: 10 K rows / shard the largest shard hit 214 MB which is close
#: to the HF dataset-viewer memory cliff that triggered the
#: original ``JobManagerCrashedError``. Dropping to 5 K rows / shard
#: fans the 158 K-row corpus into ~32 shards of ~50-110 MB each --
#: well inside the viewer's per-job heartbeat budget, and still
#: under the 500 MB HF parquet ceiling for any single file. Names
#: follow the HF Datasets convention:
#: ``documents-NNNNN-of-NNNNN.parquet``.
CHUNK_SIZE = 5_000

#: Parquet row-group size inside each shard. Smaller groups let the
#: viewer (and any downstream `datasets` consumer with `streaming=True`)
#: skim rows without materialising a multi-MB row group into RAM.
#: With ~10 K rows per shard, 512 rows/group gives ~20 groups per
#: shard -- a sweet spot for both random access and sequential reads.
PARQUET_ROW_GROUP_SIZE = 512

#: Figure stems we generate via :mod:`packages.datasites.vbpl.viz`.
#: Used by ``push_to_hf`` as the canonical "what should ship" list.
#: Splits into ``overview-*`` (six corpus-level aggregates: legal-area
#: treemap, scope→doc-type sunburst, doc-type bilingual bars, year
#: stacked area, doc-type×year heatmap, agency bars) and
#: ``embedding-*-umap`` (five UMAP scatter facets, all plotly+kaleido
#: PNGs). PCA coordinates are still in the reducer parquet but not
#: rendered -- on this corpus PCA separates the same clusters as
#: UMAP without adding insight. t-SNE was retired entirely in the
#: May-2026 global-reducer rerun: a single t-SNE fit on the full
#: 158 K × 2 048-D matrix took prohibitively long under
#: ``random_state=0``, and the reduced parquet no longer carries
#: ``tsne_x`` / ``tsne_y`` columns.
#:
#: The ``cluster-id`` UMAP facet was retired in May-2026: HDBSCAN on
#: this corpus emits ~85 % of points as the ``-1`` noise bucket on
#: the default reducer settings, so the figure was visually empty.
OVERVIEW_FIGURES = (
    "overview-legalarea-treemap.png",
    "overview-scope-doctype-sunburst.png",
    "overview-doctype-bars.png",
    "overview-year-stack.png",
    "overview-doctype-year-heatmap.png",
    "overview-agency-bars.png",
)
EMBEDDING_FIGURES = (
    "embedding-scope-umap.png",
    "embedding-doc-type-umap.png",
    "embedding-legal-type-umap.png",
    "embedding-legal-area-umap.png",
    "embedding-year-umap.png",
)
ALL_FIGURES = OVERVIEW_FIGURES + EMBEDDING_FIGURES


# ----------------------------------------------------- parquet schema


_DOCUMENT_SCHEMA = pa.schema([
    # Identification
    pa.field("doc_name",            pa.string()),
    pa.field("item_id",             pa.string()),
    pa.field("scope",               pa.string()),
    pa.field("source",              pa.string()),
    pa.field("source_url",          pa.string()),
    pa.field("api_url",             pa.string()),

    # Sidebar metadata (promoted)
    pa.field("title",               pa.string()),
    pa.field("doc_type",            pa.string()),    # canonical short code
    pa.field("legal_type",          pa.string()),    # canonical VI full name
    pa.field("legal_area",          pa.string()),    # canonical VI area name
    # ``doc_number`` is now a list because a small minority of vbpl
    # docs pack several identifiers into one source cell separated by
    # ``" và "`` ("and") or ASCII commas. Single-value docs (99%+)
    # ship a one-element list; an empty list maps to ``null`` here
    # (parquet rejects empty lists with strict schema enforcement).
    pa.field("doc_number",             pa.list_(pa.string()), nullable=True),
    pa.field("issue_date",       pa.string()),
    pa.field("year",                pa.int32()),
    pa.field("issuing_body",    pa.string()),
    pa.field("summary",           pa.string()),

    # Body
    pa.field("markdown",            pa.string()),

    # Stats
    pa.field("num_pages",           pa.int32()),
    pa.field("num_sections",        pa.int32()),
    pa.field("num_paragraphs",      pa.int32()),
    pa.field("num_sentences",       pa.int32()),
    pa.field("char_len",            pa.int32()),
    pa.field("text_hash",           pa.string()),

    # Provenance
    pa.field("parser_model",        pa.string()),
    pa.field("parser_runtime",      pa.string()),
    pa.field("body_source",         pa.string()),
    pa.field("parsed_at",           pa.string()),
    pa.field("confidence",          pa.float64()),

    # Hierarchical structure + entities (JSON-serialised)
    pa.field("structure_json",      pa.string()),
    pa.field("extracted_json",      pa.string()),

    # File attachments downloaded from the gateway minio (JSON list)
    pa.field("file_paths_json",     pa.string()),
])


# ----------------------------------------------------- record projection


def _project_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Turn one Extractor JSONL record into the parquet row shape.

    Body policy (post May-2026 recovery):

    * **``body_source == "shell_html"``** -- the Next.js shell fallback;
      vbpl never delivered a real body for these documents (the May-2026
      live-API retry confirmed the publisher only ships metadata for
      this class). Ship the row with ``markdown=None``, ``char_len=0``,
      ``text_hash=None``; the bibliographic columns (title, agency,
      doc_number, ...) are still populated so consumers retain the citation
      handle.
    * **Everything else** -- pass the already-chain-normalised
      ``markdown`` straight through, falling back to
      :func:`_synthesize_metadata_markdown` only when the body is
      empty AND the row still has enough metadata to build a useful
      stub (one or two stragglers in the corpus). Returns ``None``
      only when even the stub is empty (zero-metadata orphan row).

    .. note::

       Earlier versions of this projection treated a leading
       ``"Lỗi "`` token on the title as a vbpl editorial "broken
       record" marker and forced ``markdown=None``. That heuristic
       was retired in May 2026 -- corpus audit showed every such
       title is a legitimate use of the Vietnamese noun
       ``Lỗi``/``lỗi``/``loi`` ("fault / error"; e.g. "Lỗi Ban
       hành Quy chế hoạt động của hội đồng tư vấn mua sắm tài
       sản"), not a marker. We now respect the body the parse +
       extract pipelines produced for those rows regardless of
       title shape.

    Field-level normalization (``strip_markdown_junk``,
    ``clean_title``, ``normalise_doc_number_list``,
    ``normalise_issuing_body``, ...) is **not** done here -- the
    extract Curator pipeline's ``NormalizerChainStage`` is the single
    source of truth (wiki/DATASITES.md §3.5 + ``cfg.extractor.normalizers`` in
    ``configs/default.yaml``). The HF export is a pure projection on
    top of an already-canonical JSONL.
    """
    raw_markdown = _str_or_empty(rec.get("markdown"))
    body_source = rec.get("body_source")
    if isinstance(body_source, float) and math.isnan(body_source):
        body_source = None

    # Hard NULL class: shell_html-after-retry. The source genuinely
    # has no body for these (the May-2026 live-API retry confirmed
    # vbpl only ships metadata for this slice); the projection sets
    # markdown to None instead of synthesising a metadata stub so
    # the parquet accurately reflects "no body available on the
    # source".
    null_body = body_source == "shell_html"

    if null_body:
        markdown: str | None = None
    else:
        markdown = raw_markdown
        if not markdown:
            markdown = _synthesize_metadata_markdown(rec)
            if not markdown:
                return None

    structure = rec.get("structure") or {}
    stats = (structure.get("stats") or {}) if structure else {}

    # Defensive canonicalisation: extract JSONLs from prior runs may
    # carry the *raw* docType dict in ``doc_type``, the abbreviated
    # short code, or already the snake_case slug. Always re-derive
    # both the slug (the value written to the parquet column) and
    # the legal_type display name so the parquet ships clean values
    # regardless of upstream vintage.
    raw_doc_type = rec.get("doc_type")
    doc_type_value = _coerce_to_slug(raw_doc_type)
    legal_type_pretty = rec.get("legal_type") or legal_type_name(raw_doc_type)
    legal_area_pretty = rec.get("legal_area")
    if legal_area_pretty is None:
        legal_area_pretty = UNCATEGORISED_AREA

    # Stale-sitemap fallback: the gateway returns
    # ``invalid.document.entity.not.found`` for ~3% of corpus URLs,
    # so the API has no docType. The URL slug still encodes the
    # Vietnamese doc-type name -- recover the canonical code from
    # ``source_url`` so those rows aren't lumped into an "unknown"
    # bucket. Slug shape: ``.../van-ban/chi-tiet/<slug>--<id>``.
    if doc_type_value is None:
        source_url = rec.get("source_url") or ""
        if isinstance(source_url, str) and source_url:
            tail = source_url.rsplit("/", 1)[-1]
            slug = tail.rsplit("--", 1)[0] if "--" in tail else tail
            inferred = code_from_slug(slug)
            if inferred is not None:
                doc_type_value = CANONICAL_CODE_TO_SLUG.get(inferred)
                if not legal_type_pretty:
                    legal_type_pretty = CANONICAL_CODE_TO_NAME.get(inferred)

    # ``doc_number`` arrives as a ``list[str]`` from the chain
    # (``vbpl_doc_number_list`` normalizer + the ``doc_number`` column on
    # the per-doc JSONL); the projection simply maps an empty list
    # to ``None`` so consumers see a clean null signal.
    raw_doc_number = rec.get("doc_number")
    if isinstance(raw_doc_number, list):
        doc_number_list = [x for x in raw_doc_number if isinstance(x, str) and x]
    elif isinstance(raw_doc_number, str) and raw_doc_number.strip():
        # Legacy single-string vintage -- accept and split on commas
        # so old JSONLs still flow through. The chain runs on new
        # data; this branch is a compat shim for partial reruns.
        doc_number_list = [s.strip() for s in raw_doc_number.split(",") if s.strip()]
    else:
        doc_number_list = []
    doc_number_value: list[str] | None = doc_number_list if doc_number_list else None

    return _scrub_nan({
        # Identification
        "doc_name":   rec.get("doc_name"),
        "item_id":    rec.get("item_id"),
        "scope":      rec.get("scope"),
        "source":     rec.get("source"),
        "source_url": rec.get("source_url"),
        "api_url":    rec.get("api_url"),

        # Sidebar metadata.
        # ``title`` arrives already cleaned by ``vbpl_clean_title``
        # (legal_type / doc_number head peeling, cross-ref stripping,
        # decorative quote stripping). The chain returns ``None``
        # for degenerate titles (e.g. just a doc-num token like
        # ``"1938/QĐ-UBND"``); the parquet then ships
        # ``title=null`` with the bibliographic columns
        # (doc_number, legal_type, ...) still carrying the citation
        # handle. The Vietnamese noun ``Lỗi`` ("fault / error")
        # is preserved wherever it appears in source titles --
        # it's a legitimate subject word, not a CMS marker.
        "title":            rec.get("title"),
        "doc_type":         doc_type_value,
        "legal_type":       legal_type_pretty,
        "legal_area":       legal_area_pretty,
        "doc_number":          doc_number_value,
        "issue_date":    rec.get("issue_date"),
        "year":             _year_from(rec.get("issue_date")),
        "issuing_body": rec.get("issuing_body"),
        "summary":        rec.get("summary"),

        # Body. ``markdown`` is None for shell_html-after-retry rows
        # so the parquet faithfully signals "no body on source"
        # instead of synthesising a stub.
        "markdown":     markdown,

        # Stats. ``char_len`` and ``text_hash`` are recomputed from
        # the post-cleanup ``markdown`` so consumers can dedupe on
        # the actual exported body, not the upstream cached value.
        # NULL-markdown rows get ``char_len=0`` and ``text_hash=None``.
        "num_pages":      _coerce_int(rec.get("num_pages")),
        "num_sections":   _coerce_int(stats.get("num_sections")),
        "num_paragraphs": _coerce_int(stats.get("num_paragraphs")),
        "num_sentences":  _coerce_int(stats.get("num_sentences")),
        "char_len":       0 if markdown is None else len(markdown),
        "text_hash":      (
            None if markdown is None
            else hashlib.sha256(markdown.encode("utf-8")).hexdigest()[:32]
        ),

        # Provenance
        "parser_model":   rec.get("parser_model"),
        "parser_runtime": rec.get("parser_runtime"),
        "body_source":    rec.get("body_source"),
        "parsed_at":      rec.get("parsed_at"),
        "confidence":     rec.get("confidence"),

        # JSON serialisation: dump structure / extracted as compact strings.
        "structure_json": (
            json.dumps(structure, ensure_ascii=False) if structure else None
        ),
        "extracted_json": (
            json.dumps(rec["extracted"], ensure_ascii=False)
            if rec.get("extracted") else None
        ),
        "file_paths_json": (
            json.dumps(rec["file_paths"], ensure_ascii=False)
            if rec.get("file_paths") else None
        ),
    })


def _str_or_empty(value: Any) -> str:
    """Return ``value`` as a stripped string, or ``""`` for null-ish input.

    Treats ``None``, ``float('nan')``, and any non-string non-numeric
    falsy value as empty. Pandas writes missing cells through
    ``json.dumps`` as ``NaN`` and ``json.loads`` round-trips that
    back to a Python ``float('nan')`` which is **truthy** -- the old
    ``rec.get("x") or ""`` pattern in synthesis / projection
    therefore short-circuited on NaN and crashed ``.strip()``.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _is_nan(value: Any) -> bool:
    """True iff ``value`` is exactly a ``float('nan')`` (pandas null)."""
    return isinstance(value, float) and math.isnan(value)


def _scrub_nan(row: dict[str, Any]) -> dict[str, Any]:
    """Replace ``float('nan')`` values with ``None`` in a projected row.

    PyArrow ``from_pylist`` rejects NaN where a string / binary column
    is declared in the schema. Pandas serialises missing cells via
    ``json.dumps`` as ``NaN`` and ``json.loads`` round-trips that to
    Python's truthy ``float('nan')``; the projection therefore sees
    NaN in any nullable string column that was empty in the source
    DataFrame. Mapping NaN -> ``None`` makes the row pyarrow-safe
    without forcing every individual ``rec.get(...)`` access to
    defensively coerce.
    """
    return {k: (None if _is_nan(v) else v) for k, v in row.items()}


def _synthesize_metadata_markdown(rec: dict[str, Any]) -> str:
    """Build a minimal markdown body from the sidebar metadata.

    Used for the ~1 doc in the corpus where the gateway returned
    ``hasOriginalPdf=True`` but no actual body text or PDF download
    URL (i.e. the document exists in the legacy index but its body
    was never migrated to the modern SPA). The synthesized body
    preserves every field the embedder + downstream consumers care
    about: title, document number, issue date, issuing agency,
    document type, abstract. Empty fields are dropped silently so
    the output is whatever metadata the source actually exposed.
    """
    # ``_str_or_empty`` defends against the ``float('nan')`` values
    # that surface when the per-doc JSONL was serialised from a
    # pandas DataFrame (pandas writes NaN through ``json.dumps`` as
    # ``NaN`` and ``json.loads`` round-trips it back to a Python
    # ``float('nan')``); a truthy NaN slipped through the old
    # ``rec.get(...) or ""`` pattern and crashed ``.strip()``.
    title = _str_or_empty(rec.get("title"))
    raw_doc_number = rec.get("doc_number")
    if isinstance(raw_doc_number, list):
        doc_number = ", ".join(
            _str_or_empty(x) for x in raw_doc_number if _str_or_empty(x)
        ).strip()
    else:
        doc_number = _str_or_empty(raw_doc_number)
    legal_type = _str_or_empty(rec.get("legal_type"))
    agency = _str_or_empty(rec.get("issuing_body"))
    date = _str_or_empty(rec.get("issue_date"))
    trich = _str_or_empty(rec.get("summary"))
    if not (title or doc_number or agency or trich):
        return ""
    lines: list[str] = []
    if title:
        lines.append(f"# {title}")
        lines.append("")
    bullets: list[str] = []
    if legal_type:
        bullets.append(f"- Loại văn bản · Document type: {legal_type}")
    if doc_number:
        bullets.append(f"- Số hiệu · Document number: {doc_number}")
    if date:
        bullets.append(f"- Ngày ban hành · Issue date: {date}")
    if agency:
        bullets.append(f"- Cơ quan ban hành · Issuing agency: {agency}")
    if bullets:
        lines.extend(bullets)
        lines.append("")
    if trich:
        lines.append("## Trích yếu · Abstract")
        lines.append("")
        lines.append(trich)
        lines.append("")
    lines.append(
        "> _Lưu ý: cổng vbpl.vn chỉ trả về metadata cho văn bản này; "
        "phần nội dung không khả dụng trên gateway công cộng. "
        "(Note: the vbpl.vn gateway returned only metadata for this "
        "document; the body is not publicly available.)_",
    )
    return "\n".join(lines).strip()


def _coerce_int(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_to_slug(raw: Any) -> str | None:
    """Normalise ``doc_type`` to its snake_case slug across all vintages.

    Accepts the four shapes the column has had in this repo's
    history:

    * ``None`` / empty -- returned as-is.
    * A dict ``{"code": "QĐ", "name": "Quyết định", ...}`` -- the
      raw API shape; passed through :func:`doc_type_slug` which
      delegates to :func:`canonical_code` for the code lookup.
    * A short code string ``"QĐ"`` / ``"TTLT"`` / ``"CThi"`` -- the
      shape the parquet shipped before this rework. Mapped to the
      slug via :data:`CANONICAL_CODE_TO_SLUG` (or the raw->canonical
      lookup chain for non-canonical legacy codes like ``"CThi"``).
    * A snake_case slug string ``"quyet_dinh"`` -- already in the
      target form; round-tripped through
      :data:`SLUG_TO_CANONICAL_CODE` to validate and return.

    Returns ``None`` for unrecognised inputs so the caller can fall
    back to the URL-slug heuristic.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return None
        if s in SLUG_TO_CANONICAL_CODE:
            return s  # already a slug
        # Otherwise try to interpret as a (possibly legacy) code or
        # short canonical abbreviation via doc_type_slug.
    return doc_type_slug(raw)


def _year_from(date_str: Any) -> int | None:
    """Pull the YYYY year off an ISO ``YYYY-MM-DD`` issue-date string."""
    if not date_str or not isinstance(date_str, str):
        return None
    if len(date_str) >= 4 and date_str[:4].isdigit():
        try:
            year = int(date_str[:4])
            if 1900 <= year <= 2100:
                return year
        except ValueError:
            pass
    return None


def _iter_per_doc_jsonl(jsonl_dir: Path) -> Iterator[dict[str, Any]]:
    """Yield one record per ``<doc_name>.jsonl`` under ``jsonl_dir``.

    The vbpl extract Curator pipeline writes the raw per-doc tier
    (wiki/DATASITES.md §3.5) -- one JSON object per file keyed by
    ``doc_name``. We sort by filename so the projection is
    deterministic across runs. Sidecars / stage manifests that
    happen to share the ``*.jsonl`` extension (``sitemap.jsonl``,
    ``docs.jsonl``, the retired single-file ``extract.jsonl``, …)
    are skipped.
    """
    for path in sorted(jsonl_dir.glob("*.jsonl")):
        if path.name in _NON_DOC_JSONL_NAMES:
            continue
        if path.name.startswith("extract.jsonl."):
            continue
        yield from iter_jsonl(path)


def _iter_projected(
    jsonl_dir: Path,
) -> Iterator[tuple[dict[str, Any] | None, dict[str, Any]]]:
    """Yield ``(projected_or_None, raw_record)`` so the manifest can count empties."""
    for rec in _iter_per_doc_jsonl(jsonl_dir):
        yield _project_record(rec), rec


def _write_sharded_parquet(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
    chunk_size: int,
    row_group_size: int,
) -> list[Path]:
    """Write ``rows`` as fixed-size parquet shards (``chunk_size`` rows each).

    File naming: ``documents-NNNNN-of-KKKKK.parquet``. Each shard is
    written with a small ``row_group_size`` so the HF dataset viewer
    (and any ``streaming=True`` consumer) can skim rows without
    materialising large row groups into memory.

    Also wipes any *legacy* single-file ``documents.parquet`` and any
    stale shard files from a previous run with a different shard
    count so the published folder stays in sync with the YAML
    ``data_files: documents-*.parquet`` glob.

    Row count per shard is fixed at ``chunk_size`` (10 K by default,
    matching the cross-corpus convention shared with ``anle`` /
    ``congbobanan``); the trailing shard absorbs the remainder.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from packages.common.hf import coerce_for_schema

    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >=1, got {chunk_size}")

    legacy = out_dir / "documents.parquet"
    if legacy.exists():
        logger.info("removing legacy single-file %s", legacy)
        legacy.unlink()
    for stale in sorted(out_dir.glob("documents-*.parquet")):
        stale.unlink()

    coerced = coerce_for_schema(rows, _DOCUMENT_SCHEMA)
    total = len(coerced)
    num_shards = max(1, (total + chunk_size - 1) // chunk_size)
    shard_paths: list[Path] = []
    total_bytes = 0
    for i in range(num_shards):
        chunk = coerced[i * chunk_size:(i + 1) * chunk_size]
        if not chunk:
            continue
        table = pa.Table.from_pylist(chunk, schema=_DOCUMENT_SCHEMA)
        shard_path = out_dir / f"documents-{i:05d}-of-{num_shards:05d}.parquet"
        pq.write_table(
            table,
            shard_path,
            compression="zstd",
            row_group_size=row_group_size,
        )
        shard_paths.append(shard_path)
        size_mb = shard_path.stat().st_size / 1024 / 1024
        total_bytes += shard_path.stat().st_size
        logger.info(
            "wrote shard %s (%d rows, %.1f MB)",
            shard_path.name, table.num_rows, size_mb,
        )
    logger.info(
        "wrote %d parquet shards, %d total rows, %.1f MB combined",
        len(shard_paths), total, total_bytes / 1024 / 1024,
    )
    return shard_paths


# ----------------------------------------------------- embedding viz


def _read_embedder_info(reduced_dir: Path) -> tuple[str | None, int | None]:
    """Sniff one reduced parquet to learn the actual embedder model + dim.

    Returns ``(model_id, embedding_dim)`` so the dataset card mentions
    the embedder that *actually* ran, rather than the default declared
    in ``cfg.embedder``. This matters when the operator overrides the
    embedder at runtime (e.g. swapping nemotron-1b for the multilingual
    mpnet fallback).
    """
    if not reduced_dir.is_dir():
        return None, None
    parquets = sorted(reduced_dir.glob("*.parquet"))
    if not parquets:
        return None, None
    try:
        df = pd.read_parquet(parquets[0], columns=["embedding_model_id", "embedding_dim"])
    except Exception:
        return None, None
    if df.empty:
        return None, None
    model_id = df["embedding_model_id"].iloc[0]
    dim = df["embedding_dim"].iloc[0]
    return (
        str(model_id) if model_id is not None else None,
        int(dim) if dim is not None else None,
    )


def _render_figures(
    parquet_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    reduced_dir: Path,
    out_dir: Path,
    *,
    embed_model_id: str | None = None,
    embed_dim: int | None = None,
) -> dict[str, Path]:
    """Render the full plotly figure pack via :mod:`viz`.

    Six corpus-overview figures (treemap, sunburst, doc-type bars,
    year stack, doc-type×year heatmap, agency bars) plus six UMAP
    scatter facets, all PNG via kaleido. Returns a ``{stem: path}``
    dict where ``stem`` is the figure's canonical identifier (e.g.
    ``"legalarea_treemap"``); empty when chromium can't be found
    (the rest of ``hf_export`` still ships in that case).
    """
    from packages.datasites.vbpl import viz

    return viz.render_all(
        manifest=manifest,
        rows=parquet_rows,
        reduced_dir=reduced_dir,
        out_dir=out_dir,
        embed_model_id=embed_model_id or "nvidia/llama-nemotron-embed-1b-v2",
        embed_dim=embed_dim or 2048,
    )


# ----------------------------------------------------- analytics


def _build_manifest(
    rows: list[dict[str, Any]],
    *,
    raw_total: int,
) -> dict[str, Any]:
    """Compute corpus-wide roll-ups consumed by the dataset card."""
    n = len(rows)
    by_scope = Counter(r["scope"] or "unknown" for r in rows)
    by_doc_type = Counter(r["doc_type"] or "unknown" for r in rows)
    by_legal_type = Counter(
        r.get("legal_type") or "unknown" for r in rows
    )
    by_legal_area = Counter(
        r.get("legal_area") or UNCATEGORISED_AREA for r in rows
    )
    by_agency = Counter(
        (r.get("issuing_body") or "unknown") for r in rows
    )
    by_year = Counter(r["year"] for r in rows if r["year"] is not None)
    by_body_source = Counter(r["body_source"] or "unknown" for r in rows)
    char_lens = [r["char_len"] for r in rows if r["char_len"]]
    para_counts = [r["num_paragraphs"] for r in rows if r["num_paragraphs"]]
    sent_counts = [r["num_sentences"] for r in rows if r["num_sentences"]]
    pages = [r["num_pages"] for r in rows if r["num_pages"]]
    has_attachment = sum(1 for r in rows if r.get("file_paths_json"))
    # Count rows that ship with markdown=null. This is now just the
    # ``body_source == "shell_html"`` slice that the May-2026
    # live-API recovery confirmed has no body on the source; the
    # manifest exposes the count so the dataset card can document
    # the gap with a precise number.
    null_markdown_rows = sum(1 for r in rows if r.get("markdown") is None)

    def _pct(c: Counter, top_n: int = 25) -> dict[str, dict[str, Any]]:
        return {
            k: {"count": v, "share": v / max(n, 1)}
            for k, v in c.most_common(top_n)
        }

    return {
        "corpus": {
            "documents":    n,
            "raw_rows":     raw_total,
            "dropped_empty": raw_total - n,
            "null_markdown_rows": null_markdown_rows,
            "with_structure":   sum(
                1 for r in rows if r.get("structure_json") is not None
            ),
            "with_attachment":  has_attachment,
            "char_len":   _summary(char_lens),
            "pages":      _summary(pages),
            "paragraphs": _summary(para_counts),
            "sentences":  _summary(sent_counts),
        },
        "by_scope":         _pct(by_scope),
        "by_doc_type":      _pct(by_doc_type, top_n=30),
        "by_legal_type":    _pct(by_legal_type, top_n=30),
        "by_legal_area":    _pct(by_legal_area, top_n=25),
        "by_agency":        _pct(by_agency, top_n=15),
        "by_year":          {str(k): v for k, v in sorted(by_year.items())},
        "by_body_source":   _pct(by_body_source),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def _summary(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    s = sorted(values)
    n = len(s)
    return {
        "n":      n,
        "min":    s[0],
        "max":    s[-1],
        "mean":   round(sum(s) / n, 1),
        "median": s[n // 2],
    }


# ----------------------------------------------------- dataset card


def _format_int(n: int) -> str:
    return f"{n:,}"


def _yaml_frontmatter(
    manifest: dict[str, Any], license_id: str,
) -> str:
    n = manifest["corpus"]["documents"]
    if n < 1_000:
        size_cat = "n<1K"
    elif n < 10_000:
        size_cat = "1K<n<10K"
    elif n < 100_000:
        size_cat = "10K<n<100K"
    elif n < 1_000_000:
        size_cat = "100K<n<1M"
    else:
        size_cat = "1M<n<10M"
    return f"""---
language:
- vi
license: {license_id}
pretty_name: "Vietnamese National Legal Database (vbpl.vn)"
size_categories:
- {size_cat}
task_categories:
- text-classification
- text-retrieval
- question-answering
- text-generation
- summarization
tags:
- legal
- vietnamese
- vietnam
- law
- statute
- regulation
- legislation
- moj
- ministry-of-justice
source_datasets:
- original
configs:
- config_name: documents
  default: true
  data_files:
  - split: train
    path: documents-*.parquet
---
"""


def _bar(c: dict[str, dict[str, Any]], top_n: int = 12) -> str:
    rows = ["| Value | Count | Share |", "|---|---:|---:|"]
    for k, v in list(c.items())[:top_n]:
        rows.append(f"| `{k}` | {_format_int(v['count'])} | {100*v['share']:.1f}% |")
    return "\n".join(rows)


def _year_block(by_year: dict[str, int], top_n: int = 30) -> str:
    if not by_year:
        return "_(no year metadata in this slice)_"
    items = sorted(by_year.items(), reverse=True)[:top_n]
    rows = ["| Year | Count |", "|---:|---:|"]
    for k, v in items:
        rows.append(f"| {k} | {_format_int(v)} |")
    return "\n".join(rows)


#: Captions for the six overview figures. Bilingual VI/EN, kept short
#: so they don't compete with the chart itself.
_OVERVIEW_CAPTIONS: tuple[tuple[str, str, str, str], ...] = (
    (
        "legalarea_treemap",
        "overview-legalarea-treemap.png",
        "Phân bố theo lĩnh vực pháp luật",
        "Top 25 legal areas sized by document count. The pale "
        "`Chưa phân loại` ('uncategorised') rectangle covers ~72% on "
        "its own -- vbpl.vn editors haven't tagged most legacy docs.",
    ),
    (
        "scope_doctype_sunburst",
        "overview-scope-doctype-sunburst.png",
        "Phạm vi → loại văn bản",
        "Two-level radial split: `scope` (inner) → `doc_type` (outer, "
        "top 12 per scope). Reveals where the corpus weight sits -- "
        "`dia_phuong` (~66%) is mostly `QĐ` (decisions) and `NQ` "
        "(resolutions), while `trung_uong` mixes `QĐ` / `NĐ` "
        "(decrees) / `TT` (circulars).",
    ),
    (
        "doctype_bars",
        "overview-doctype-bars.png",
        "Top 20 loại văn bản",
        "Document types ranked by count, with the canonical short "
        "code, full Vietnamese name, and an English gloss.",
    ),
    (
        "year_stack",
        "overview-year-stack.png",
        "Số văn bản theo năm và phạm vi",
        "Stacked area of documents issued per year, split by `scope`. "
        "The legacy CMS migration concentrates early years; modern "
        "post-2010 output is dominated by `dia_phuong`.",
    ),
    (
        "doctype_year_heatmap",
        "overview-doctype-year-heatmap.png",
        "Loại văn bản qua các năm",
        "Top-12 `doc_type` × year heatmap (log₁₀ scale). Shows the "
        "shift from `CT` directives in the early 90s to `QĐ` decisions "
        "as the workhorse instrument from 2000 onwards.",
    ),
    (
        "agency_bars",
        "overview-agency-bars.png",
        "Top 15 cơ quan ban hành",
        "Top issuing agencies. Quốc hội + Chính phủ + the larger "
        "ministries dominate `trung_uong`; provincial People's "
        "Councils / Committees split the `dia_phuong` half.",
    ),
)


def _overview_viz_section(viz_paths: dict[str, Path]) -> str:
    """Markdown block embedding the six corpus-overview PNGs.

    Each picture has a single-paragraph bilingual caption. Pictures
    that are missing from ``viz_paths`` (e.g. kaleido failed to
    render) are silently skipped -- the section just gets shorter.
    """
    rendered = [
        (label, fname, vi, en)
        for label, fname, vi, en in _OVERVIEW_CAPTIONS
        if label in viz_paths
    ]
    if not rendered:
        return ""
    blocks = [
        "## Tổng quan trực quan · Visual overview\n",
        "Sáu hình tóm tắt cấu trúc của corpus. Tất cả hình được "
        "tạo bằng `plotly` + `kaleido` (Chromium headless) và lưu "
        "thành PNG tĩnh để hiển thị trong dataset card. — Six pictures "
        "summarising the corpus structure. All figures are rendered "
        "with `plotly` + `kaleido` (headless Chromium) and saved as "
        "static PNG for the dataset card.\n",
    ]
    for label, fname, vi, en in rendered:
        blocks.append(f"### {vi}\n")
        blocks.append(f"![{vi}](./{fname})\n")
        blocks.append(f"{en}\n")
    return "\n".join(blocks) + "\n"


def _embed_viz_section(
    viz_paths: dict[str, Path],
    *,
    embed_model_id: str | None = None,
    embed_dim: int | None = None,
) -> str:
    """Markdown block embedding the five UMAP scatter PNGs.

    One section per colour facet, in a fixed order (scope first as
    the binary baseline). The intro paragraph names the embedder +
    dim so the card stays accurate when the operator overrides the
    model. The HDBSCAN ``cluster_id`` facet was retired in the
    May-2026 global-reducer rerun because ~85 % of points fall into
    the ``-1`` noise bucket on the default settings.
    """
    facet_order = (
        ("scope",       "embedding-scope-umap.png"),
        ("doc_type",    "embedding-doc-type-umap.png"),
        ("legal_type",  "embedding-legal-type-umap.png"),
        ("legal_area",  "embedding-legal-area-umap.png"),
        ("year",        "embedding-year-umap.png"),
    )
    rendered = [
        (facet, fname)
        for facet, fname in facet_order
        if f"embedding_{facet}_umap" in viz_paths
    ]
    if not rendered:
        return ""

    model_id = embed_model_id or "nvidia/llama-nemotron-embed-1b-v2"
    dim = embed_dim or 2048
    blocks: list[str] = ["## Trực quan hoá embedding · Embedding visualization\n"]
    blocks.append(
        f"Mỗi điểm là một văn bản pháp luật; toạ độ là vector embedding "
        f"{dim}-D từ `{model_id}` chiếu xuống 2D bằng **UMAP fit "
        f"globally** trên toàn bộ kho văn bản (không chia batch) để "
        f"toạ độ so sánh được giữa các tài liệu. Năm mặt phân hoạch: "
        f"`scope`, `doc_type` (mã ngắn), `legal_type` (tên đầy đủ), "
        f"`legal_area` (lĩnh vực pháp luật), `year`. Các nhãn tail-end "
        f"(sau top-18) được dồn vào nhóm *Khác / Other* màu xám để chú "
        f"giải đọc được. — Each dot is one legal document; coordinates "
        f"are the 2D UMAP projection of a {dim}-D embedding from "
        f"`{model_id}`, **fit globally across the whole corpus** so "
        f"positions are directly comparable across documents. Five "
        f"facets: `scope`, `doc_type` (canonical short code), "
        f"`legal_type` (canonical full name), `legal_area` (subject "
        f"domain), `year`. Tail-end labels beyond the top 18 are "
        f"collapsed into a grey *Khác / Other* bucket to keep the "
        f"legend legible.\n",
    )
    for facet, fname in rendered:
        title = f"UMAP colored by `{facet}`"
        blocks.append(f"### {title}\n")
        blocks.append(f"![{title}](./{fname})\n")
    return "\n".join(blocks) + "\n"


def _render_card(
    manifest: dict[str, Any],
    repo_owner: str,
    repo_name: str,
    license_id: str,
    viz_paths: dict[str, Path] | None = None,
    embed_model_id: str | None = None,
    embed_dim: int | None = None,
) -> str:
    n = manifest["corpus"]["documents"]
    raw = manifest["corpus"]["raw_rows"]
    dropped = manifest["corpus"]["dropped_empty"]
    cl = manifest["corpus"]["char_len"]
    pa_ = manifest["corpus"]["paragraphs"]
    se = manifest["corpus"]["sentences"]
    pg = manifest["corpus"]["pages"]
    front = _yaml_frontmatter(manifest, license_id)
    overview_block = _overview_viz_section(viz_paths or {})
    viz_block = _embed_viz_section(
        viz_paths or {},
        embed_model_id=embed_model_id,
        embed_dim=embed_dim,
    )
    # The "How the corpus was built" prose mentions the embedder; thread
    # the actual model+dim through so the card matches what shipped.
    embed_model_pretty = embed_model_id or "nvidia/llama-nemotron-embed-1b-v2"
    embed_dim_pretty = embed_dim or 2048
    # Embeddable-row count for the recovery prose: total documents
    # minus rows that ship with markdown=null (shell_html-after-retry
    # bucket only -- the legacy "Lỗi" prefix rule was retired in
    # May 2026 because audit showed those titles are legitimate uses
    # of the Vietnamese noun for "fault / error").
    shell_html_count = manifest.get(
        "by_body_source", {},
    ).get("shell_html", {}).get("count", 0)
    null_markdown_count = manifest["corpus"].get(
        "null_markdown_rows", shell_html_count,
    )
    embeddable_corpus_size = max(
        0, manifest["corpus"]["documents"] - null_markdown_count,
    )
    body = rf"""
# Vietnamese National Legal Database — `vbpl.vn`

> 🇻🇳 **Tóm tắt.** Bộ dữ liệu mức **văn bản** của
> **Cơ sở dữ liệu Quốc gia về pháp luật** thu thập từ cổng
> [`vbpl.vn`](https://vbpl.vn/) do Bộ Tư pháp vận hành. Bao gồm
> luật, pháp lệnh, nghị định, thông tư, quyết định, nghị quyết,
> chỉ thị… ở cả cấp **trung ương** (Quốc hội, Chính phủ, các bộ)
> lẫn **địa phương** (HĐND/UBND 63 tỉnh, thành). Mỗi dòng là một
> văn bản pháp luật kèm markdown đã chuẩn hoá tiếng Việt
> (NFC + chính tả hiện đại sau 1984) và lớp cấu trúc phân cấp đầy
> đủ: **document → section → paragraph → sentence**.
>
> 🇬🇧 **Summary.** Document-level corpus of Vietnam's
> **National Legal Database** harvested from
> [`vbpl.vn`](https://vbpl.vn/) (operated by the Ministry of Justice).
> Covers laws, ordinances, decrees, circulars, decisions,
> resolutions, directives, etc. across both **central** (National
> Assembly, Government, ministries) and **provincial** (People's
> Council / People's Committee of the 63 provinces and cities)
> levels of authority. Each row is one legal document with its
> Vietnamese-normalised markdown body (NFC + modern post-1984
> orthography) and a hierarchical structure layer:
> **document → section → paragraph → sentence**, every unit
> carrying a stable id + char span back into the markdown.

## Tổng quan · At a glance

| Chỉ số · Metric | Giá trị · Value |
|---|---:|
| Văn bản công bố · Documents | **{_format_int(n)}** |
| Tổng số bản ghi đầu vào · Raw extract rows | {_format_int(raw)} |
| Loại bỏ vì rỗng · Dropped (empty body) | {_format_int(dropped)} |
| Không có thân bài · `markdown` is null (source has no body) | {_format_int(null_markdown_count)} |
| Có cấu trúc · With structure layer | {_format_int(manifest['corpus']['with_structure'])} |
| Có tệp đính kèm · With downloaded attachment | {_format_int(manifest['corpus']['with_attachment'])} |
| Trung vị ký tự · Median chars / doc | {_format_int(cl['median']) if cl['median'] else '–'} |
| Trung vị trang · Median pages / doc | {_format_int(pg['median']) if pg['median'] else '–'} |
| Trung vị đoạn văn · Median paragraphs / doc | {_format_int(pa_['median']) if pa_['median'] else '–'} |
| Trung vị câu · Median sentences / doc | {_format_int(se['median']) if se['median'] else '–'} |

{overview_block}## Phạm vi · Scope split

Bộ dữ liệu chia làm hai nhánh: ``trung_uong`` (văn bản pháp luật do
Quốc hội + Chính phủ + các bộ ngành Trung ương ban hành) và
``dia_phuong`` (HĐND/UBND của 63 tỉnh, thành). — The corpus splits
into ``trung_uong`` (central authorities: National Assembly,
Government, ministries) and ``dia_phuong`` (the 63 provinces and
cities, mostly People's Council / People's Committee output).

{_bar(manifest['by_scope'])}

## Loại văn bản · `doc_type` + `legal_type`

Mỗi văn bản được gắn **slug tiếng Việt không dấu** (`doc_type`,
ví dụ `quyet_dinh`, `nghi_dinh`, `thong_tu_lien_tich`) lẫn tên đầy
đủ tiếng Việt (`legal_type`, ví dụ `Quyết định`, `Nghị định`,
`Thông tư liên tịch`). Slug được sinh tự động từ tên đầy đủ qua
[`packages.datasites.vbpl.codes.slugify_vi`](https://github.com/tmquan/ViLA/blob/main/packages/datasites/vbpl/codes.py)
nên người đọc hiểu ngay loại văn bản mà không cần tra cứu bảng
mã. Mã ngắn cũ (`QĐ`, `NĐ`, `TTLT`, ...) vẫn xuất hiện trong số
hiệu (`43/2026/NĐ-CP`) và có thể khôi phục từ
`SLUG_TO_CANONICAL_CODE`. — Each document carries a self-describing
**ASCII snake_case slug** (`doc_type`, e.g. `quyet_dinh`,
`nghi_dinh`, `thong_tu_lien_tich`) and the canonical Vietnamese
full name (`legal_type`, e.g. `Quyết định`, `Nghị định`,
`Thông tư liên tịch`). The slug is automatically derived from the
full Vietnamese name via
[`slugify_vi`](https://github.com/tmquan/ViLA/blob/main/packages/datasites/vbpl/codes.py)
so a reader can interpret the doc type without consulting a separate
codebook. The compact short code (`QĐ`, `NĐ`, `TTLT`, ...) still
appears inside `doc_number` itself (`43/2026/NĐ-CP`) and can be
recovered from any slug via the `SLUG_TO_CANONICAL_CODE` table in
`codes.py`. The set of slugs follows Luật Ban hành Văn bản Quy phạm
Pháp luật 2015 — `hien_phap` / `bo_luat` / `luat` / `phap_lenh` /
`lenh` / `nghi_quyet` / `nghi_dinh` / `quyet_dinh` / `thong_tu` /
`chi_thi` / `sac_lenh` / `van_ban_hop_nhat` / `thong_tu_lien_tich`
/ ...

### By slug · `doc_type`

{_bar(manifest['by_doc_type'])}

### By full name · `legal_type`

{_bar(manifest['by_legal_type'])}

## Lĩnh vực pháp luật · `legal_area`

`legal_area` is the canonical Vietnamese label pulled from the
``documentFields[]`` block on each detail-API response (~250
distinct labels in the corpus). Roughly two thirds of the documents
are tagged `Chưa phân loại` ("uncategorised") on the source portal;
we preserve that literal so consumers can see the gap explicitly
instead of nulling it. Top areas below; embedding scatters below
show the structural overlap between adjacent areas.

{_bar(manifest['by_legal_area'], top_n=20)}

## Cơ quan ban hành · Issuing agency

Top issuing agencies (top 15). Quốc hội + Chính phủ + Bộ Tài chính
+ Bộ Tư pháp + ... thường chiếm phần lớn `trung_uong`; các tỉnh
chia khá đều phần `dia_phuong`.

{_bar(manifest['by_agency'], top_n=15)}

## Năm ban hành · Year of issue

Phân bố năm theo `issue_date` (ISO `YYYY-MM-DD`); rỗng nếu cổng
không cung cấp được trường này. — Year distribution from
`issue_date`; null when the source portal didn't expose the issue
date.

{_year_block(manifest['by_year'])}

## Nguồn nội dung · Body provenance

Một văn bản trên `vbpl.vn` có thể có **HTML thân bài** (do API SPA
trả về sau khi qua reCAPTCHA) và/hoặc một **tệp đính kèm**
(`.pdf` / `.doc` / `.docx`). Pipeline ưu tiên parse tệp khi có,
quay về HTML khi không. — A vbpl document may have an inline
**body HTML** (returned by the SPA's API after reCAPTCHA) and/or a
downloadable **attachment** (`.pdf` / `.doc` / `.docx`). The pipeline
prefers parsing the file when present and falls back to the HTML
otherwise.

{_bar(manifest['by_body_source'])}

## Lược đồ bảng `documents` · `documents` schema

The parquet has three families of columns:

### Identification + meta

| Field | Type | Description |
|---|---|---|
| `doc_name` / `item_id` | string | Stable document id (= the `--<id>` suffix of the source URL). Mostly numeric (`186739`); legacy docs use `vbpqta_<n>` (Văn bản pháp quy toàn văn) or `vbpqdinhchinh_<n>` (corrigendum). |
| `scope` | string | `trung_uong` (central) \| `dia_phuong` (provincial). |
| `source` | string | Source host, always `vbpl.vn`. |
| `source_url` / `api_url` | string | Deep link back to the portal page / the underlying gateway API. |
| `title` | string (nullable) | Document subject after the full title scrub: (1) baseline cleanup (NFC, HTML-entity decode, `"số số"` doubling fix, trailing sentence punctuation), (2) decorative quote stripping at the title boundary and around inline phrases — leading / trailing / paired runs of `"`, `'`, `"`, `'`, `'`, `'` are removed but **word-internal single quotes are preserved** so legitimate Vietnamese / loan-word names survive intact (`Đắk R'lấp`, `M'nông`, `D'Ran`, `Côte d'Ivoire`, `Ea T'ling`, `Đạ M'ri`, …), (3) leading `"<Legal-type> số <Number> "` boilerplate head peeled (e.g. `"Quyết định số 143/QĐ-KHTC Ban hành Quy chế"` → `"Ban hành Quy chế"`), and (4) every `"<DocType> [số] <DocNum> [ngày <date>]"` cross-reference of OTHER documents stripped from anywhere in the body (`"Bãi bỏ Nghị quyết số 84/2018/NQ-HĐND ngày 07/12/2018 của HĐND tỉnh..."` → `"Bãi bỏ của HĐND tỉnh..."`). The Vietnamese noun `Lỗi` / `lỗi` / `loi` ("fault / error") is preserved wherever it appears — it's a legitimate subject word (e.g. `"Lỗi Ban hành Quy chế hoạt động của hội đồng tư vấn mua sắm tài sản"`), not a CMS marker. `null` for the pathological tail where the entire source title was just a bare doc-num token (e.g. `"1938/QĐ-UBND"`); other bibliographic columns still carry the citation handle. |
| `doc_type` | string | Self-describing **ASCII snake_case slug** of the Vietnamese doc-type name (`quyet_dinh`, `nghi_dinh`, `thong_tu_lien_tich`, `chi_thi`, `van_ban_hop_nhat`, ...). Auto-derived from `legal_type` via `slugify_vi`; the compact short code (`QĐ`, `NĐ`, `TTLT`, ...) still appears in `doc_number` itself (`43/2026/NĐ-CP`) and is recoverable via `SLUG_TO_CANONICAL_CODE`. |
| `legal_type` | string | Canonical Vietnamese **full name** for the document type (`Luật`, `Nghị định`, `Thông tư`, `Quyết định`, `Nghị quyết`, `Chỉ thị`, …). Round-trips with `doc_type` via `slugify_vi`. |
| `legal_area` | string | Legal area / subject domain (`Đất đai`, `Đường bộ`, `Lĩnh vực giá`, …). Defaults to `Chưa phân loại` when the source portal hasn't tagged the doc. |
| `doc_number` | list&lt;string&gt; (nullable) | Document number(s) as a list of canonical short forms (e.g. `["43/2026/NĐ-CP"]`). A small minority of vbpl rows pack several identifiers into one source cell separated by Vietnamese ` và ` ("and") or `,`; those ship as multi-element lists (e.g. `["142/2009/QĐ-TTg", "49/2012/QĐ-TTg"]`). Source-side cruft is stripped: leading legal-type words (`"Nghị quyết số: 528/2018/UBTVQH14"` → `["528/2018/UBTVQH14"]`), trailing annotations (`"109/2005/QĐ-BCA (A11)"` → `["109/2005/QĐ-BCA"]`, `"49/2007/TTLT-BTC-BGD ngày 18/5/2007"` → `["49/2007/TTLT-BTC-BGD"]`), and the legitimate `"Không số"` ("no number") sentinel is preserved. `null` for rows with no usable number. |
| `issue_date` | string | Issue date, ISO `YYYY-MM-DD`. |
| `year` | int32 | Year extracted from `issue_date`. |
| `issuing_body` | string | Issuing agency (e.g. `"Chính phủ"`, `"Bộ Tài chính"`, `"Hội đồng nhân dân tỉnh A"`). |
| `summary` | string | Abstract / summary. |

### Body + stats

| Field | Type | Description |
|---|---|---|
| `markdown` | string (nullable) | NFC-normalised, modern-orthography Vietnamese markdown (page-segmented with `## Page N` headings when parsed from a PDF). Gateway / Word / Next.js scaffolding is stripped: the `Document Content\\n\\nbody {{ font-family: ... }}\\np {{ margin: ... }}` API preamble (~50 % of bodies), Word `<!-- /* Font Definitions */ @font-face {{ ... }} p.MsoNormal {{ ... }} -->` stylesheet dumps, standalone CSS rule blocks gated by property / selector / structural CSS tells, Ant Design `:where(.css-...){{ ... }}@keyframes ...{{ ... }}` chains, orphan selector fragments, and malformed inline `<span lang="..." style="...">` tags. ~90 % of rows lose 1-200 KB of boilerplate; total corpus markdown shrunk by ~1.77 GB / 42 %. **Null** when `body_source == "shell_html"` after the May-2026 live-API recovery (the source genuinely has no body for those legacy IDs); bibliographic metadata (title, agency, doc_number, ...) is still populated on NULL-markdown rows so consumers retain the citation handle. |
| `num_pages` | int32 | Page count from the parser (PDF/DOCX only). |
| `num_sections` / `num_paragraphs` / `num_sentences` | int32 | Counts from the structure layer. |
| `char_len` | int32 | Character length of `markdown` **after** the junk-strip pass (recomputed at export time so consumers can dedupe on the actual shipped body). |
| `text_hash` | string | SHA-256 first-32 hex of `markdown` **after** the junk-strip pass (re-run-stable id; differs from the upstream `extract.jsonl` hash for any row that had scaffolding removed). |
| `parser_model` | string | Backend that produced the markdown (`local/pypdf`, `local/markdownify`, `nvidia/nemoretriever-parse`, …). |
| `parser_runtime` | string | The configured `parser.runtime` (`local` / `nim` / `hybrid`). |
| `body_source` | string | Which source produced the body: `file` (downloaded PDF/.doc/.docx), `body_html` (API-captured), `shell_html` (Next.js shell fallback -- the gateway never delivered a real body; published with `markdown=null` after the May-2026 live-API recovery sweep confirmed the source publishes only metadata for those legacy IDs). |
| `parsed_at` | string | ISO 8601 parser timestamp. |

### Hierarchy + entities

| Field | Type | Description |
|---|---|---|
| `structure_json` | string | Full :class:`DocumentStructure` (meta + stats + sections + paragraphs + sentences) as JSON. Includes char-span back-pointers so any unit can be located in `markdown` precisely. |
| `extracted_json` | string | Generic NER + statute-link extraction (entities, relations, statute_refs) as JSON. |
| `file_paths_json` | string | Downloaded attachments as JSON list of `{{file_url, file_name, file_type, local_path}}`. |

Quick load:

```python
import json
from datasets import load_dataset

ds = load_dataset("{repo_owner}/{repo_name}", split="train")
row = ds[0]
print(row["doc_type"], row["doc_number"], row["issue_date"])
# e.g. "quyet_dinh 143/QĐ-KHTC 2018-01-29"
print(row["title"])
# e.g. "Ban hành Quy chế quản lý ngân sách ngành Tư pháp"
# (the redundant "Quyết định số 143/QĐ-KHTC " head has been
#  stripped; recombine via f"{{row['legal_type']}} số {{row['doc_number']}} {{row['title']}}"
#  if your downstream pipeline still wants the full original.)
structure = json.loads(row["structure_json"])
for sec in structure.get("sections", []):
    print(sec["kind"], sec["label"])
```

## Embedding + reduction artefacts

The Hub ships the **`documents` config only**
(`documents-*.parquet`, one row per document, with text +
structure). The {embed_dim_pretty}-D
`{embed_model_pretty}` embeddings, the global UMAP / PCA
coordinates, and the HDBSCAN cluster ids are computed during the
build (over {_format_int(embeddable_corpus_size)} embeddable rows,
i.e. all documents with non-null `markdown`) and used to render
the scatter PNGs below, but they are **not bundled as separate
parquet configs on the Hub** -- 1.3 GB of dense vectors per
re-build is too costly to ship for a corpus this size when the
embeddings are deterministic from `markdown` + a model id.
Re-derive the same matrices locally via:

```bash
git clone https://github.com/<owner>/ViLA   # the build repo
python -m packages.datasites.vbpl --pipeline embed
python -m packages.datasites.vbpl.\_reduce\_inproc   # global UMAP
```

which yields `parquet/embeddings/<doc_name>.parquet` and
`parquet/reduced/<doc_name>.parquet` per-doc shards keyed back to
`documents.doc_name`.

{viz_block}## Cách thu thập + chuẩn hoá · How the corpus was built

The crawler is a six-stage pipeline (`harvest` → `detail` → `parse`
→ `extract` → `embed` → `reduce`) that walks vbpl.vn's public
sitemap chain (32 shards, ~160 K URLs total), drives a headless
Chromium tab against each detail page so Google's invisible
reCAPTCHA v2 mints the per-session Bearer token (the SPA's
`/api/qtdc/public/doc/...` gateway is otherwise inaccessible),
intercepts the resulting authenticated XHRs, downloads any
`.pdf` / `.doc` / `.docx` attachment, and routes the body through:

1. **Parse** -- pypdf for PDFs, docx2txt for `.docx`, an
   `antiword` / `catdoc` / `libreoffice` subprocess fallback for
   legacy `.doc`, `markdownify` for HTML bodies returned inline.
2. **Vietnamese normalisation** -- ftfy NFC + tone-mark
   canonicalisation (`Toà → Tòa`, `hoà → hòa`, `thuỷ → thủy`) +
   PDF whitespace cleanup. Every regex / segmenter downstream then
   sees a single canonical orthography.
3. **Generic + structure extractor** -- regex / dictionary NER +
   Vietnamese statute linker (`Điều N khoản M Luật ...`, dates
   `dd/MM/yyyy`, courts, agencies, document numbers) + a
   hierarchical `DocumentStructure` (sections / paragraphs /
   sentences with back-pointers).
4. **Embed** -- `{embed_model_pretty}` ({embed_dim_pretty}-D) over the
   normalised markdown; sliding-window mean pool when a doc exceeds
   the model's native context window.
5. **Reduce** -- PCA + UMAP on the embedding matrix + HDBSCAN
   cluster ids. The May-2026 rerun switched from per-batch
   reduction (each Curator `DocumentBatch` independently) to a
   **global in-process fit** over the full {_format_int(embeddable_corpus_size)}
   × {embed_dim_pretty}-D matrix so the projection is comparable
   across documents (`packages/datasites/vbpl/_reduce_inproc.py`).
   t-SNE was dropped from the rerun because a single global fit
   under `random_state=0` (single-threaded by construction) took
   prohibitively long on this corpus; PCA + UMAP cover the same
   visual story without it. cuML on a GPU worker; sklearn /
   umap-learn / hdbscan otherwise.

All five layers are deterministic and re-runnable; re-running any
stage with the same `--limit` is a no-op (each stage skips
already-produced outputs).

### Body-recovery rerun (May 2026)

A targeted retry against the public gateway
`https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/<id>`
recovered **3,849 / 15,351** previously bodyless documents (the
gateway now publishes real `documentContent.content` HTML for that
slice even though our cached SPA fetch came back empty in
2026-04). The remaining ~11.5K are genuinely bodyless on the
official vbpl source (legacy / cancelled documents the publisher
no longer carries a full text for) and ship in this dataset with
`markdown=null`. Bibliographic metadata (title, agency, document
number, issue date, ...) is still populated on those rows so
consumers retain the citation handle even when no body is
available. Per-doc
embedding parquets for the NULL-markdown rows are dropped from
the embedding corpus entirely so the reducer fits on the
embeddable rows only ({_format_int(embeddable_corpus_size)}-row corpus).

Captured: `{manifest.get('completed_at')}`.

## Nguồn · Source

* Portal: <https://vbpl.vn/>
* Backend gateway: `vbpl-bientap-gateway.moj.gov.vn`
* Publisher: Ministry of Justice of Vietnam (Bộ Tư pháp)
* Sitemap: <https://vbpl.vn/sitemap.xml>

## Giấy phép · License

Văn bản gốc được Bộ Tư pháp công bố trên cổng thông tin công cộng
(`Allow: /` trong `robots.txt`). Bản phân phối lại này dùng giấy
phép **{license_id.upper()}**; vui lòng kiểm tra điều khoản sử
dụng của trang nguồn trước khi tái phân phối thương mại. — The
source documents are published by the Ministry of Justice on a
public portal (its `robots.txt` allows `/` and disallows only
`/api/`). This redistribution is shared under
**{license_id.upper()}**; please check the source-website terms of
use before commercial redistribution.

## Trích dẫn · Citation

If you use this dataset, please cite both **the redistribution on
Hugging Face** and **the original source** (Cơ sở dữ liệu Quốc gia
về pháp luật, Bộ Tư pháp Việt Nam):

```bibtex
@misc{{vbpl_2026,
  title        = {{Vietnamese National Legal Database (vbpl.vn)}},
  author       = {{TMQuan}},
  year         = {{2026}},
  howpublished = {{\url{{https://huggingface.co/datasets/{repo_owner}/{repo_name}}}}},
  note         = {{Document-level mirror with a hierarchical structure layer (DocumentMeta + Section + Paragraph + Sentence) over Vietnam's National Legal Database, central + provincial scope.}}
}}

@misc{{vbpl_moj_2026,
  title        = {{Vietnamese National Legal Database}},
  author       = {{{{Cơ sở dữ liệu Quốc gia về pháp luật}}}},
  year         = {{2026}},
  howpublished = {{\url{{https://vbpl.vn/}}}},
  note         = {{Official portal for Vietnam's National Legal Database (laws, ordinances, decrees, circulars, decisions, ...) at central and provincial levels, published by the Ministry of Justice (Bộ Tư pháp).}}
}}
```
"""
    return front + body


# ----------------------------------------------------- entry points


def export(
    jsonl_dir: Path,
    out_dir: Path,
    *,
    reduced_dir: Path = DEFAULT_REDUCED_DIR,
    license_id: str = DEFAULT_LICENSE,
    repo_owner: str = DEFAULT_REPO_OWNER,
    repo_name: str = DEFAULT_REPO_NAME,
) -> dict[str, Path]:
    """Materialise the HF folder. Returns the paths it produced.

    Reads the raw per-doc tier (``<jsonl_dir>/<doc_name>.jsonl``,
    one record per file -- the canonical extract output per
    wiki/DATASITES.md §3.5), projects each record through
    :func:`_project_record`, and writes the
    ``documents-NNNNN-of-KKKKK.parquet`` consumption tier under
    ``out_dir`` alongside the dataset card / manifest / figures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not jsonl_dir.is_dir():
        raise FileNotFoundError(
            f"jsonl dir missing: {jsonl_dir}. Run --pipeline extract first.",
        )

    rows: list[dict[str, Any]] = []
    raw_total = 0
    for projected, _raw in _iter_projected(jsonl_dir):
        raw_total += 1
        if projected is not None:
            rows.append(projected)

    logger.info(
        "projected %d/%d rows (dropped %d empty-markdown)",
        len(rows), raw_total, raw_total - len(rows),
    )
    if not rows:
        raise FileNotFoundError(
            f"no usable JSONL records under {jsonl_dir}/*.jsonl "
            f"(every row had empty markdown). Run the parse + "
            f"extract pipelines on a host where the detail stage "
            f"actually fetched bodies.",
        )

    shard_paths = _write_sharded_parquet(
        rows,
        out_dir=out_dir,
        chunk_size=CHUNK_SIZE,
        row_group_size=PARQUET_ROW_GROUP_SIZE,
    )

    manifest = _build_manifest(rows, raw_total=raw_total)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s", manifest_path)

    # Sniff the actual embedder model + dim from the reduced parquet
    # so the card reflects what shipped (mpnet preview vs nemotron).
    # Done before rendering so the figure titles get the right model.
    embed_model_id, embed_dim = _read_embedder_info(reduced_dir)
    if embed_model_id:
        logger.info(
            "embed model from reduced parquet: %s (dim=%s)",
            embed_model_id, embed_dim,
        )

    # Render every overview + embedding figure via plotly+kaleido.
    # Returns {} (and warns) when chromium can't be discovered; the
    # rest of the export still ships in that case.
    viz_paths = _render_figures(
        rows,
        manifest,
        reduced_dir,
        out_dir,
        embed_model_id=embed_model_id,
        embed_dim=embed_dim,
    )

    readme_path = out_dir / "README.md"
    readme_path.write_text(
        _render_card(
            manifest, repo_owner, repo_name, license_id,
            viz_paths=viz_paths,
            embed_model_id=embed_model_id,
            embed_dim=embed_dim,
        ),
        encoding="utf-8",
    )
    logger.info(
        "wrote dataset card: %s (%d bytes)",
        readme_path, readme_path.stat().st_size,
    )

    paths: dict[str, Path] = {
        "manifest":  manifest_path,
        "readme":    readme_path,
    }
    for i, sp in enumerate(shard_paths):
        paths[f"documents_shard_{i:05d}"] = sp
    for label, p in viz_paths.items():
        paths[f"viz_{label}"] = p
    return paths


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description=(
            "Materialise the vbpl raw per-doc jsonl tier into an "
            "HF-ready folder."
        ),
    )
    parser.add_argument("--jsonl-dir",   type=Path, default=DEFAULT_JSONL_DIR,
                        help="directory holding the per-doc <doc>.jsonl files")
    parser.add_argument("--reduced-dir", type=Path, default=DEFAULT_REDUCED_DIR)
    parser.add_argument("--out-dir",     type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--license",     default=DEFAULT_LICENSE)
    parser.add_argument("--repo-owner",  default=DEFAULT_REPO_OWNER)
    parser.add_argument("--repo-name",   default=DEFAULT_REPO_NAME)
    args = parser.parse_args(argv)

    paths = export(
        jsonl_dir=args.jsonl_dir,
        reduced_dir=args.reduced_dir,
        out_dir=args.out_dir,
        license_id=args.license,
        repo_owner=args.repo_owner,
        repo_name=args.repo_name,
    )
    print("HF folder ready:")
    for k, p in paths.items():
        print(f"  {k:24s} -> {p}  ({p.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
