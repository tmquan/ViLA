"""Shared helpers for the vbpl crawler.

Holds the output-path layout builder and the field lists used by the
sitemap harvester + detail writer so the JSONL schemas stay
consistent across the two stages.

vbpl docs come in two scopes:

* ``trung_uong`` -- central legal documents (~55 K, 11 sitemap shards)
* ``dia_phuong`` -- provincial legal documents (~105 K, 21 sitemap shards)

The on-disk layout under ``<output_dir>/<host>/`` separates the two so
a downstream pipeline (or a HF dataset card) can address either set
independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.common import SiteLayout, build_layout as _build_layout_common


#: Allowed values for ``cfg.scraper.scopes`` and the on-disk directory
#: names under ``html/`` and ``pdf/``.
SCOPES: tuple[str, ...] = ("trung_uong", "dia_phuong")


#: Sitemap JSONL columns emitted by :func:`packages.datasites.vbpl.scraper.run_harvest`.
SITEMAP_JSONL_FIELDS: list[str] = [
    "item_id",
    "scope",
    "slug",
    "url",
    "lastmod",
    "changefreq",
    "priority",
    "harvested_at",
]


#: Detail JSONL columns emitted by :func:`packages.datasites.vbpl.scraper.run_detail`.
#:
#: Order matches the canonical column order downstream readers
#: (``pyarrow.json.read_json`` / ``pandas.read_json(lines=True)``)
#: should see.
DETAIL_JSONL_FIELDS: list[str] = [
    "item_id",
    "scope",
    "source",
    "source_url",
    "api_url",
    "scraped_at",
    "scrape_run_id",
    "doc_type",
    "so_hieu",
    "ngay_ban_hanh",
    "co_quan_ban_hanh",
    "trich_yeu",
    "title",
    "body_html",
    "body_text",
    "body_char_len",
    "body_text_hash",
    "file_paths",
    "html_path",
    "fetch_status",
    "fetch_error",
]


#: Minimal JSONL columns the Embedder pipeline reads from the extract
#: output. Keeping the projection narrow keeps the per-batch payload
#: small in the JsonlReader. The rest of the row is left on disk and
#: re-joined later by ``doc_name`` + ``text_hash`` if needed.
EMBEDDER_JSONL_READ_FIELDS: list[str] = ["doc_name", "text_hash", "markdown"]


#: Parquet columns written by the Embedder pipeline. ``doc_name`` +
#: ``text_hash`` is the join key back to the extract JSONL.
EMBEDDER_PARQUET_FIELDS: list[str] = [
    "doc_name",
    "text_hash",
    "embedding",
    "embedding_dim",
    "embedding_model_id",
    "embedding_text_hash",
    "embedding_chunks_used",
    "embedding_chunking",
]


#: Parquet columns written by the Reducer pipeline. Superset of the
#: Embedder output plus the reducer coordinate columns and cluster id.
REDUCER_PARQUET_FIELDS: list[str] = [
    *EMBEDDER_PARQUET_FIELDS,
    "pca_x",
    "pca_y",
    "pca_z",
    "tsne_x",
    "tsne_y",
    "tsne_z",
    "umap_x",
    "umap_y",
    "umap_z",
    "cluster_id",
]


#: JSONL columns emitted by :func:`packages.datasites.vbpl.scraper.run_extract`.
#:
#: Order matches the canonical column order downstream consumers
#: (HF dataset card, embedder JSONL reader, retrieval index loader)
#: should see. Mirrors anle's :data:`EXTRACTOR_JSONL_FIELDS` shape so
#: the same downstream tooling works on either corpus, plus vbpl-
#: specific sidebar metadata (``so_hieu``, ``co_quan_ban_hanh``, ...).
EXTRACTOR_JSONL_FIELDS: list[str] = [
    # shared: source / IO bookkeeping
    "doc_name",
    "item_id",
    "scope",
    "source",
    "source_url",
    "api_url",
    "html_path",
    "md_path",
    "file_paths",
    # shared: parser output
    "markdown",
    "num_pages",
    "confidence",
    "parser_model",
    "parser_runtime",
    "body_source",
    "parsed_at",
    # shared: legal-extract output
    "text_hash",
    "char_len",
    "extracted",
    "structure",
    # vbpl sidebar metadata
    "title",
    "doc_type",
    "so_hieu",
    "ngay_ban_hanh",
    "co_quan_ban_hanh",
    "trich_yeu",
    # provenance
    "scrape_run_id",
    "parse_run_id",
    "extract_run_id",
    "extracted_at",
]


def build_layout(cfg: Any) -> SiteLayout:
    """Ensure every output directory exists and return the :class:`SiteLayout`.

    vbpl's data root layout under ``<output_dir>/<host>/`` is::

        html/sitemaps/<shard>.xml          # raw sitemap shards (cache)
        html/<scope>/<item_id>.html        # raw detail-page shells
        html/<scope>/<item_id>.api.json    # captured /api/qtdc/... JSON
        pdf/<scope>/<item_id>.{pdf,doc,docx}  # downloaded files (when present)
        md/<scope>/<item_id>.md            # parsed markdown
        md/<scope>/<item_id>.meta.json     # parse-stage sidecar metadata
        jsonl/sitemap.jsonl                # one row per detail URL
        jsonl/docs.jsonl                   # one row per document detail
        jsonl/extract.jsonl                # one row per extracted markdown
        jsonl/manifest.json / parse_manifest.json / extract_manifest.json
        logs/                              # reserved for run logs

    Uses the shared ``"html"`` layout profile (creates html / jsonl /
    logs / site_root) plus vbpl-specific subdirs for per-scope HTML,
    binary, and markdown caches.
    """
    layout = SiteLayout.from_cfg(cfg)
    extras = (layout.html_dir / "sitemaps", layout.md_dir)
    for scope in SCOPES:
        extras = extras + (
            layout.html_dir / scope,
            layout.pdf_dir / scope,
            layout.md_dir / scope,
        )
    return _build_layout_common(cfg, profile="html", extra_dirs=extras)


def sitemaps_dir(layout: SiteLayout) -> Path:
    """Cache dir for raw ``sitemap-*.xml`` shards."""
    return layout.html_dir / "sitemaps"


def scope_html_dir(layout: SiteLayout, scope: str) -> Path:
    """Per-scope detail HTML cache dir (``html/trung_uong/`` / ``html/dia_phuong/``)."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; valid: {SCOPES}")
    return layout.html_dir / scope


def scope_pdf_dir(layout: SiteLayout, scope: str) -> Path:
    """Per-scope binary file cache dir (``pdf/trung_uong/`` / ``pdf/dia_phuong/``)."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; valid: {SCOPES}")
    return layout.pdf_dir / scope


def scope_md_dir(layout: SiteLayout, scope: str) -> Path:
    """Per-scope markdown output dir (``md/trung_uong/`` / ``md/dia_phuong/``)."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; valid: {SCOPES}")
    return layout.md_dir / scope


__all__ = [
    "DETAIL_JSONL_FIELDS",
    "EMBEDDER_JSONL_READ_FIELDS",
    "EMBEDDER_PARQUET_FIELDS",
    "EXTRACTOR_JSONL_FIELDS",
    "REDUCER_PARQUET_FIELDS",
    "SCOPES",
    "SITEMAP_JSONL_FIELDS",
    "build_layout",
    "extract_jsonl_path",
    "scope_html_dir",
    "scope_md_dir",
    "scope_pdf_dir",
    "sitemaps_dir",
]


def extract_jsonl_path(layout: SiteLayout) -> Path:
    """Path of the consolidated extract.jsonl emitted by the extract stage."""
    return layout.jsonl_dir / "extract.jsonl"
