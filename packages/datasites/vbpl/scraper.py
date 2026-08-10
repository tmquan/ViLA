"""Top-level pipeline dispatch for the vbpl crawler.

Six stages, run by name:

    harvest -- walk https://vbpl.vn/sitemap.xml + every child shard,
               write jsonl/sitemap.jsonl (one row per detail-page URL).
    detail  -- read sitemap.jsonl, drive headless Chromium against
               every URL, intercept the authenticated /api/qtdc/...
               XHR responses, write jsonl/docs.jsonl + cached
               html/<scope>/<id>.html (+ optional pdf/<scope>/<id>.*).
    parse   -- read docs.jsonl + downloaded files, parse PDFs/.doc/.docx
               (pypdf + antiword/catdoc fallbacks for .doc) and convert
               body_html to markdown via markdownify; write
               md/<scope>/<id>.md + sibling <id>.meta.json.
    extract -- read md + meta through a Curator pipeline, run the
               declarative cfg.extractor.normalizers chain
               (wiki/DATASITES.md §3.5) + GenericExtractor +
               LegalStructureExtractor; write the raw per-doc tier
               (jsonl/<doc>.jsonl, one file per doc) and then
               coalesce into the parquet consumption tier
               (parquet/extract/extract-NNNNN-of-KKKKK.parquet).
    embed   -- read parquet/extract shards, embed ``markdown`` via
               cfg.embedder.runtime (NIM by default; HF on GPU as
               alternative); write the parquet consumption tier
               (parquet/embed/embed-NNNNN-of-KKKKK.parquet).
    reduce  -- read parquet/embed shards, fit PCA + t-SNE + UMAP +
               HDBSCAN over the full embedding matrix; write
               parquet/reduce/reduce-NNNNN-of-KKKKK.parquet.
    all     -- run all six in order (default).

Stages are decoupled so a partial run resumes cheaply: each stage
short-circuits on the on-disk output of the previous one. Re-running
an earlier stage with the same ``--limit`` is idempotent because
each stage skips already-produced outputs (raw per-doc tier:
filename-level via ``mode="ignore"``; parquet consumption tier:
shard-level, see wiki/DATASITES.md §10).

``harvest`` / ``detail`` / ``parse`` run in-process (no Ray);
``extract`` / ``embed`` / ``reduce`` build a
:class:`nemo_curator.pipeline.Pipeline` and dispatch through the
shared executor / Ray bootstrap. Each Curator pipeline opens and
tears down its own Ray context (idempotent -- ``init_ray`` is a
no-op when Ray is already up via ``ignore_reinit_error=True``) so a
single ``--pipeline all`` run shares one Ray cluster.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from packages.common.io import (
    coalesce_jsonl_to_parquet_shards,
    coalesce_per_doc_parquet_to_shards,
    resolve_doc_chunk_size,
    resolve_row_group_size,
)
from packages.datasites.vbpl._shared import (
    EXTRACTOR_JSONL_FIELDS,
    build_layout,
)
from packages.datasites.vbpl.components import (
    VbplDetailDownloader,
    VbplDetailRebuilder,
    VbplDocumentParser,
    VbplListingHarvester,
    VbplSitemapHarvester,
)
from packages.datasites.vbpl.embed import build_embed_pipeline
from packages.datasites.vbpl.extract import build_extract_pipeline
from packages.datasites.vbpl.reduce import build_reduce_pipeline

logger = logging.getLogger(__name__)


PIPELINE_NAMES = ("harvest", "detail", "parse", "extract", "embed", "reduce")
ALL_PIPELINES_ORDER = list(PIPELINE_NAMES)


def run_harvest(cfg: Any) -> Path:
    """Walk every sitemap shard, write sitemap.jsonl. Returns its path."""
    layout = build_layout(cfg)
    harv = VbplSitemapHarvester(cfg, layout)
    sitemap_path, total = harv.run()
    logger.info(
        "harvest complete: rows=%d sitemap_jsonl=%s",
        total,
        sitemap_path,
    )
    return sitemap_path


def run_harvest_spa(cfg: Any) -> Path:
    """SPA listing harvest for the post-2026 reCAPTCHA-gated Next.js UI.

    Replacement for the dead sitemap harvest (vbpl.vn stopped publishing
    per-document sitemap URLs). Drives Chromium through the reCAPTCHA v3
    flow and scrapes document links from the rendered listing, writing
    the SAME ``sitemap.jsonl`` the detail stage consumes. Requires a
    display for reCAPTCHA to pass -- see
    :class:`packages.datasites.vbpl.components.listing_harvester.VbplListingHarvester`
    (run under ``xvfb-run -a`` or with ``scraper.headless=false``).
    """
    layout = build_layout(cfg)
    harv = VbplListingHarvester(cfg, layout)
    sitemap_path, total = harv.run()
    logger.info(
        "harvest_spa complete: rows=%d sitemap_jsonl=%s", total, sitemap_path,
    )
    return sitemap_path


def run_detail(cfg: Any) -> Path:
    """Drive Playwright per ItemID, write docs.jsonl. Returns its path."""
    layout = build_layout(cfg)
    return VbplDetailDownloader(cfg, layout).run()


def run_rebuild_docs(cfg: Any) -> Path:
    """Rebuild docs.jsonl from cached html/<scope>/<id>.api.json.

    Offline equivalent of :func:`run_detail`: no Playwright, no network.
    Replays :func:`detail_record_from_api_json` over every cached API JSON
    capture so docs.jsonl reflects the *current* mapping (e.g. when new
    metadata fields like ``issue_date`` / ``issuing_authority`` are
    added to the parser without re-running the slow browser fetch).

    The existing docs.jsonl is moved to ``docs.jsonl.bak-<timestamp>``
    before the new file is renamed into place atomically. Downstream
    ``parse`` / ``extract`` / ``embed`` / ``reduce`` then pick up the
    refreshed metadata on their next run.
    """
    layout = build_layout(cfg)
    return VbplDetailRebuilder(cfg, layout).run()


def run_parse(cfg: Any) -> Path:
    """Walk docs.jsonl + files, write per-item markdown + meta. Returns md_dir."""
    layout = build_layout(cfg)
    return VbplDocumentParser(cfg, layout).run()


def run_extract(cfg: Any) -> Path:
    """Curator extract pipeline + post-coalesce into parquet shards.

    Two output tiers per wiki/DATASITES.md §3.5:

    1. **Raw per-doc tier** — ``jsonl/<doc>.jsonl`` (one file per
       document, keyed by ``doc_name``). Written by
       :class:`packages.pipeline.io.JsonlPerDocWriter` inside the
       Curator pipeline.
    2. **Parquet consumption tier** —
       ``parquet/extract/extract-NNNNN-of-KKKKK.parquet``
       (``cfg.shards.doc_chunk_size`` rows per shard; vbpl ships
       at 5 K because rows are fat with ``structure_json`` +
       ``extracted_json``). Coalesced from the per-doc JSONL after
       the Curator pipeline returns.

    Returns the parquet directory so ``--pipeline all`` can feed it
    to ``run_embed``.
    """
    layout = build_layout(cfg)
    pipeline = build_extract_pipeline(cfg)
    _run_curator_pipeline(cfg, pipeline)
    out_dir = _coalesce_extract_shards(cfg, layout)
    return out_dir


def _coalesce_extract_shards(cfg: Any, layout: Any) -> Path:
    """Read jsonl/<doc>.jsonl shards and write parquet/extract/*.parquet."""
    doc_chunk_size = resolve_doc_chunk_size(cfg)
    row_group_size = resolve_row_group_size(cfg)
    jsonl_paths = sorted(layout.jsonl_dir.glob("*.jsonl"))
    # Drop the legacy single-file extract.jsonl + the staging
    # ``extract_shards/`` directory the prior pipeline shipped; the
    # canonical raw tier is now per-doc ``<doc>.jsonl`` files.
    jsonl_paths = [
        p for p in jsonl_paths
        if p.name not in {
            "sitemap.jsonl", "docs.jsonl",
            "extract.jsonl",
            "extract_manifest.json", "parse_manifest.json",
            "manifest.json",
        } and not p.name.startswith("extract.jsonl.")
    ]
    out_dir = layout.extract_parquet_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "coalesce extract: %d per-doc JSONL shards -> %s "
        "(doc_chunk_size=%d, row_group_size=%d)",
        len(jsonl_paths), out_dir, doc_chunk_size, row_group_size,
    )
    written = coalesce_jsonl_to_parquet_shards(
        jsonl_paths=jsonl_paths,
        out_dir=out_dir,
        stage="extract",
        fields=EXTRACTOR_JSONL_FIELDS,
        doc_chunk_size=doc_chunk_size,
        row_group_size=row_group_size,
    )
    logger.info(
        "coalesce extract: wrote %d parquet shards under %s",
        len(written), out_dir,
    )
    return out_dir


def run_embed(cfg: Any) -> Path:
    """Embed extract.jsonl rows -> embeddings parquet via Curator + Ray."""
    return _run_curator_pipeline(cfg, build_embed_pipeline(cfg)) or (
        build_layout(cfg).embeddings_dir
    )


def run_reduce(cfg: Any) -> Path:
    """Run PCA / t-SNE / UMAP + HDBSCAN over embeddings -> reduced parquet."""
    return _run_curator_pipeline(cfg, build_reduce_pipeline(cfg)) or (
        build_layout(cfg).reduced_dir
    )


def _run_curator_pipeline(cfg: Any, pipeline: Any) -> Path | None:
    """Bootstrap Ray + an executor, run a Curator Pipeline, tear down.

    The Ray init / shutdown is idempotent across multiple Curator
    stages in one process (Ray ignores re-init via
    ``ignore_reinit_error=True``) so ``--pipeline all`` can run
    embed + reduce back-to-back without reinit cost.
    """
    from packages.pipeline import build_executor, init_ray, shutdown_ray

    init_ray(cfg)
    try:
        logger.info("=== curator pipeline %s ===\n%s",
                    pipeline.name, pipeline.describe())
        executor = build_executor(cfg)
        results = pipeline.run(executor=executor)
        logger.info(
            "curator pipeline %s finished: %d output tasks",
            pipeline.name, len(results or []),
        )
    finally:
        # Only tear down when this process owns the Ray cluster.
        # Connecting to a remote ray://... cluster keeps Ray alive.
        if not cfg.ray.get("address"):
            shutdown_ray()
    return None


def run_rechunk(cfg: Any) -> Path:
    """Coalesce legacy per-doc parquet -> consumption-tier shards.

    One-shot migration step (wiki/DATASITES.md §3.5 + §10) that reads the
    historical ``parquet/embeddings/<doc>.parquet`` +
    ``parquet/reduced/<doc>.parquet`` per-document files (one parquet
    per row, the legacy "raw + consumption tiers fused" shape) and
    re-writes them as ``parquet/embed/embed-NNNNN-of-KKKKK.parquet``
    + ``parquet/reduce/reduce-NNNNN-of-KKKKK.parquet`` shards of
    ``cfg.shards.doc_chunk_size`` rows each (5 K for vbpl --
    configs/default.yaml).

    Runs serially (no Ray): each per-doc parquet is a single row, so
    reading 147 K of them is I/O-bound and a single process saturates
    the disk faster than scheduling Ray actors over them. Skips a
    side cleanly if the source directory is empty (already migrated).

    Returns the embed shard directory (``parquet/embed/``); the
    reduce shards land next to it under ``parquet/reduce/``.
    """
    layout = build_layout(cfg)
    doc_chunk_size = resolve_doc_chunk_size(cfg)
    row_group_size = resolve_row_group_size(cfg)

    for tier in ("embed", "reduce"):
        src = layout.embeddings_dir if tier == "embed" else layout.reduced_dir
        dst = (
            layout.embed_parquet_dir if tier == "embed"
            else layout.reduce_parquet_dir
        )
        src_count = len(list(src.glob("*.parquet"))) if src.exists() else 0
        if src_count == 0:
            logger.info(
                "rechunk %s: no per-doc parquet under %s; skipping",
                tier, src,
            )
            continue
        logger.info(
            "rechunk %s: %d per-doc parquet files in %s "
            "-> %s shards under %s (doc_chunk_size=%d)",
            tier, src_count, src, tier, dst, doc_chunk_size,
        )
        written = coalesce_per_doc_parquet_to_shards(
            per_doc_dir=src,
            out_dir=dst,
            stage=tier,
            doc_chunk_size=doc_chunk_size,
            row_group_size=row_group_size,
        )
        logger.info(
            "rechunk %s: wrote %d shards under %s", tier, len(written), dst,
        )
    return layout.embed_parquet_dir


PIPELINES: dict[str, Callable[[Any], Path]] = {
    "harvest": run_harvest,
    "harvest_spa": run_harvest_spa,
    "detail": run_detail,
    "rebuild_docs": run_rebuild_docs,
    "parse": run_parse,
    "extract": run_extract,
    "embed": run_embed,
    "reduce": run_reduce,
    "rechunk": run_rechunk,
}


def run_pipeline(cfg: Any, name: str) -> Path:
    """Dispatch to the named pipeline. ``name='all'`` is handled in __main__."""
    if name not in PIPELINES:
        raise ValueError(
            f"unknown pipeline {name!r}; choices: {list(PIPELINES) + ['all']}"
        )
    return PIPELINES[name](cfg)


__all__ = [
    "ALL_PIPELINES_ORDER",
    "PIPELINES",
    "PIPELINE_NAMES",
    "run_detail",
    "run_embed",
    "run_extract",
    "run_harvest",
    "run_harvest_spa",
    "run_parse",
    "run_pipeline",
    "run_rebuild_docs",
    "run_rechunk",
    "run_reduce",
]
