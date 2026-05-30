"""Top-level pipeline dispatch for the thuvienphapluat_banan crawler.

Six stages, run by name (mirrors the vbpl hybrid contract,
wiki/DATASITES.md §13.4):

    harvest  -- walk /banan/tim-ban-an?page=N pages, write
                jsonl/listings.jsonl + jsonl/taxonomy.json.
    detail   -- read listings.jsonl, fetch /banan/ban-an/x-<id> for
                every row (slugless shortcut → canonical slug-URL via
                a redirect), parse sidebar + body, write jsonl/docs.jsonl.
    parse    -- read docs.jsonl, convert body_html → markdown via
                markdownify, write md/<ban_an_id>.md + .meta.json.
    extract  -- Curator pipeline: read md/*.md, run NormalizerChainStage
                + LegalExtractStage, write the raw per-doc tier
                (jsonl/<doc>.jsonl) and coalesce into the parquet
                consumption tier (parquet/extract/extract-*.parquet).
    embed    -- Curator pipeline: read parquet/extract shards, embed
                ``markdown`` via cfg.embedder.runtime (NIM default),
                write parquet/embed/embed-*.parquet.
    reduce   -- Curator pipeline: read parquet/embed shards, fit PCA +
                t-SNE + UMAP + HDBSCAN over the full embedding matrix,
                write parquet/reduce/reduce-*.parquet.
    all      -- run all six in order (default).

Stages are decoupled so a partial run resumes cheaply: each stage
short-circuits on the on-disk output of the previous one. Re-running
an earlier stage with the same ``--limit`` is idempotent because each
stage skips already-produced outputs.

``harvest`` / ``detail`` / ``parse`` run in-process (no Ray); the
remaining three build a :class:`nemo_curator.pipeline.Pipeline` and
dispatch through the shared executor / Ray bootstrap. Each Curator
pipeline opens and tears down its own Ray context (idempotent —
``init_ray`` is a no-op when Ray is already up via
``ignore_reinit_error=True``) so a single ``--pipeline all`` run
shares one Ray cluster.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from packages.common.io import (
    coalesce_jsonl_to_parquet_shards,
    resolve_doc_chunk_size,
    resolve_row_group_size,
)
from packages.datasites.thuvienphapluat_banan._shared import (
    EXTRACTOR_JSONL_FIELDS,
    build_layout,
)
from packages.datasites.thuvienphapluat_banan.components import (
    BananDetailDownloader,
    BananDocumentParser,
    BananHarvester,
)
from packages.datasites.thuvienphapluat_banan.embed import build_embed_pipeline
from packages.datasites.thuvienphapluat_banan.extract import (
    build_extract_pipeline,
)
from packages.datasites.thuvienphapluat_banan.reduce import (
    build_reduce_pipeline,
)

logger = logging.getLogger(__name__)


PIPELINE_NAMES = ("harvest", "detail", "parse", "extract", "embed", "reduce")
ALL_PIPELINES_ORDER = list(PIPELINE_NAMES)


# ---- in-process stages ---------------------------------------------------


def run_harvest(cfg: Any) -> Path:
    """Enumerate ban_an_id range, write listings.jsonl + taxonomy.json.

    Since 2026-05 the paginated ``/banan/tim-ban-an`` listing endpoint
    is permanently fronted by Cloudflare Turnstile and unreachable
    from headless clients. The harvester instead enumerates the
    integer ``ban_an_id`` space (sibling of thuvienphapluat_tnpl);
    the downstream detail stage fills sidebar metadata from the
    slugless detail HTML, and the parse stage runs on the embedded
    CDN PDFs.
    """
    layout = build_layout(cfg)
    harv = BananHarvester(cfg, layout)
    state = harv.run()
    lst_path, _ = harv.write_outputs(state)
    logger.info(
        "harvest complete: ids=%d range=[%d..%d] listings_jsonl=%s",
        len(state.listings), state.id_start, state.max_id, lst_path,
    )
    return lst_path


def run_detail(cfg: Any) -> Path:
    """Fetch + parse every detail page, write docs.jsonl."""
    layout = build_layout(cfg)
    return BananDetailDownloader(cfg, layout).run()


def run_parse(cfg: Any) -> Path:
    """Convert body_html → markdown, write per-doc md + meta. Returns md_dir."""
    layout = build_layout(cfg)
    return BananDocumentParser(cfg, layout).run()


# ---- Curator stages ------------------------------------------------------


def run_extract(cfg: Any) -> Path:
    """Curator extract pipeline + post-coalesce into parquet shards.

    Two output tiers per wiki/DATASITES.md §3.5:

    1. **Raw per-doc tier** — ``jsonl/<doc>.jsonl`` (one file per
       judgment, keyed by ``doc_name = ban_an_id``). Written by
       ``JsonlPerDocWriter`` inside the Curator pipeline.
    2. **Parquet consumption tier** —
       ``parquet/extract/extract-NNNNN-of-KKKKK.parquet``
       (``cfg.shards.doc_chunk_size`` rows per shard; thuvienphapluat_banan
       ships at the cross-corpus 10 K default). Coalesced from the
       per-doc JSONL after the Curator pipeline returns.

    Returns the parquet directory so ``--pipeline all`` can feed it
    to ``run_embed``.
    """
    layout = build_layout(cfg)
    pipeline = build_extract_pipeline(cfg)
    _run_curator_pipeline(cfg, pipeline)
    return _coalesce_extract_shards(cfg, layout)


def _coalesce_extract_shards(cfg: Any, layout: Any) -> Path:
    """Read jsonl/<doc>.jsonl shards and write parquet/extract/*.parquet."""
    doc_chunk_size = resolve_doc_chunk_size(cfg)
    row_group_size = resolve_row_group_size(cfg)
    jsonl_paths = sorted(layout.jsonl_dir.glob("*.jsonl"))
    # Drop the stage-control JSONL files (only per-doc shards count).
    jsonl_paths = [
        p for p in jsonl_paths
        if p.name not in {
            "listings.jsonl", "docs.jsonl",
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
    """Embed parquet/extract rows -> parquet/embed via Curator + Ray."""
    return _run_curator_pipeline(cfg, build_embed_pipeline(cfg)) or (
        build_layout(cfg).embed_parquet_dir
    )


def run_reduce(cfg: Any) -> Path:
    """Run PCA / t-SNE / UMAP + HDBSCAN over embeddings -> reduced parquet."""
    return _run_curator_pipeline(cfg, build_reduce_pipeline(cfg)) or (
        build_layout(cfg).reduce_parquet_dir
    )


def _run_curator_pipeline(cfg: Any, pipeline: Any) -> Path | None:
    """Bootstrap Ray + an executor, run a Curator Pipeline, tear down.

    The Ray init / shutdown is idempotent across multiple Curator stages
    in one process (Ray ignores re-init via ``ignore_reinit_error=True``)
    so ``--pipeline all`` can run embed + reduce back-to-back without
    reinit cost.
    """
    from packages.pipeline import build_executor, init_ray, shutdown_ray

    init_ray(cfg)
    try:
        logger.info(
            "=== curator pipeline %s ===\n%s",
            pipeline.name, pipeline.describe(),
        )
        executor = build_executor(cfg)
        results = pipeline.run(executor=executor)
        logger.info(
            "curator pipeline %s finished: %d output tasks",
            pipeline.name, len(results or []),
        )
    finally:
        if not cfg.ray.get("address"):
            shutdown_ray()
    return None


# ---- registry ------------------------------------------------------------


PIPELINES: dict[str, Callable[[Any], Path]] = {
    "harvest": run_harvest,
    "detail":  run_detail,
    "parse":   run_parse,
    "extract": run_extract,
    "embed":   run_embed,
    "reduce":  run_reduce,
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
    "run_parse",
    "run_pipeline",
    "run_reduce",
]
