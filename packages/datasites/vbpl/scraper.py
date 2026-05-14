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
    extract -- read md + meta, run NFC + Vietnamese tone-mark
               normalization + GenericExtractor + LegalStructureExtractor;
               write jsonl/extract.jsonl with the canonical
               EXTRACTOR_JSONL_FIELDS schema.
    embed   -- read jsonl/extract.jsonl, embed `markdown` via the
               configured cfg.embedder.runtime (NIM by default; HF
               on GPU as alternative); write
               parquet/embeddings/<doc_name>.parquet.
    reduce  -- read parquet/embeddings, fit PCA + t-SNE + UMAP +
               HDBSCAN over the full embedding matrix; write
               parquet/reduced/<doc_name>.parquet with the projection
               coordinates and cluster_id columns.
    all     -- run all six in order (default).

Stages are decoupled so a partial run resumes cheaply: each stage
short-circuits on the on-disk output of the previous one. Re-running
an earlier stage with the same ``--limit`` is idempotent because
each stage skips already-produced outputs.

The first four stages run in-process (no Ray); ``embed`` and
``reduce`` build a :class:`nemo_curator.pipeline.Pipeline` and
dispatch through the shared executor / Ray bootstrap. Each Curator
pipeline opens and tears down its own Ray context (idempotent --
``init_ray`` is a no-op when Ray is already up via
``ignore_reinit_error=True``) so a single ``--pipeline all`` run
shares one Ray cluster.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from packages.datasites.vbpl._shared import build_layout
from packages.datasites.vbpl.components import (
    VbplDetailDownloader,
    VbplDocumentExtractor,
    VbplDocumentParser,
    VbplSitemapHarvester,
)
from packages.datasites.vbpl.embed import build_embed_pipeline
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


def run_detail(cfg: Any) -> Path:
    """Drive Playwright per ItemID, write docs.jsonl. Returns its path."""
    layout = build_layout(cfg)
    return VbplDetailDownloader(cfg, layout).run()


def run_parse(cfg: Any) -> Path:
    """Walk docs.jsonl + files, write per-item markdown + meta. Returns md_dir."""
    layout = build_layout(cfg)
    return VbplDocumentParser(cfg, layout).run()


def run_extract(cfg: Any) -> Path:
    """Run normalize + generic + structure layers; write extract.jsonl."""
    layout = build_layout(cfg)
    return VbplDocumentExtractor(cfg, layout).run()


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


PIPELINES: dict[str, Callable[[Any], Path]] = {
    "harvest": run_harvest,
    "detail": run_detail,
    "parse": run_parse,
    "extract": run_extract,
    "embed": run_embed,
    "reduce": run_reduce,
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
