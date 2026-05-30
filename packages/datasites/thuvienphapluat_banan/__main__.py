"""CLI entry for the six-stage thuvienphapluat_banan crawler.

    # Run every stage in order: harvest -> detail -> parse -> extract
    #                            -> embed -> reduce.
    python -m packages.datasites.thuvienphapluat_banan --pipeline all

    # Walk the paginated /banan/tim-ban-an surface, cache each page
    # under html/listings/, write listings.jsonl + taxonomy.json.
    python -m packages.datasites.thuvienphapluat_banan --pipeline harvest

    # Cheap smoke run: only walk the first 3 listing pages.
    python -m packages.datasites.thuvienphapluat_banan --pipeline harvest \\
        --override scraper.max_pages=3

    # Fetch every judgment detail via the slugless /banan/ban-an/x-<id>
    # shortcut. Resumable from the html/items/<id>.html caches.
    python -m packages.datasites.thuvienphapluat_banan --pipeline detail

    # Bounded smoke test (only fetch the first 20 detail pages).
    python -m packages.datasites.thuvienphapluat_banan --pipeline detail --limit 20

    # Convert body_html -> markdown via markdownify (in-process; no NIM
    # or PDF parsing, the portal renders the full judgment text inline).
    python -m packages.datasites.thuvienphapluat_banan --pipeline parse

    # Run NFC + Vietnamese-tone normalization + GenericExtractor +
    # LegalStructureExtractor over the markdown, write the raw per-doc
    # tier and coalesce the parquet consumption tier (Curator + Ray).
    python -m packages.datasites.thuvienphapluat_banan --pipeline extract

    # Embed parquet/extract rows -> parquet/embed/ (NIM by default;
    # requires NVIDIA_API_KEY).
    python -m packages.datasites.thuvienphapluat_banan --pipeline embed

    # PCA + t-SNE + UMAP + HDBSCAN on the embeddings -> parquet/reduce/.
    python -m packages.datasites.thuvienphapluat_banan --pipeline reduce

The harvest / detail / parse stages run in-process (no Ray;
:class:`packages.common.PoliteSession` + a thread pool sharing one
rate-limited bucket). The extract / embed / reduce stages are
:class:`nemo_curator.pipeline.Pipeline` instances dispatched through
the shared executor / Ray bootstrap, identical to vbpl. The
dispatcher in :mod:`packages.datasites.thuvienphapluat_banan.scraper`
opens and tears down a Ray context per Curator pipeline.

All real work is delegated to
:func:`packages.common.runner.run_crawler_site`; this module only
encodes the per-site pipeline registry + module wiring.
"""

from __future__ import annotations

import sys

from packages.common.runner import run_crawler_site
from packages.datasites.thuvienphapluat_banan.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_crawler_site(
        site="thuvienphapluat_banan",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        run_pipeline=run_pipeline,
        description="Run the thuvienphapluat_banan crawler.",
        pipeline_help=(
            "Which stage to run. 'all' runs harvest -> detail -> parse "
            "-> extract -> embed -> reduce; individual names re-run one "
            "stage against the prior stage's on-disk output."
        ),
        argv=argv,
        # extract + embed + reduce are Curator pipelines that honour
        # cfg.executor.name + cfg.ray.address.
        accept_ray_flags=True,
    )


if __name__ == "__main__":
    sys.exit(main())
