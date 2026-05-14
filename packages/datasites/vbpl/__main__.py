"""CLI entry for the six-stage vbpl crawler.

    # Run every stage in order: harvest -> detail -> parse -> extract
    #                             -> embed -> reduce.
    python -m packages.datasites.vbpl --pipeline all

    # Walk only the public sitemap (cheap, ~30 s; produces sitemap.jsonl).
    python -m packages.datasites.vbpl --pipeline harvest

    # Fetch every document detail via headless Chromium (slow; ~90 h
    # at the default 0.5 QPS x 2 workers for ~160 K docs; resumable
    # from the per-ItemID html cache).
    python -m packages.datasites.vbpl --pipeline detail

    # Parse every downloaded artefact (PDF/.docx/.doc) plus the
    # captured body_html from the API into per-doc markdown.
    python -m packages.datasites.vbpl --pipeline parse

    # Run NFC + Vietnamese-tone normalization + GenericExtractor +
    # LegalStructureExtractor over the markdown, write extract.jsonl.
    python -m packages.datasites.vbpl --pipeline extract

    # Embed extract.jsonl rows -> parquet/embeddings/<doc>.parquet.
    # Requires NVIDIA_API_KEY for the default cfg.embedder.runtime=nim.
    python -m packages.datasites.vbpl --pipeline embed

    # PCA + t-SNE + UMAP + HDBSCAN on the embeddings ->
    # parquet/reduced/<doc>.parquet. GPU-accelerated when cuML is on
    # the worker; CPU sklearn / umap-learn fallback otherwise.
    python -m packages.datasites.vbpl --pipeline reduce

    # Bounded smoke test (e.g. only process 10 docs end-to-end).
    python -m packages.datasites.vbpl --pipeline detail  --limit 10
    python -m packages.datasites.vbpl --pipeline parse   --limit 10
    python -m packages.datasites.vbpl --pipeline extract --limit 10

    # Central documents only (skip the 21 provincial sitemap shards).
    python -m packages.datasites.vbpl --pipeline all \\
        --override scraper.scopes='[trung_uong]'

The harvest / detail / parse / extract stages run in-process (no Ray;
:class:`packages.common.PoliteSession` for harvest, Playwright for
detail, ThreadPoolExecutor for parse + extract). The embed and
reduce stages are :class:`nemo_curator.pipeline.Pipeline` instances
dispatched through the shared executor / Ray bootstrap, identical to
anle / congbobanan; the dispatcher in
:mod:`packages.datasites.vbpl.scraper` opens and tears down a Ray
context per Curator pipeline so this CLI doesn't need ``--executor``
or ``--ray-address`` flags for the common single-machine case.

All real work is delegated to
:func:`packages.common.runner.run_crawler_site`; this module only
encodes the per-site pipeline registry + module wiring.
"""

from __future__ import annotations

import sys

from packages.common.runner import run_crawler_site
from packages.datasites.vbpl.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_crawler_site(
        site="vbpl",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        run_pipeline=run_pipeline,
        description="Run the vbpl.vn crawler.",
        pipeline_help=(
            "Which stage to run. 'all' runs harvest -> detail -> parse "
            "-> extract -> embed -> reduce; individual names re-run one "
            "stage against the prior stage's on-disk output."
        ),
        argv=argv,
        # vbpl's embed + reduce stages are Curator pipelines that
        # honour cfg.executor.name + cfg.ray.address.
        accept_ray_flags=True,
    )


if __name__ == "__main__":
    sys.exit(main())
