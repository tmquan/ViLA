"""CLI entry for the three-stage thuvienphapluat_tnpl crawler.

    # Run all stages (harvest -> detail -> translate) end-to-end.
    python -m packages.datasites.thuvienphapluat_tnpl --pipeline all

    # Just walk the homepage + write listings.jsonl (~1 GET; seconds).
    python -m packages.datasites.thuvienphapluat_tnpl --pipeline harvest

    # Fetch every term detail (~2.4h at 2 QPS for ~17k probe ids;
    # resumable from html/items/<id>.html caches).
    python -m packages.datasites.thuvienphapluat_tnpl --pipeline detail

    # Bounded smoke test (only fetch 20 detail pages).
    python -m packages.datasites.thuvienphapluat_tnpl --pipeline detail --limit 20

    # Translate every row through the pinned NIM Nemotron model
    # (resumable from translations/<term_id>.json caches; requires
    # NVIDIA_API_KEY).
    python -m packages.datasites.thuvienphapluat_tnpl --pipeline translate

    # Pin a different translator model.
    python -m packages.datasites.thuvienphapluat_tnpl --pipeline translate \\
        --override translator.model_id=nvidia/llama-3.1-nemotron-70b-instruct

Like pbgdpl / phapdien this crawler does NOT run under nemo_curator's
:class:`Pipeline` / executor stack: tnpl serves HTML-only term pages
with no PDF / OCR step, so the Ray plumbing is unnecessary.
Concurrency comes from a thread pool sharing one rate-limited
:class:`packages.common.PoliteSession` (detail stage) or a thread
pool fanned over the NIM endpoint (translate stage).
"""

from __future__ import annotations

import sys

from packages.common.runner import run_crawler_site
from packages.datasites.thuvienphapluat_tnpl.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_crawler_site(
        site="thuvienphapluat_tnpl",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        run_pipeline=run_pipeline,
        description="Run the thuvienphapluat_tnpl crawler.",
        pipeline_help=(
            "Which stage to run. 'all' runs harvest -> detail -> translate "
            "in sequence; individual names re-run one stage against the "
            "prior stage's on-disk output."
        ),
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
