"""CLI entry for the two-stage pbgdpl crawler.

    # Run both stages (harvest -> detail) end-to-end.
    python -m packages.datasites.pbgdpl --pipeline all

    # Just walk the listings + LinhVuc taxonomy (cheap, ~5-10 minutes
    # at the default 2 QPS; produces listings.jsonl + taxonomy.json).
    python -m packages.datasites.pbgdpl --pipeline harvest

    # Fetch every Q&A detail (slow, ~15 minutes at 2 QPS for ~4600
    # items; resumable from the items HTML cache).
    python -m packages.datasites.pbgdpl --pipeline detail

    # Bounded smoke test (only fetch 10 detail pages).
    python -m packages.datasites.pbgdpl --pipeline detail --limit 10

    # Skip the per-LinhVuc walk during harvest (faster, but the
    # resulting listings.jsonl will have empty lv_ids per row).
    python -m packages.datasites.pbgdpl --pipeline harvest \\
        --override scraper.walk_lv=false

Unlike the anle / congbobanan datasites this crawler does NOT run
under nemo_curator's :class:`Pipeline` / executor stack: pbgdpl serves
HTML-only Q&A pairs with no PDF / OCR step, so the heavyweight Ray
plumbing is unnecessary. Concurrency comes from a thread pool sharing
one rate-limited :class:`packages.common.PoliteSession`.

All real work is delegated to
:func:`packages.common.runner.run_crawler_site`; this module only
encodes the per-site pipeline registry + module wiring.
"""

from __future__ import annotations

import sys

from packages.common.runner import run_crawler_site
from packages.datasites.pbgdpl.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_crawler_site(
        site="pbgdpl",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        run_pipeline=run_pipeline,
        description="Run the pbgdpl crawler.",
        pipeline_help=(
            "Which stage to run. 'all' runs harvest -> detail in sequence; "
            "individual names re-run one stage against the prior stage's "
            "on-disk output."
        ),
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
