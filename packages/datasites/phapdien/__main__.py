"""CLI entry for the phapdien.moj.gov.vn crawler.

Examples::

    python -m packages.datasites.phapdien --pipeline all
    python -m packages.datasites.phapdien --pipeline detail --limit 10

All real work is delegated to
:func:`packages.common.runner.run_crawler_site`; this module only
encodes the per-site pipeline registry + module wiring.
"""

from __future__ import annotations

import sys

from packages.common.runner import run_crawler_site
from packages.datasites.phapdien.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_crawler_site(
        site="phapdien",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        run_pipeline=run_pipeline,
        description="Run the phapdien crawler.",
        pipeline_help="'all' runs tree -> detail; individual names re-run one stage.",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
