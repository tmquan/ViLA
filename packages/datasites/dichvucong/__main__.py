"""CLI entry for the dichvucong curation pipelines.

    # Crawl every search page, then flatten to per-procedure JSONL:
    python -m packages.datasites.dichvucong --pipeline all --executor xenna
    python -m packages.datasites.dichvucong --pipeline crawl --limit 5
    python -m packages.datasites.dichvucong --pipeline extract

    # Incremental freshness diff (run after extract):
    python -m packages.datasites.dichvucong.reconcile

All real work is delegated to
:func:`packages.common.runner.run_curator_site`; this module only
encodes the per-site pipeline registry + module wiring.
"""

from __future__ import annotations

import sys

from packages.common.runner import run_curator_site
from packages.datasites.dichvucong.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_curator_site(
        site="dichvucong",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        build_pipeline=build_pipeline,
        description="Run the dichvucong (Cổng Dịch vụ công Quốc gia) pipelines.",
        pipeline_help=(
            "Which pipeline to run. 'all' runs crawl -> extract; "
            "run `python -m packages.datasites.dichvucong.reconcile` "
            "afterwards for the incremental freshness diff."
        ),
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
