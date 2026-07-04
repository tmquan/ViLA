"""CLI for the dichvucong new-portal Playwright crawler.

    python -m packages.datasites.dichvucong --pipeline list --limit 200
    python -m packages.datasites.dichvucong --pipeline detail --limit 200
    python -m packages.datasites.dichvucong --pipeline all
"""

from __future__ import annotations

import sys

from packages.common.runner import run_crawler_site
from packages.datasites.dichvucong.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_crawler_site(
        site="dichvucong",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        run_pipeline=run_pipeline,
        description="Run the dichvucong new-portal (/api/v1) Playwright crawler.",
        pipeline_help="'all' runs list -> detail (full structured procedure detail).",
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
