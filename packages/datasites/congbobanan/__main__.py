"""CLI entry for the five-pipeline congbobanan curation flow.

    # Run all five sequentially (download -> parse -> extract -> embed -> reduce)
    python -m packages.datasites.congbobanan --pipeline all --executor xenna --limit 100

    # Re-run a single stage against existing inputs on disk
    python -m packages.datasites.congbobanan --pipeline embed
    python -m packages.datasites.congbobanan --pipeline reduce

    # Bounded smoke test
    python -m packages.datasites.congbobanan --pipeline download \\
        --override scraper.start_id=1 scraper.end_id=10

    # Remote Ray cluster (VN-egress mandatory: the host rejects non-VN IPs)
    python -m packages.datasites.congbobanan --pipeline all \\
        --executor ray_actor_pool --ray-address ray://head:10001 \\
        --override scraper.proxy=http://vn-egress:3128

All real work is delegated to
:func:`packages.common.runner.run_curator_site`; this module only
encodes the per-site pipeline registry + module wiring.
"""

from __future__ import annotations

import sys

from packages.common.runner import run_curator_site
from packages.datasites.congbobanan.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_curator_site(
        site="congbobanan",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        build_pipeline=build_pipeline,
        description="Run the congbobanan curation pipelines.",
        pipeline_help=(
            "Which pipeline to run. 'all' runs download -> parse -> extract "
            "-> embed -> reduce in sequence; individual names re-run one "
            "step against the prior step's on-disk output."
        ),
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
