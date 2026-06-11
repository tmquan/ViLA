"""CLI entry for the five-pipeline luutru curation flow.

    # Run all five sequentially (download -> parse -> extract -> embed -> reduce)
    python -m packages.datasites.luutru --pipeline all --executor xenna --limit 3

    # Re-run a single stage against existing inputs on disk
    python -m packages.datasites.luutru --pipeline extract
    python -m packages.datasites.luutru --pipeline embed --executor ray_actor_pool
    python -m packages.datasites.luutru --pipeline reduce

    # Attach to a remote Ray cluster
    python -m packages.datasites.luutru --pipeline extract \\
        --executor ray_actor_pool --ray-address ray://head:10001

Each pipeline shares the same executor + Ray init; Ray is torn down
only when we started it locally. All real work is delegated to
:func:`packages.common.runner.run_curator_site`; this module only
encodes the per-site pipeline registry + module wiring.
"""

from __future__ import annotations

import sys

from packages.common.runner import run_curator_site
from packages.datasites.luutru.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    return run_curator_site(
        site="luutru",
        pipelines=PIPELINES,
        all_order=ALL_PIPELINES_ORDER,
        build_pipeline=build_pipeline,
        description="Run the luutru curation pipelines.",
        pipeline_help=(
            "Which pipeline to run. 'all' runs download -> parse -> extract "
            "-> embed -> reduce in sequence; individual names re-run one "
            "step against the prior step's on-disk output."
        ),
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
