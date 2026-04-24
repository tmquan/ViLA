"""CLI entry for the four-pipeline anle curation flow.

    # Run all four sequentially (download -> extract -> embed -> reduce)
    python -m packages.datasites.anle --pipeline all --executor xenna --limit 3

    # Re-run a single stage against existing inputs on disk
    python -m packages.datasites.anle --pipeline embed --executor ray_actor_pool
    python -m packages.datasites.anle --pipeline reduce

    # Attach to a remote Ray cluster
    python -m packages.datasites.anle --pipeline extract \\
        --executor ray_actor_pool --ray-address ray://head:10001

Each pipeline shares the same executor + Ray init; Ray is torn down
only when we started it locally.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from packages.common import (
    apply_log_level,
    build_arg_parser,
    find_site_config,
    load_and_override,
)
from packages.common.schemas import PipelineCfg
from packages.datasites.anle.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_pipeline,
)
from packages.pipeline import build_executor, init_ray, shutdown_ray

logger = logging.getLogger(__name__)

SITE = "anle"
_PIPELINE_CHOICES = [*PIPELINES.keys(), "all"]


def _build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser(description=f"Run the {SITE} curation pipelines.")
    parser.add_argument(
        "--pipeline",
        default="all",
        choices=_PIPELINE_CHOICES,
        help=(
            "Which pipeline to run. 'all' runs download -> extract -> embed "
            "-> reduce in sequence; individual names re-run one step against "
            "the prior step's on-disk output."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    apply_log_level(args.log_level)

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else find_site_config(args.config_name or SITE)
    )

    overrides = list(args.override)
    if args.executor:
        overrides.append(f"executor.name={args.executor}")
    if args.ray_address:
        overrides.append(f"ray.address={args.ray_address}")
    if args.limit is not None:
        overrides.append(f"limit={args.limit}")
    if args.output:
        overrides.append(
            f"output_dir={str(Path(args.output).expanduser().resolve())}"
        )

    cfg = load_and_override(
        config_path=config_path,
        overrides=overrides,
        schema_cls=PipelineCfg,
    )

    selected: list[str] = (
        list(ALL_PIPELINES_ORDER) if args.pipeline == "all" else [args.pipeline]
    )
    logger.info("running pipelines: %s", selected)

    init_ray(cfg)
    rc = 0
    try:
        for name in selected:
            pipeline = build_pipeline(cfg, name)
            logger.info("=== pipeline %s ===\n%s", name, pipeline.describe())
            executor = build_executor(cfg)
            results = pipeline.run(executor=executor)
            logger.info(
                "pipeline %s finished: %d output tasks",
                name, len(results or []),
            )
    except Exception:
        logger.exception("pipeline run failed")
        rc = 1
    finally:
        if not cfg.ray.get("address"):
            shutdown_ray()
    return rc


if __name__ == "__main__":
    sys.exit(main())
