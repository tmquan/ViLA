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
from packages.datasites.pbgdpl.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_pipeline,
)

logger = logging.getLogger(__name__)

SITE = "pbgdpl"
_PIPELINE_CHOICES = [*PIPELINES.keys(), "all"]


def _build_parser() -> argparse.ArgumentParser:
    parser = build_arg_parser(description=f"Run the {SITE} crawler.")
    parser.add_argument(
        "--pipeline",
        default="all",
        choices=_PIPELINE_CHOICES,
        help=(
            "Which stage to run. 'all' runs harvest -> detail in sequence; "
            "individual names re-run one stage against the prior stage's "
            "on-disk output."
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
    # ``--executor`` / ``--ray-address`` are accepted for CLI symmetry
    # with the other datasites but ignored: this crawler does not
    # bootstrap Ray. We log a hint rather than silently dropping the
    # flags so a user trying to wire pbgdpl into their Ray launcher
    # notices the discrepancy.
    if args.executor or args.ray_address:
        logger.info(
            "ignoring --executor / --ray-address (pbgdpl runs in-process)",
        )
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

    rc = 0
    try:
        for name in selected:
            logger.info("=== pipeline %s ===", name)
            out_path = run_pipeline(cfg, name)
            logger.info("pipeline %s finished: %s", name, out_path)
    except Exception:
        logger.exception("pipeline run failed")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
