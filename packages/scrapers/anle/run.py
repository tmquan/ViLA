"""Orchestrator for the anle 6-stage pipeline.

Runs each stage in order, respects --stop-after, reuses the same config +
override rules as the individual stage scripts. Every stage is
resume-aware on its own progress file, so the orchestrator just calls
each stage's `main(argv)` in sequence with the forwarded CLI args.

Run:
    python -m packages.scrapers.anle.run --config-name anle
    python -m packages.scrapers.anle.run --stop-after reduce
    python -m packages.scrapers.anle.run --force   # redo everything
"""

from __future__ import annotations

import logging
import sys
from typing import Callable

from packages.scrapers.anle import (
    embedder,
    extractor,
    parser as anle_parser,
    reducer,
    scraper,
    visualizer,
)
from packages.scrapers.common.cli import apply_log_level, build_arg_parser

logger = logging.getLogger(__name__)


STAGES: list[tuple[str, Callable[[list[str]], int]]] = [
    ("scrape", scraper.main),
    ("parse", anle_parser.main),
    ("extract", extractor.main),
    ("embed", embedder.main),
    ("reduce", reducer.main),
    ("visualize", visualizer.main),
]


def main(argv: list[str] | None = None) -> int:
    # Use build_arg_parser so --stop-after / --config / --override / etc
    # are all recognized; we then forward raw argv to each stage.
    arg_parser = build_arg_parser(
        description="Orchestrator for the anle 6-stage pipeline.",
        stage="scrape",
    )
    arg_parser.add_argument(
        "--start-from",
        default=None,
        choices=[name for name, _ in STAGES],
        help="Begin the run at this stage (skips preceding stages).",
    )
    args = arg_parser.parse_args(argv)
    apply_log_level(args.log_level)

    # Forward every arg except --start-from (which the orchestrator consumes).
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    raw_argv = _strip_arg(raw_argv, "--start-from")

    started = args.start_from is None
    stop_after = args.stop_after

    for stage_name, stage_main in STAGES:
        if not started:
            if stage_name == args.start_from:
                started = True
            else:
                logger.info("skip %s (before --start-from)", stage_name)
                continue
        logger.info("==== stage: %s ====", stage_name)
        rc = stage_main(raw_argv)
        if rc != 0:
            logger.error("stage %s failed (rc=%d); stopping pipeline", stage_name, rc)
            return rc
        if stop_after and stage_name == stop_after:
            logger.info("reached --stop-after=%s; exiting", stop_after)
            break
    return 0


def _strip_arg(argv: list[str], name: str) -> list[str]:
    """Remove a '--foo value' pair (or '--foo=value') from an argv list."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == name:
            i += 2  # consume name + value
            continue
        if a.startswith(name + "="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


if __name__ == "__main__":
    sys.exit(main())
