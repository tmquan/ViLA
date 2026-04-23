"""Orchestrator for the congbobanan 6-stage pipeline.

Identical policy to ``packages.scrapers.anle.run``: each stage is a
standalone, resume-aware script; this module just calls each stage's
``main(argv)`` in sequence and honors ``--start-from`` / ``--stop-after``.

Only stage 1 (scrape) has a congbobanan-specific implementation -- the
site is ID-enumerated, not listing-walked. Stages 2-6 (parse, extract,
embed, reduce, visualize) are site-agnostic and reused verbatim from
``packages.scrapers.anle``. The anle-specific vila.precedents
normalization (extractor "layer 2") is gated by
``cfg.extractor.run_site_layer`` and left enabled by default in the
anle config but disabled in ``configs/default.yaml`` here -- running
the site layer would misapply anle rules to court-judgment records.

Run:
    python -m packages.scrapers.congbobanan.run --config-name congbobanan
    python -m packages.scrapers.congbobanan.run --config-name congbobanan --stop-after parse
    python -m packages.scrapers.congbobanan.run --config-name congbobanan --start-from embed
    python -m packages.scrapers.congbobanan.run --config-name congbobanan --force
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

from packages.scrapers.anle import (
    embedder,
    extractor,
    parser as anle_parser,
    reducer,
    visualizer,
)
from packages.scrapers.common.cli import apply_log_level, build_arg_parser
from packages.scrapers.congbobanan import scraper as congbo_scraper

logger = logging.getLogger(__name__)


STAGES: list[tuple[str, Callable[[list[str]], int]]] = [
    ("scrape", congbo_scraper.main),
    ("parse", anle_parser.main),
    ("extract", extractor.main),
    ("embed", embedder.main),
    ("reduce", reducer.main),
    ("visualize", visualizer.main),
]

# Stages 2-6 live under packages.scrapers.anle and default to
# --config-name anle when neither --config nor --config-name is given.
# If the operator only passes --config-name congbobanan, translate that
# into an absolute --config path so the anle stage scripts pick it up.
_CONGBO_CONFIGS_DIR = Path(__file__).parent / "configs"


def main(argv: list[str] | None = None) -> int:
    arg_parser = build_arg_parser(
        description="Orchestrator for the congbobanan 6-stage pipeline.",
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

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    raw_argv = _strip_arg(raw_argv, "--start-from")
    raw_argv = _translate_config_name(raw_argv, _CONGBO_CONFIGS_DIR)

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


# ----------------------------------------------------------------- argv utils


def _strip_arg(argv: list[str], name: str) -> list[str]:
    """Remove a '--foo value' pair (or '--foo=value') from an argv list."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == name:
            i += 2
            continue
        if a.startswith(name + "="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _translate_config_name(argv: list[str], configs_dir: Path) -> list[str]:
    """Rewrite ``--config-name X`` into ``--config <configs_dir>/X.yaml``.

    Required because stages 2-6 are imported from
    ``packages.scrapers.anle``; their ``CONFIGS_DIR`` points at anle's
    config folder, so ``--config-name congbobanan`` would resolve to
    ``packages/scrapers/anle/configs/congbobanan.yaml`` (which does not
    exist). Pinning ``--config`` to an absolute path is unambiguous.
    """
    if "--config" in argv or any(a.startswith("--config=") for a in argv):
        return argv
    out: list[str] = []
    i = 0
    consumed_name: str | None = None
    while i < len(argv):
        a = argv[i]
        if a == "--config-name":
            consumed_name = argv[i + 1] if i + 1 < len(argv) else None
            i += 2
            continue
        if a.startswith("--config-name="):
            consumed_name = a.split("=", 1)[1]
            i += 1
            continue
        out.append(a)
        i += 1
    if consumed_name:
        cfg_path = (configs_dir / f"{consumed_name}.yaml").resolve()
        out.extend(["--config", str(cfg_path)])
    return out


if __name__ == "__main__":
    sys.exit(main())
