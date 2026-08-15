"""CLI entry point for the timeline builder.

Usage::

    python -m packages.extractor.timeline \\
        --canonical-dir data/samplebanan.toaan.gov.vn/entities/canonical \\
        --md-dir        data/samplebanan.toaan.gov.vn/md \\
        --output        data/samplebanan.toaan.gov.vn \\
        [--limit N]
        [--window-chars 1500]
        [--config PATH]
        [--built-at ISO8601]   # pin built_at for byte-stable runs

See ``wiki/TIMELINE.md § 7`` for the reproduction recipe.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from nemo_curator.tasks import DocumentBatch
from omegaconf import OmegaConf

from packages.extractor.timeline.build import (
    TimelineBuildStage,
    aggregate_timelines_jsonl,
    list_doc_names,
)

logger = logging.getLogger("packages.extractor.timeline")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m packages.extractor.timeline",
        description=(
            "Build deterministic case timelines from canonical NER "
            "records for visual analytics."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "default.yaml",
        help="YAML config file (default: configs/default.yaml).",
    )
    p.add_argument(
        "--canonical-dir",
        type=Path,
        help="Directory of canonical NER per-doc JSON files (overrides config).",
    )
    p.add_argument(
        "--md-dir",
        type=Path,
        help="Directory of source <doc>.md files (overrides config).",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="Output root for timelines/ + timelines.jsonl (overrides config).",
    )
    p.add_argument(
        "--window-chars",
        type=int,
        help="Cluster window in characters (overrides config).",
    )
    p.add_argument(
        "--built-at",
        type=str,
        default=None,
        help=(
            "Pin the built_at timestamp (ISO 8601 UTC, e.g. "
            "2026-05-25T00:00:00Z) for byte-stable reruns. "
            "Default: current UTC."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N docs (lex order). Useful for smoke runs.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )
    return p


def _load_config(args: argparse.Namespace) -> OmegaConf:
    cfg = OmegaConf.load(args.config)
    if args.canonical_dir is not None:
        cfg.input.canonical_dir = str(args.canonical_dir)
    if args.md_dir is not None:
        cfg.input.md_dir = str(args.md_dir)
    if args.output is not None:
        cfg.output.root = str(args.output)
    if args.window_chars is not None:
        cfg.cluster.window_chars = int(args.window_chars)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    cfg = _load_config(args)
    canonical_dir = Path(cfg.input.canonical_dir)
    md_dir = Path(cfg.input.md_dir)
    output_root = Path(cfg.output.root)
    window_chars = int(cfg.cluster.window_chars)

    logger.info("canonical_dir = %s", canonical_dir)
    logger.info("md_dir        = %s", md_dir)
    logger.info("output_root   = %s", output_root)
    logger.info("window_chars  = %d", window_chars)

    if not canonical_dir.exists():
        logger.error(
            "canonical_dir does not exist: %s "
            "(run the NER pipeline first to materialise it)",
            canonical_dir,
        )
        return 2

    doc_names = list_doc_names(canonical_dir)
    if args.limit is not None:
        doc_names = doc_names[: args.limit]
    if not doc_names:
        logger.error(
            "no canonical records found under %s", canonical_dir,
        )
        return 2
    logger.info("doc_names: %d total", len(doc_names))

    built_at = args.built_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("built_at      = %s", built_at)

    stage = TimelineBuildStage(
        canonical_dir=canonical_dir,
        md_dir=md_dir,
        output_root=output_root,
        cluster_window_chars=window_chars,
        built_at=built_at,
    )
    batch = DocumentBatch(
        task_id="timeline_build",
        dataset_name="timeline",
        data=pd.DataFrame({"doc_name": doc_names}),
    )
    timelines = list(stage.process(batch).to_pandas()["timeline"])

    n_dated_total = sum(t.stats.n_dated for t in timelines)
    n_ambient_total = sum(t.stats.n_ambient for t in timelines)
    n_events_total = sum(t.stats.n_events for t in timelines)
    n_unlocated_total = sum(t.stats.n_unlocated_entities for t in timelines)

    out = aggregate_timelines_jsonl(output_root=output_root, doc_names=doc_names)
    logger.info("timelines.jsonl: %s (%d docs)", out, len(doc_names))
    logger.info(
        "totals: events=%d (dated=%d, ambient=%d) unlocated_entities=%d",
        n_events_total, n_dated_total, n_ambient_total, n_unlocated_total,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
