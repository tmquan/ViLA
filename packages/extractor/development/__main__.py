"""CLI entry point for the case-development builder.

Usage::

    python -m packages.extractor.development \\
        --canonical-dir data/samplebanan.toaan.gov.vn/entities/canonical \\
        --md-dir        data/samplebanan.toaan.gov.vn/md \\
        --output        data/samplebanan.toaan.gov.vn \\
        [--limit N]
        [--config PATH]
        [--built-at ISO8601]   # pin built_at for byte-stable runs
        [--log-level LVL]

See ``wiki/DEVELOPMENT.md § 7`` for the reproduction recipe.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from omegaconf import OmegaConf

from packages.extractor.development.build import (
    aggregate_developments_jsonl,
    build_one,
    list_doc_names,
)

logger = logging.getLogger("packages.extractor.development")


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m packages.extractor.development",
        description=(
            "Build deterministic case-development records from "
            "canonical NER records for procedural-arc analytics."
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
        help=(
            "Output root; writes development/ + developments.jsonl "
            "(overrides config)."
        ),
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

    logger.info("canonical_dir = %s", canonical_dir)
    logger.info("md_dir        = %s", md_dir)
    logger.info("output_root   = %s", output_root)

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

    n_phases_total = 0
    n_routed_total = 0
    n_unrouted_total = 0
    n_meta_intro_total = 0
    n_main_intro_total = 0
    for doc_name in doc_names:
        dev = build_one(
            doc_name=doc_name,
            canonical_dir=canonical_dir,
            md_dir=md_dir,
            output_root=output_root,
            built_at=built_at,
        )
        n_phases_total += dev.stats.n_phases
        n_routed_total += dev.stats.n_entities_routed
        n_unrouted_total += dev.stats.n_unrouted
        n_meta_intro_total += dev.stats.n_metadata_introduced
        n_main_intro_total += dev.stats.n_maindata_introduced

    out = aggregate_developments_jsonl(
        output_root=output_root, doc_names=doc_names,
    )
    logger.info("developments.jsonl: %s (%d docs)", out, len(doc_names))
    logger.info(
        "totals: phases=%d routed=%d unrouted=%d "
        "meta_introduced=%d main_introduced=%d",
        n_phases_total,
        n_routed_total,
        n_unrouted_total,
        n_meta_intro_total,
        n_main_intro_total,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
