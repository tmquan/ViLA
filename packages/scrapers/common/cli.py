"""Shared CLI argument parser for every scraper stage.

Every stage script (scraper, parser, extractor, embedder, reducer,
visualizer) accepts this common set so operators see a consistent
interface. A stage may extend the parser with its own stage-specific
flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser(description: str, stage: str) -> argparse.ArgumentParser:
    """Return a parser pre-populated with the shared scraper flags.

    Args:
        description: Human-readable one-liner used in --help.
        stage:       Stage name, one of
                     'scrape','parse','extract','embed','reduce','visualize'.
    """
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config (defaults to configs/default.yaml).",
    )
    p.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Named config under configs/ (hfdata-style). Resolves to configs/<name>.yaml.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("./data"),
        help="Output root directory (default: ./data).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Thread pool size for I/O-bound stages (default: 4).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N items (useful for smoke tests).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore progress.json and restart from scratch.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Redo all items regardless of completeness. "
            "Stronger than --no-resume: overwrites outputs."
        ),
    )
    p.add_argument(
        "--stop-after",
        type=str,
        default=None,
        choices=["scrape", "parse", "extract", "embed", "reduce", "visualize"],
        help="When invoked via run.py, stop after this stage.",
    )
    p.add_argument(
        "--proxy",
        type=str,
        default=None,
        help=(
            "HTTP/HTTPS/SOCKS5 proxy for geo-locked hosts "
            "(e.g. socks5h://127.0.0.1:1080 or http://user:pass@vn.proxy:8080). "
            "Honored by the scraper stage only."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=(),
        metavar="KEY=VALUE",
        help=(
            "Hydra-style dotlist overrides applied after config load, e.g. "
            "--override embedder.batch_size=16 num_workers=8"
        ),
    )
    p.set_defaults(stage=stage)
    return p


def load_and_override(
    config_path: Path,
    overrides: list[str] | tuple[str, ...] = (),
    schema_cls: type | None = None,
):
    """Load a YAML config, merge on top of an optional typed schema, apply overrides.

    Returns an OmegaConf DictConfig. This is the canonical one-call helper
    every stage script uses.
    """
    from omegaconf import OmegaConf

    from packages.scrapers.common.config import (
        apply_overrides,
        load_config,
        structured_config,
    )

    parts = []
    if schema_cls is not None:
        parts.append(structured_config(schema_cls))
    parts.append(load_config(config_path))
    cfg = parts[0] if len(parts) == 1 else OmegaConf.merge(*parts)
    cfg = apply_overrides(cfg, list(overrides))
    return cfg


def apply_log_level(log_level: str) -> None:
    """Configure root logger with the requested level and a terse format."""
    import logging

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
