"""Shared CLI argument parser for the Curator pipeline driver.

Every datasite entry point (``python -m packages.datasites.<site>``)
uses the same small set of flags. Stage selection, resume, and
per-stage thread pools are gone: the :class:`nemo_curator.pipeline.Pipeline`
owns composition and idempotency, and a :class:`BaseExecutor`
(XennaExecutor / RayActorPoolExecutor / RayDataExecutor) owns parallelism.

    --config / -c <path>        Explicit YAML path.
    --config-name <name>        Resolved to packages/datasites/<name>/configs/<name>.yaml.
    --executor {xenna,ray_actor_pool,ray_data}
    --ray-address <addr>        None | "auto" | "ray://head:10001".
    --limit N                   Cap URLs sent to the download stage.
    --output <dir>              Data root (default: ./data).
    --log-level <level>
    --override KEY=VALUE ...    OmegaConf dotlist overrides applied last.
"""

from __future__ import annotations

import argparse
from pathlib import Path

EXECUTOR_CHOICES = ("xenna", "ray_actor_pool", "ray_data")


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Return a parser pre-populated with the shared pipeline-driver flags."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config (absolute or relative to cwd).",
    )
    p.add_argument(
        "--config-name",
        type=str,
        default=None,
        help=(
            "Named config; resolves to "
            "packages/datasites/<name>/configs/<name>.yaml."
        ),
    )
    p.add_argument(
        "--executor",
        type=str,
        default=None,
        choices=EXECUTOR_CHOICES,
        help=(
            "Curator backend to run the pipeline on. Overrides cfg.executor.name. "
            "Defaults to the config value (xenna)."
        ),
    )
    p.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help=(
            "Ray bootstrap address. Use 'auto' for a local cluster or "
            "'ray://<head>:10001' for Ray Client. Overrides cfg.ray.address."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on URLs handed to the download stage (smoke-test aid).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output data root (default: cfg.output_dir, usually ./data).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level for the driver process.",
    )
    p.add_argument(
        "--override",
        nargs="*",
        default=(),
        metavar="KEY=VALUE",
        help=(
            "OmegaConf dotlist overrides applied after config load, e.g. "
            "--override embedder.batch_size=16 executor.mode=batch"
        ),
    )
    return p


def load_and_override(
    config_path: Path,
    overrides: list[str] | tuple[str, ...] = (),
    schema_cls: type | None = None,
):
    """Load a YAML config, merge on top of an optional typed schema, apply overrides.

    Returns an OmegaConf DictConfig. This is the canonical one-call helper
    every datasite __main__.py uses.
    """
    from omegaconf import OmegaConf

    from packages.common.config import (
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


__all__ = [
    "EXECUTOR_CHOICES",
    "apply_log_level",
    "build_arg_parser",
    "load_and_override",
]
