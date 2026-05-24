"""Shared CLI ``main()`` entry points for every datasite.

Two flavours, since the underlying execution model splits cleanly in two:

* :func:`run_curator_site` -- for Curator-based pipelines that need
  Ray initialisation, an :class:`Executor`, and per-stage ``pipeline.run``
  dispatch (``anle``, ``congbobanan``).
* :func:`run_crawler_site` -- for HTML-only crawlers that run a single
  in-process function per pipeline (``pbgdpl``, ``phapdien``).

Both share a private bootstrap helper that turns the standard
``packages.common.build_arg_parser`` flags + a ``--pipeline`` selector
into a resolved :class:`packages.common.PipelineCfg`. Each site's
``__main__.py`` shrinks to a 5-line wrapper that supplies the
site-specific module references.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from packages.common.cli import (
    apply_log_level,
    build_arg_parser,
    load_and_override,
)
from packages.common.config import find_site_config
from packages.common.schemas import PipelineCfg

logger = logging.getLogger(__name__)


# ----------------------------------------------------- shared bootstrap


def _build_parser(
    site: str,
    *,
    pipeline_choices: Sequence[str],
    description: str,
    pipeline_help: str,
) -> argparse.ArgumentParser:
    parser = build_arg_parser(description=description)
    parser.add_argument(
        "--pipeline",
        default="all",
        choices=list(pipeline_choices),
        help=pipeline_help,
    )
    return parser


def _resolve_cfg(
    args: argparse.Namespace,
    *,
    site: str,
    accept_ray_flags: bool,
) -> Any:
    """Turn parsed args into a resolved :class:`PipelineCfg`.

    ``accept_ray_flags=False`` (crawler sites) emits an info log if the
    user passes ``--executor`` or ``--ray-address`` rather than
    silently dropping them. Pipeline selection is computed by the
    caller from ``args.pipeline`` -- this helper is config-only.
    """
    apply_log_level(args.log_level)

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else find_site_config(args.config_name or site)
    )

    overrides = list(args.override)
    if accept_ray_flags:
        if args.executor:
            overrides.append(f"executor.name={args.executor}")
        if args.ray_address:
            overrides.append(f"ray.address={args.ray_address}")
    elif args.executor or args.ray_address:
        logger.info(
            "ignoring --executor / --ray-address (%s runs in-process)",
            site,
        )
    if args.limit is not None:
        overrides.append(f"limit={args.limit}")
    if args.output:
        overrides.append(
            f"output_dir={Path(args.output).expanduser().resolve()!s}"
        )

    return load_and_override(
        config_path=config_path,
        overrides=overrides,
        schema_cls=PipelineCfg,
    )


# ----------------------------------------------------- curator runner


def run_curator_site(
    *,
    site: str,
    pipelines: Mapping[str, Any],
    all_order: Sequence[str],
    build_pipeline: Callable[[Any, str], Any],
    description: str | None = None,
    pipeline_help: str | None = None,
    argv: list[str] | None = None,
) -> int:
    """``main()`` for a Curator-based datasite (anle / congbobanan).

    Bootstraps Ray, builds an executor per pipeline, and dispatches to
    ``build_pipeline(cfg, name)`` for each name in
    ``--pipeline`` (or ``all_order`` if ``--pipeline=all``).
    """
    # Lazy import: only Curator sites need the Ray + executor stack.
    from packages.pipeline import build_executor, init_ray, shutdown_ray

    parser = _build_parser(
        site,
        pipeline_choices=[*pipelines.keys(), "all"],
        description=description or f"Run the {site} curation pipelines.",
        pipeline_help=(
            pipeline_help
            or "Which pipeline to run. 'all' runs every step in declared "
               "order; individual names re-run one step against the prior "
               "step's on-disk output."
        ),
    )
    args = parser.parse_args(argv)
    cfg = _resolve_cfg(args, site=site, accept_ray_flags=True)

    selected: list[str] = (
        list(all_order) if args.pipeline == "all" else [args.pipeline]
    )
    logger.info("running pipelines: %s", selected)

    init_ray(cfg)
    rc = 0
    try:
        # Fail-fast contract: the first pipeline to raise -- either
        # during ``build_pipeline`` or ``pipeline.run`` -- aborts the
        # remaining ``selected`` list. ``rc=1`` is returned and the
        # stage name + traceback land in the log via ``exception``.
        # Sites that want soft-fail-across-pipelines should call this
        # function per pipeline and aggregate their own return codes.
        for idx, name in enumerate(selected):
            try:
                pipeline = build_pipeline(cfg, name)
                logger.info(
                    "=== pipeline %s ===\n%s", name, pipeline.describe(),
                )
                executor = build_executor(cfg)
                results = pipeline.run(executor=executor)
            except Exception:
                skipped = list(selected[idx + 1:])
                logger.exception(
                    "pipeline %s failed; aborting remaining pipelines: %s",
                    name, skipped,
                )
                rc = 1
                break
            logger.info(
                "pipeline %s finished: %d output tasks",
                name, len(results or []),
            )
    finally:
        if not cfg.ray.get("address"):
            shutdown_ray()
    return rc


# ----------------------------------------------------- crawler runner


def run_crawler_site(
    *,
    site: str,
    pipelines: Mapping[str, Any],
    all_order: Sequence[str],
    run_pipeline: Callable[[Any, str], Any],
    description: str | None = None,
    pipeline_help: str | None = None,
    argv: list[str] | None = None,
    accept_ray_flags: bool = False,
) -> int:
    """``main()`` for an HTML-only or hybrid crawler datasite.

    Each pipeline is a single in-process function; ``run_pipeline(cfg,
    name)`` is expected to return a ``Path`` (or any value the caller
    wants logged) for the per-stage output.

    ``accept_ray_flags=True`` opts in to honouring ``--executor`` /
    ``--ray-address`` -- useful for *hybrid* sites whose first
    stages run in-process but later stages dispatch a
    :class:`nemo_curator.pipeline.Pipeline` through Ray (vbpl). The
    default ``False`` matches pure HTML crawlers (pbgdpl / phapdien)
    that have no Ray-bound stages.
    """
    parser = _build_parser(
        site,
        pipeline_choices=[*pipelines.keys(), "all"],
        description=description or f"Run the {site} crawler.",
        pipeline_help=(
            pipeline_help
            or "Which stage to run. 'all' runs every stage in declared "
               "order; individual names re-run one stage against the prior "
               "stage's on-disk output."
        ),
    )
    args = parser.parse_args(argv)
    cfg = _resolve_cfg(args, site=site, accept_ray_flags=accept_ray_flags)

    selected: list[str] = (
        list(all_order) if args.pipeline == "all" else [args.pipeline]
    )
    logger.info("running pipelines: %s", selected)

    rc = 0
    # Fail-fast contract: first pipeline to raise aborts the remainder.
    for idx, name in enumerate(selected):
        logger.info("=== pipeline %s ===", name)
        try:
            out = run_pipeline(cfg, name)
        except Exception:
            skipped = list(selected[idx + 1:])
            logger.exception(
                "pipeline %s failed; aborting remaining pipelines: %s",
                name, skipped,
            )
            rc = 1
            break
        logger.info("pipeline %s finished: %s", name, out)
    return rc


__all__ = [
    "run_crawler_site",
    "run_curator_site",
]
