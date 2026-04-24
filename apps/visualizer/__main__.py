"""Render every visualization artifact from the pipeline's parquet output.

    python -m apps.visualizer --config-name anle

Usage mirrors the pipeline CLI; this tool reads
``data/<host>/parquet/*.parquet``, applies ontology fill-ins, and
walks the :data:`packages.visualizer.RENDERER_REGISTRY`. Output lands
in ``data/<host>/viz/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from packages.common import (
    SiteLayout,
    apply_log_level,
    find_site_config,
    load_and_override,
)
from packages.common.ontology import load_ontology
from packages.common.schemas import PipelineCfg
from packages.embedder.base import model_slug
from packages.visualizer import RENDERER_REGISTRY
from packages.visualizer.base import build_dataset

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render ViLA visualization artifacts from pipeline parquet output."
    )
    p.add_argument("-c", "--config", type=Path, default=None)
    p.add_argument("--config-name", type=str, default=None)
    p.add_argument("--output", type=Path, default=None, help="Override cfg.output_dir.")
    p.add_argument("--force", action="store_true", help="Re-render every artifact.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--override", nargs="*", default=(), metavar="KEY=VALUE")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    apply_log_level(args.log_level)

    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else find_site_config(args.config_name or "anle")
    )
    overrides = list(args.override)
    if args.output:
        overrides.append(
            f"output_dir={str(Path(args.output).expanduser().resolve())}"
        )
    cfg = load_and_override(
        config_path=config_path, overrides=overrides, schema_cls=PipelineCfg
    )

    layout = SiteLayout(
        output_root=Path(str(cfg.output_dir)).expanduser().resolve(),
        host=str(cfg.host),
    )
    viz_dir = layout.site_root / "viz"
    layout.ensure_dirs(viz_dir)

    onto = load_ontology()
    slug = model_slug(str(cfg.embedder.model_id))
    # The Reducer pipeline terminates at parquet/reduced/*.parquet;
    # the Extractor pipeline terminates at jsonl/*.jsonl. Join on
    # doc_name (see packages.visualizer.base.load_pipeline_output).
    df = build_dataset(layout.reduced_dir, onto, jsonl_dir=layout.jsonl_dir)
    if df.empty:
        logger.warning(
            "no parquet output at %s; run the pipeline first "
            "(python -m packages.datasites.%s).",
            layout.parquet_dir, cfg.host.split(".")[0],
        )
        return 0

    counts: dict[str, int] = {"scatters": 0, "distributions": 0, "misc": 0}
    for cls in RENDERER_REGISTRY:
        renderer = cls()
        written = renderer.render(
            df,
            out_dir=viz_dir,
            cfg=cfg,
            onto=onto,
            slug=slug,
            force=args.force,
        )
        counts.setdefault(renderer.bucket, 0)
        counts[renderer.bucket] += written

    logger.info("visualizer done: %s (out_dir=%s)", counts, viz_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
