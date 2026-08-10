"""In-process driver for the anle Reducer stage.

Run with::

    python -m packages.datasites.anle._reduce_inproc

A no-frills driver that reads every embedding parquet under
``parquet/embeddings/``, fits PCA / t-SNE / UMAP + HDBSCAN on the full
matrix in one pass, and writes per-doc parquets under
``parquet/reduced/``. Output schema matches what
``packages.datasites.anle --pipeline reduce`` produces; only the
runtime differs (no Ray, no Curator executor).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from nemo_curator.tasks import DocumentBatch

from packages.common import build_layout, find_site_config, load_and_override
from packages.common.schemas import PipelineCfg
from packages.datasites.anle._shared import REDUCER_PARQUET_FIELDS
from packages.reducer.stage import ReducerStage

logger = logging.getLogger(__name__)


def run(cfg) -> int:
    """Read embeddings parquet, fit reducers, write per-doc reduced parquet."""
    layout = build_layout(cfg)
    emb_dir = layout.embeddings_dir
    out_dir = layout.reduced_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(emb_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no embedding parquets under {emb_dir}; run the embed pipeline first.",
        )
    logger.info("loading %d embedding parquets from %s", len(files), emb_dir)
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    logger.info("loaded %d rows; embedding dim=%d", len(df), len(df["embedding"].iloc[0]))

    stage = ReducerStage(cfg=cfg)
    stage.setup(None)
    batch = DocumentBatch(task_id="reduce_all", dataset_name="anle", data=df)
    out_batch = stage.process(batch)
    out_df = out_batch.to_pandas()

    # Per-doc parquet writeback. Project to the canonical column set
    # (REDUCER_PARQUET_FIELDS) so the output schema is stable.
    keep = [c for c in REDUCER_PARQUET_FIELDS if c in out_df.columns]
    out_df = out_df[keep]

    n = 0
    for _, row in out_df.iterrows():
        doc_name = str(row.get("doc_name") or "").strip()
        if not doc_name:
            continue
        one = pd.DataFrame([row.to_dict()], columns=keep)
        one.to_parquet(out_dir / f"{doc_name}.parquet", index=False)
        n += 1

    # Quick sanity log of how each method came out.
    for col in ("pca_x", "tsne_x", "umap_x"):
        if col in out_df.columns:
            nn = out_df[col].notna().sum()
            logger.info("  %s: %d / %d non-null", col, nn, len(out_df))
    logger.info("wrote %d reduced parquets under %s", n, out_dir)
    return n


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="In-process anle Reducer (no Ray).",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    config_path = args.config or find_site_config("anle")
    overrides: list[str] = []
    if args.output:
        overrides.append(f"output_dir={args.output.expanduser().resolve()}")
    cfg = load_and_override(
        config_path=config_path, overrides=overrides, schema_cls=PipelineCfg,
    )
    n = run(cfg)
    print(f"wrote {n} reduced parquet files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
