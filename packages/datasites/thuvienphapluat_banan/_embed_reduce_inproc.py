"""In-process embed + reduce driver for thuvienphapluat_banan (T3 fallback).

This is the **escape hatch** for environments without a Ray cluster
(wiki/DATASITES.md §10a.2 trigger T3). It builds the same canonical
:class:`packages.embedder.stage` / :class:`packages.reducer.stage`
stages the Curator pipelines use, but runs them in a single process
against the parquet outputs the curator pipeline would have produced.

Use this **only** when:

1. You have a corpus small enough that a single-actor fit is acceptable
   (≲ 10 K judgments; UMAP scales O(n × log n) but the matrix has to
   fit in RAM).
2. The production ``--pipeline embed`` / ``--pipeline reduce`` is
   blocked (no Ray, no GPU, no NIM key, …) but you still need
   ``hf_export.py`` to find an embedding shard.

Output mirrors the production parquet consumption tier so re-running
``--pipeline embed reduce`` on top of this in-process output is a
no-op (``mode="ignore"`` keys off the same shard filenames).

Run via::

    python -m packages.datasites.thuvienphapluat_banan._embed_reduce_inproc
    python -m packages.datasites.thuvienphapluat_banan._embed_reduce_inproc \\
        --limit 100   # smoke-test on the first 100 judgments
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from packages.common import find_site_config, load_config
from packages.datasites.thuvienphapluat_banan._shared import (
    EMBEDDER_PARQUET_FIELDS,
    EMBEDDER_PARQUET_READ_FIELDS,
    REDUCER_PARQUET_FIELDS,
    build_layout,
)

logger = logging.getLogger(__name__)


def _read_extract_parquet(
    parquet_dir: Path, *, limit: int | None,
) -> Any:
    """Read every ``extract-*.parquet`` shard into one pandas DataFrame."""
    try:
        import pandas as pd
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow + pandas required for the in-process driver; "
            "`pip install pyarrow pandas`"
        ) from exc

    shards = sorted(parquet_dir.glob("extract-*-of-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no extract-*.parquet in {parquet_dir}; "
            f"run --pipeline extract first.",
        )
    logger.info("reading %d extract shards from %s", len(shards), parquet_dir)
    tables = [pq.read_table(str(s)) for s in shards]
    df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
    if limit is not None:
        df = df.head(int(limit))
    return df


def _run_embed(cfg: Any, df: Any) -> Any:
    """Run :func:`build_embedder_stage` over the DataFrame."""
    try:
        from nemo_curator.tasks import DocumentBatch
        from packages.embedder.stage import build_embedder_stage
    except ImportError as exc:
        raise RuntimeError(
            "packages.embedder / nemo_curator required; install them "
            "(see requirements.txt) before running the in-process driver"
        ) from exc

    # Project to the columns the embedder reads.
    cols = [c for c in EMBEDDER_PARQUET_READ_FIELDS if c in df.columns]
    df_proj = df[cols].copy()

    stage = build_embedder_stage(cfg)
    stage.setup(None)
    batch = DocumentBatch(
        task_id="banan-inproc-embed", dataset_name="thuvienphapluat_banan",
        data=df_proj,
    )
    out_batch = stage.process(batch)
    return out_batch.to_pandas() if hasattr(out_batch, "to_pandas") else out_batch.data


def _run_reduce(cfg: Any, df_embed: Any) -> Any:
    try:
        from nemo_curator.tasks import DocumentBatch
        from packages.reducer.stage import ReducerStage
    except ImportError as exc:
        raise RuntimeError(
            "packages.reducer required; install nemo_curator + reducer extras"
        ) from exc

    stage = ReducerStage(cfg=cfg)
    stage.setup(None)
    batch = DocumentBatch(
        task_id="banan-inproc-reduce", dataset_name="thuvienphapluat_banan",
        data=df_embed,
    )
    out = stage.process(batch)
    return out.to_pandas() if hasattr(out, "to_pandas") else out.data


def _write_shards(df: Any, out_dir: Path, *, stage: str) -> int:
    """Write the DataFrame as a single ``<stage>-00000-of-00001.parquet`` shard."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    # Project to the canonical column list so the output is byte-substitutable
    # with the production shard layout.
    fields = (
        EMBEDDER_PARQUET_FIELDS if stage == "embed" else REDUCER_PARQUET_FIELDS
    )
    cols = [c for c in fields if c in df.columns]
    table = pa.Table.from_pandas(df[cols], preserve_index=False)
    out_path = out_dir / f"{stage}-00000-of-00001.parquet"
    pq.write_table(table, out_path, row_group_size=1024)
    logger.info("wrote %s (%d rows, %d columns)", out_path, len(df), len(cols))
    return 1


def run(cfg: Any, *, limit: int | None = None) -> dict[str, int]:
    layout = build_layout(cfg)
    df = _read_extract_parquet(layout.extract_parquet_dir, limit=limit)

    logger.info("running in-process embed over %d rows", len(df))
    df_embed = _run_embed(cfg, df)
    n_embed = _write_shards(df_embed, layout.embed_parquet_dir, stage="embed")

    logger.info("running in-process reduce over %d rows", len(df_embed))
    df_reduce = _run_reduce(cfg, df_embed)
    n_reduce = _write_shards(df_reduce, layout.reduce_parquet_dir, stage="reduce")

    return {"embed_shards": n_embed, "reduce_shards": n_reduce}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-name", default="thuvienphapluat_banan")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows of parquet/extract/.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = find_site_config(args.config_name)
    cfg = load_config(cfg_path)
    counts = run(cfg, limit=args.limit)
    print(
        f"_embed_reduce_inproc done: embed={counts['embed_shards']} "
        f"reduce={counts['reduce_shards']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
