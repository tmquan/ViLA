"""In-process driver for the anle Embedder stage.

Run with::

    python -m packages.datasites.anle._embed_inproc \
        --config packages/datasites/anle/configs/anle_nemotron3.yaml \
        --output ~/data

A no-frills driver that reads every extracted JSONL under ``jsonl/``,
embeds each document with the configured backend (:class:`NimEmbedderStage`
-- NIM or local HuggingFace runtime), and writes per-doc parquets under
``parquet/embeddings/``. Output schema matches what
``packages.datasites.anle --pipeline embed`` produces; only the runtime
differs (no Ray, no Curator executor).

This is the counterpart to :mod:`packages.datasites.anle._reduce_inproc`
and is the supported path on single-host boxes where cosmos-xenna cannot
see the GPU (e.g. a GB10 without ``pynvml``), so the Ray/xenna scheduler
refuses the ``gpus=1.0`` embed actor with "Not enough GPU resources".
The embedding runs on the GPU exactly as it would inside a Curator actor
-- we just skip the executor that can't place it.

Resumable: documents whose ``<doc_name>.parquet`` already exists under
``parquet/embeddings/`` are skipped, so an interrupted run re-runs cheap.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from nemo_curator.tasks import DocumentBatch

from packages.common import build_layout, find_site_config, load_and_override
from packages.common.schemas import PipelineCfg
from packages.datasites.anle._shared import (
    EMBEDDER_JSONL_READ_FIELDS,
    EMBEDDER_PARQUET_FIELDS,
)
from packages.embedder.stage import NimEmbedderStage

logger = logging.getLogger(__name__)


def _read_jsonl_docs(jsonl_dir: Path, fields: list[str]) -> pd.DataFrame:
    """Load every ``<doc>.jsonl`` (one JSON object per file) into a frame."""
    rows: list[dict] = []
    for p in sorted(jsonl_dir.glob("*.jsonl")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("skipping unparseable jsonl %s", p)
            continue
        rows.append({k: obj.get(k) for k in fields})
    return pd.DataFrame(rows)


def run(cfg, *, batch_size: int = 32) -> int:
    """Read extract JSONL, embed each doc, write per-doc embedding parquet."""
    layout = build_layout(cfg)
    jsonl_dir = layout.jsonl_dir
    out_dir = layout.embeddings_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_jsonl_docs(jsonl_dir, EMBEDDER_JSONL_READ_FIELDS)
    if df.empty:
        raise FileNotFoundError(
            f"no jsonl docs under {jsonl_dir}; run --pipeline extract first.",
        )

    # Resume: drop docs already embedded on disk.
    before = len(df)
    df = df[~df["doc_name"].map(lambda d: (out_dir / f"{d}.parquet").exists())]
    df = df.reset_index(drop=True)
    logger.info(
        "loaded %d docs from %s (%d already embedded, %d to do)",
        before, jsonl_dir, before - len(df), len(df),
    )
    if df.empty:
        logger.info("nothing to do; all docs already embedded")
        return before

    stage = NimEmbedderStage(cfg=cfg)
    stage.setup(None)
    logger.info(
        "embedder ready: model=%s runtime=%s dim=%s device=%s",
        stage._entry.model_id, cfg.embedder.runtime,
        stage._backend.embedding_dim, getattr(stage._backend, "_device", "?"),
    )

    keep = list(EMBEDDER_PARQUET_FIELDS)
    written = 0
    for start in range(0, len(df), batch_size):
        sub = df.iloc[start : start + batch_size].reset_index(drop=True)
        out = stage.process(
            DocumentBatch(task_id=f"embed_{start}", dataset_name="anle", data=sub)
        ).to_pandas()
        cols = [c for c in keep if c in out.columns]
        for _, row in out[cols].iterrows():
            doc_name = str(row.get("doc_name") or "").strip()
            if not doc_name:
                continue
            pd.DataFrame([row.to_dict()], columns=cols).to_parquet(
                out_dir / f"{doc_name}.parquet", index=False
            )
            written += 1
        logger.info("embedded %d / %d docs", min(start + batch_size, len(df)), len(df))

    logger.info("wrote %d embedding parquets under %s", written, out_dir)
    return written


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="In-process anle Embedder (no Ray).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    config_path = args.config or find_site_config("anle")
    overrides: list[str] = []
    if args.output:
        overrides.append(f"output_dir={args.output.expanduser().resolve()}")
    cfg = load_and_override(
        config_path=config_path, overrides=overrides, schema_cls=PipelineCfg,
    )
    n = run(cfg, batch_size=args.batch_size)
    print(f"wrote {n} embedding parquet files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
