"""In-process embed + reduce for dichvucong (no Ray / no xenna).

Run with::

    python -m packages.datasites.dichvucong._embed_reduce_inproc --stage embed
    python -m packages.datasites.dichvucong._embed_reduce_inproc --stage reduce
    python -m packages.datasites.dichvucong._embed_reduce_inproc --stage all

* **embed**  — reads ``jsonl/procedures.jsonl`` (the rich ``content_text``
  body), embeds via :class:`NimEmbedderStage` in-process (HTTP-bound, no
  GPU actor), writes one ``parquet/embeddings/<doc_name>.parquet`` per row.
  Resumable: rows whose parquet already exists are skipped.
* **reduce** — loads every embedding parquet in one pass, fits PCA + UMAP +
  t-SNE (cuML GPU when available) on the full matrix, writes a single
  ``parquet/reduced/reduced.parquet``.

Why in-process: the Curator GPU scheduler deadlocks on this host (pynvml
absent), so the reducer's ``Resources(gpus=1)`` request never places. cuML
uses the GPU directly, so a single full-batch fit in-process sidesteps the
scheduler — identical to ``dichvucong._reduce_inproc``.
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
from packages.datasites.dichvucong._shared import (
    REDUCER_PARQUET_FIELDS,
    procedures_jsonl,
)
from packages.embedder.stage import NimEmbedderStage
from packages.reducer.stage import ReducerStage

logger = logging.getLogger(__name__)

_EMBED_BATCH = 64


def _read_procedures(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df = df[df["doc_name"].notna()].copy()
    df["doc_name"] = df["doc_name"].astype(str)
    df["content_text"] = df["content_text"].fillna("").astype(str)
    return df


def run_embed(cfg) -> int:
    layout = build_layout(cfg)
    emb_dir = layout.embeddings_dir
    emb_dir.mkdir(parents=True, exist_ok=True)
    src = procedures_jsonl(layout)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found; run the detail crawl first.")

    df = _read_procedures(src)
    # Resume: drop rows already embedded.
    done = {p.stem for p in emb_dir.glob("*.parquet")}
    todo = df[~df["doc_name"].isin(done)].reset_index(drop=True)
    logger.info("embed: %d procedures, %d already done, %d to embed",
                len(df), len(done), len(todo))
    if todo.empty:
        return 0

    stage = NimEmbedderStage(cfg=cfg)
    stage.setup(None)
    keep = ["doc_name", "embedding", "embedding_dim", "embedding_model_id",
            "embedding_text_hash", "embedding_chunks_used", "embedding_chunking"]
    written = 0
    for start in range(0, len(todo), _EMBED_BATCH):
        batch = todo.iloc[start:start + _EMBED_BATCH].copy()
        out = stage.process(
            DocumentBatch(task_id=f"embed_{start}", dataset_name="dichvucong", data=batch)
        ).to_pandas()
        for _, row in out.iterrows():
            emb = row.get("embedding")
            if emb is None or len(emb) == 0:
                continue
            rec = {k: row.get(k) for k in keep}
            pd.DataFrame([rec]).to_parquet(emb_dir / f"{row['doc_name']}.parquet", index=False)
            written += 1
        logger.info("embed: %d/%d (written %d)", min(start + _EMBED_BATCH, len(todo)), len(todo), written)
    logger.info("embed done: %d new vectors in %s", written, emb_dir)
    return written


def run_reduce(cfg) -> int:
    layout = build_layout(cfg)
    emb_dir = layout.embeddings_dir
    out_dir = layout.reduced_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pyarrow.dataset as ds

    files = sorted(emb_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no embedding parquets under {emb_dir}; run embed first.")
    logger.info("reduce: loading %d embedding parquets (single pass)...", len(files))
    df = ds.dataset(str(emb_dir), format="parquet").to_table().to_pandas()
    df = df[df["doc_name"].notna()].copy()
    df["doc_name"] = df["doc_name"].astype(str)
    df = df[df["embedding"].map(lambda v: v is not None and len(v) > 0)].reset_index(drop=True)
    logger.info("reduce: %d rows; dim=%d", len(df), len(df["embedding"].iloc[0]))

    stage = ReducerStage(cfg=cfg)
    stage.setup(None)
    out_df = stage.process(
        DocumentBatch(task_id="reduce_all", dataset_name="dichvucong", data=df)
    ).to_pandas()

    keep = [c for c in REDUCER_PARQUET_FIELDS if c in out_df.columns and c != "embedding"]
    out_df = out_df[keep].copy()
    out_df["doc_name"] = out_df["doc_name"].astype(str)
    for col in ("pca_x", "pca_y", "umap_x", "umap_y", "tsne_x", "tsne_y"):
        if col in out_df.columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce").astype("float64")
    out_path = out_dir / "reduced.parquet"
    out_df.to_parquet(out_path, index=False)
    for col in ("pca_x", "umap_x", "tsne_x"):
        if col in out_df.columns:
            logger.info("  %s: %d / %d non-null", col, out_df[col].notna().sum(), len(out_df))
    logger.info("reduce done: wrote %s (%d rows)", out_path, len(out_df))
    return len(out_df)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="In-process dichvucong embed+reduce (no Ray).")
    p.add_argument("--stage", choices=["embed", "reduce", "all"], default="all")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args(argv)
    cfg = load_and_override(
        config_path=args.config or find_site_config("dichvucong"),
        overrides=[], schema_cls=PipelineCfg,
    )
    if args.stage in ("embed", "all"):
        run_embed(cfg)
    if args.stage in ("reduce", "all"):
        run_reduce(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
