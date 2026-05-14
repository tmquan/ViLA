"""In-process embed + reduce driver for the pbgdpl Q&A corpus.

Run with::

    python -m packages.datasites.pbgdpl._embed_reduce_inproc

Reads ``data/<host>/jsonl/qa.jsonl``, embeds every answer via the
NIM embedder declared in ``cfg.embedder``, fits PCA / t-SNE / UMAP
(plus HDBSCAN cluster ids) over the full matrix, and writes a single
``data/<host>/parquet/qa_reduced.parquet`` with the join key
``item_id`` plus the reducer coords. Output is consumed by
:func:`packages.datasites.pbgdpl.viz.render_topic_umap`.

Embeds the cleaned ``answer_text`` (not the question) because
official answers are longer + richer and produce a better
clustering signal. Truncates each answer to ``--max-chars`` (default
4000) so the per-row embedding payload stays under the
1k-token NIM limit without sliding-window chunking.
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
from packages.embedder.stage import build_embedder_stage
from packages.reducer.stage import ReducerStage

logger = logging.getLogger(__name__)


def _load_qa(jsonl_path: Path, max_chars: int) -> pd.DataFrame:
    """Load qa.jsonl into a DataFrame with the columns the embedder needs."""
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            answer = (r.get("answer_text") or "").strip()
            if not answer:
                continue
            rows.append({
                "item_id":   str(r.get("item_id") or ""),
                "doc_name":  str(r.get("item_id") or ""),  # embedder/reducer key
                "lv_names":  r.get("lv_names") or [],
                "title":     r.get("title") or r.get("title_listing") or "",
                "markdown":  answer[:max_chars],
            })
    return pd.DataFrame(rows)


def run(
    cfg,
    *,
    embed_batch_size: int = 32,
    max_chars: int = 4000,
) -> Path:
    layout = build_layout(cfg)
    qa_path = layout.jsonl_dir / "qa.jsonl"
    if not qa_path.exists():
        raise FileNotFoundError(qa_path)
    parquet_dir = layout.site_root / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    out_path = parquet_dir / "qa_reduced.parquet"

    df = _load_qa(qa_path, max_chars=max_chars)
    logger.info(
        "loaded %d Q&A rows (truncated to %d chars each)", len(df), max_chars,
    )

    embedder = build_embedder_stage(cfg)
    embedder.setup(None)

    # Embed in moderate batches so a single retry doesn't redo
    # thousands of items.
    out_chunks: list[pd.DataFrame] = []
    for i in range(0, len(df), embed_batch_size):
        sub = df.iloc[i:i + embed_batch_size].copy()
        batch = DocumentBatch(
            task_id=f"qa_embed_{i}", dataset_name="pbgdpl", data=sub,
        )
        result = embedder.process(batch).to_pandas()
        out_chunks.append(result)
        if (i // embed_batch_size) % 10 == 0:
            done = min(i + embed_batch_size, len(df))
            logger.info("embedded %d / %d", done, len(df))

    embeddings_df = pd.concat(out_chunks, ignore_index=True)
    logger.info("done embedding %d rows", len(embeddings_df))

    reducer = ReducerStage(cfg=cfg)
    reducer.setup(None)
    batch = DocumentBatch(
        task_id="qa_reduce", dataset_name="pbgdpl", data=embeddings_df,
    )
    reduced = reducer.process(batch).to_pandas()

    keep = [
        "item_id", "doc_name", "lv_names", "title",
        "embedding", "embedding_dim",
        "pca_x", "pca_y", "tsne_x", "tsne_y", "umap_x", "umap_y",
        "cluster_id",
    ]
    keep = [c for c in keep if c in reduced.columns]
    reduced[keep].to_parquet(out_path, index=False)
    logger.info("wrote %s (%d rows)", out_path, len(reduced))

    for col in ("pca_x", "tsne_x", "umap_x"):
        if col in reduced.columns:
            nn = reduced[col].notna().sum()
            logger.info("  %s: %d / %d non-null", col, nn, len(reduced))

    return out_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="In-process pbgdpl embed + reduce (no Ray).",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--embed-batch-size", type=int, default=32,
        help="Q&A rows per NIM call (default: 32)",
    )
    parser.add_argument(
        "--max-chars", type=int, default=4000,
        help="Truncate each answer to this many chars (default: 4000)",
    )
    args = parser.parse_args(argv)

    config_path = args.config or find_site_config("pbgdpl")
    cfg = load_and_override(
        config_path=config_path, overrides=[], schema_cls=PipelineCfg,
    )
    out = run(
        cfg,
        embed_batch_size=args.embed_batch_size,
        max_chars=args.max_chars,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
