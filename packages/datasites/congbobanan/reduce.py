"""Reducer pipeline: embeddings parquet -> reduced parquet.

Stage chain::

    ParquetReader(embeddings_dir)
    -> ReducerStage (PCA/t-SNE/UMAP + HDBSCAN cluster_id)
    -> ParquetWriter(reduced_dir)

Reads: ``data/<host>/parquet/embeddings/*.parquet``.
Writes: ``data/<host>/parquet/reduced/*.parquet`` with reducer coords
(``{pca,tsne,umap}_{x,y,z}``) + ``cluster_id`` added to the embedding
+ id columns.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader
from nemo_curator.stages.text.io.writer import ParquetWriter

from packages.datasites.congbobanan._shared import (
    EMBEDDER_PARQUET_FIELDS,
    REDUCER_PARQUET_FIELDS,
    build_layout,
)
from packages.reducer.stage import ReducerStage


def build_reduce_pipeline(cfg: Any) -> Pipeline:
    """Return the Reducer :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-reduce",
        description="congbobanan Reducer: embeddings parquet -> reduced parquet.",
        stages=[
            ParquetReader(
                file_paths=str(layout.embeddings_dir),
                fields=list(EMBEDDER_PARQUET_FIELDS),
                files_per_partition=int(
                    cfg.get("stage_overrides", {}).get(
                        "reduce_files_per_partition", 64
                    )
                ),
            ),
            ReducerStage(cfg=cfg),
            ParquetWriter(
                path=str(layout.reduced_dir),
                fields=list(REDUCER_PARQUET_FIELDS),
                mode="ignore",
            ),
        ],
        config={"host": str(cfg.host), "reduced_dir": str(layout.reduced_dir)},
    )


__all__ = ["build_reduce_pipeline"]
