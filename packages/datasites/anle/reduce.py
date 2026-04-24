"""Reducer pipeline: embeddings parquet -> reduced parquet.

Stage chain::

    ParquetReader(embeddings_dir)
    -> ReducerStage (PCA/t-SNE/UMAP + HDBSCAN cluster_id)
    -> ParquetWriter(reduced_dir)

Reads: ``data/<host>/parquet/embeddings/*.parquet`` (Embedder output).
Writes: ``data/<host>/parquet/reduced/*.parquet`` with reducer coords
(``{pca,tsne,umap}_{x,y,z}``) and ``cluster_id`` added to the
embedding + id columns.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import ParquetReader

from packages.datasites.anle._shared import (
    EMBEDDER_PARQUET_FIELDS,
    REDUCER_PARQUET_FIELDS,
    build_layout,
)
from packages.pipeline.io import ParquetPerDocWriter
from packages.reducer.stage import ReducerStage


def build_reduce_pipeline(cfg: Any) -> Pipeline:
    """Return the Reducer :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-reduce",
        description="anle Reducer: embeddings parquet -> reduced parquet.",
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
            ParquetPerDocWriter(
                path=str(layout.reduced_dir),
                doc_name_field="doc_name",
                fields=list(REDUCER_PARQUET_FIELDS),
            ),
        ],
        config={"host": str(cfg.host), "reduced_dir": str(layout.reduced_dir)},
    )


__all__ = ["build_reduce_pipeline"]
