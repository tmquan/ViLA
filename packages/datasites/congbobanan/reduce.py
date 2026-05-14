"""Reducer pipeline factory for congbobanan: embeddings -> reduced parquet.

Stage chain (built by
:func:`packages.pipeline.factories.build_reduce_pipeline`)::

    ParquetReader(embeddings_dir)
    -> ReducerStage (PCA / t-SNE / UMAP + HDBSCAN cluster_id)
    -> ParquetPerDocWriter(reduced_dir, fields=REDUCER_PARQUET_FIELDS)
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.congbobanan._shared import (
    EMBEDDER_PARQUET_FIELDS,
    REDUCER_PARQUET_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_reduce_pipeline as _build


def build_reduce_pipeline(cfg: Any) -> Pipeline:
    """Return the congbobanan Reducer :class:`Pipeline`."""
    return _build(
        cfg,
        site="congbobanan",
        layout=build_layout(cfg),
        embedder_fields=EMBEDDER_PARQUET_FIELDS,
        reducer_fields=REDUCER_PARQUET_FIELDS,
    )


__all__ = ["build_reduce_pipeline"]
