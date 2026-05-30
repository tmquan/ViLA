"""Reducer pipeline factory for thuvienphapluat_banan: embed -> reduce.

Stage chain (built by
:func:`packages.pipeline.factories.build_reduce_pipeline`)::

    ParquetReader(embeddings_dir)
    -> ReducerStage (PCA / t-SNE / UMAP + HDBSCAN cluster_id)
    -> ParquetPerDocWriter(reduced_dir, fields=REDUCER_PARQUET_FIELDS)

The reducer fits all three projections in one pass over the full
``embedding`` matrix so the output rows carry globally-consistent
2- or 3-D coordinates plus an HDBSCAN ``cluster_id``. GPU-accelerated
via cuML when ``cfg.reducer.prefer_gpu`` is set and cuML is importable;
otherwise falls back to sklearn / umap-learn / hdbscan.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.thuvienphapluat_banan._shared import (
    EMBEDDER_PARQUET_FIELDS,
    REDUCER_PARQUET_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_reduce_pipeline as _build


def build_reduce_pipeline(cfg: Any) -> Pipeline:
    """Return the thuvienphapluat_banan Reducer :class:`Pipeline`."""
    return _build(
        cfg,
        site="thuvienphapluat_banan",
        layout=build_layout(cfg),
        embedder_fields=EMBEDDER_PARQUET_FIELDS,
        reducer_fields=REDUCER_PARQUET_FIELDS,
    )


__all__ = ["build_reduce_pipeline"]
