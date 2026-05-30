"""Embedder pipeline factory for thuvienphapluat_banan: parquet/extract → embed.

Stage chain (built by
:func:`packages.pipeline.factories.build_embed_pipeline`)::

    ParquetReader(parquet/extract/*.parquet, fields=EMBEDDER_PARQUET_READ_FIELDS)
    -> NimEmbedderStage | EmbeddingCreatorStage  (cfg.embedder.runtime)
    -> ParquetPerDocWriter(embeddings_dir, fields=EMBEDDER_PARQUET_FIELDS)

The embedder reads directly from the parquet consumption tier
(``data/<host>/parquet/extract/extract-*.parquet``) and propagates the
sidebar metadata (title / court / doc_number / trial_level /
legal_area / case_kind / procedure / year / issue_date / source_url)
through to the ``parquet/embed/`` shards so they're self-describing
without a join back to ``documents-*.parquet``.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.thuvienphapluat_banan._shared import (
    EMBEDDER_PARQUET_FIELDS,
    EMBEDDER_PARQUET_READ_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_embed_pipeline as _build


def build_embed_pipeline(cfg: Any) -> Pipeline:
    """Return the thuvienphapluat_banan Embedder :class:`Pipeline`."""
    layout = build_layout(cfg)
    return _build(
        cfg,
        site="thuvienphapluat_banan",
        layout=layout,
        read_fields=EMBEDDER_PARQUET_READ_FIELDS,
        parquet_fields=EMBEDDER_PARQUET_FIELDS,
        parquet_path=str(layout.extract_parquet_dir),
    )


__all__ = ["build_embed_pipeline"]
