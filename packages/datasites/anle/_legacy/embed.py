"""Embedder pipeline factory for anle: JSONL -> embeddings parquet.

Stage chain (built by
:func:`packages.pipeline.factories.build_embed_pipeline`)::

    JsonlReader(jsonl_dir, fields=EMBEDDER_JSONL_READ_FIELDS)
    -> NimEmbedderStage | EmbeddingCreatorStage  (cfg.embedder.runtime)
    -> ParquetPerDocWriter(embeddings_dir, fields=EMBEDDER_PARQUET_FIELDS)
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.anle._shared import (
    EMBEDDER_JSONL_READ_FIELDS,
    EMBEDDER_PARQUET_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_embed_pipeline as _build


def build_embed_pipeline(cfg: Any) -> Pipeline:
    """Return the anle Embedder :class:`Pipeline`."""
    return _build(
        cfg,
        site="anle",
        layout=build_layout(cfg),
        read_fields=EMBEDDER_JSONL_READ_FIELDS,
        parquet_fields=EMBEDDER_PARQUET_FIELDS,
    )


__all__ = ["build_embed_pipeline"]
