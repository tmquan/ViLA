"""Embedder pipeline factory for congbobanan: JSONL -> embeddings parquet.

Stage chain (built by
:func:`packages.pipeline.factories.build_embed_pipeline`)::

    JsonlReader(jsonl_dir, fields=EMBEDDER_JSONL_READ_FIELDS)
    -> NimEmbedderStage | EmbeddingCreatorStage  (cfg.embedder.runtime)
    -> ParquetPerDocWriter(embeddings_dir, fields=EMBEDDER_PARQUET_FIELDS)
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.congbobanan._shared import (
    EMBEDDER_JSONL_READ_FIELDS,
    EMBEDDER_PARQUET_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_embed_pipeline as _build


def build_embed_pipeline(cfg: Any) -> Pipeline:
    """Return the congbobanan Embedder :class:`Pipeline`."""
    return _build(
        cfg,
        site="congbobanan",
        layout=build_layout(cfg),
        read_fields=EMBEDDER_JSONL_READ_FIELDS,
        parquet_fields=EMBEDDER_PARQUET_FIELDS,
        files_per_partition=int(
            cfg.get("stage_overrides", {}).get(
                "embed_files_per_partition", 16,
            )
        ),
    )


__all__ = ["build_embed_pipeline"]
