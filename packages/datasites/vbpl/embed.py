"""Embedder pipeline factory for vbpl: parquet/extract → embeddings parquet.

Stage chain (built by
:func:`packages.pipeline.factories.build_embed_pipeline`)::

    ParquetReader(parquet/extract/*.parquet, fields=EMBEDDER_PARQUET_READ_FIELDS)
    -> NimEmbedderStage | EmbeddingCreatorStage  (cfg.embedder.runtime)
    -> ParquetPerDocWriter(embeddings_dir, fields=EMBEDDER_PARQUET_FIELDS)

vbpl follows the canonical wiki.md §3.5 pattern: the Embedder reads
directly from the parquet consumption tier produced by the extract
stage (``data/<host>/parquet/extract/extract-*.parquet``), not from
JSONL. Avoids one serialisation hop and lets the embedder pull the
sidebar metadata columns (``so_hieu``, ``ngay_ban_hanh``,
``co_quan_ban_hanh``, ``trich_yeu``, ``legal_type``, ``legal_area``,
``doc_type``, ``scope``, ``source_url``, ``title``) through to the
embed parquet shards so they're self-describing.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.vbpl._shared import (
    EMBEDDER_PARQUET_FIELDS,
    EMBEDDER_PARQUET_READ_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_embed_pipeline as _build


def build_embed_pipeline(cfg: Any) -> Pipeline:
    """Return the vbpl Embedder :class:`Pipeline`."""
    layout = build_layout(cfg)
    return _build(
        cfg,
        site="vbpl",
        layout=layout,
        read_fields=EMBEDDER_PARQUET_READ_FIELDS,
        parquet_fields=EMBEDDER_PARQUET_FIELDS,
        parquet_path=str(layout.extract_parquet_dir),
    )


__all__ = ["build_embed_pipeline"]
