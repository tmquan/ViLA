"""Embedder pipeline factory for vbpl: extract.jsonl -> embeddings parquet.

Stage chain (built by
:func:`packages.pipeline.factories.build_embed_pipeline`)::

    JsonlReader(jsonl/extract.jsonl, fields=EMBEDDER_JSONL_READ_FIELDS)
    -> NimEmbedderStage | EmbeddingCreatorStage  (cfg.embedder.runtime)
    -> ParquetPerDocWriter(embeddings_dir, fields=EMBEDDER_PARQUET_FIELDS)

vbpl emits a single consolidated ``jsonl/extract.jsonl`` (rather than
per-doc shards like anle / congbobanan) so the JSONL reader needs an
explicit file path; the shared factory accepts a ``jsonl_path``
override exactly for this case.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.vbpl._shared import (
    EMBEDDER_JSONL_READ_FIELDS,
    EMBEDDER_PARQUET_FIELDS,
    build_layout,
    extract_jsonl_path,
)
from packages.pipeline.factories import build_embed_pipeline as _build


def build_embed_pipeline(cfg: Any) -> Pipeline:
    """Return the vbpl Embedder :class:`Pipeline`."""
    layout = build_layout(cfg)
    return _build(
        cfg,
        site="vbpl",
        layout=layout,
        read_fields=EMBEDDER_JSONL_READ_FIELDS,
        parquet_fields=EMBEDDER_PARQUET_FIELDS,
        jsonl_path=str(extract_jsonl_path(layout)),
    )


__all__ = ["build_embed_pipeline"]
