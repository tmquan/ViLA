"""Embedder pipeline: JSONL -> embeddings parquet.

Stage chain::

    JsonlReader(jsonl_dir, fields=[doc_name, case_id, text_hash, markdown])
    -> NimEmbedderStage | EmbeddingCreatorStage   (cfg.embedder.runtime)
    -> ParquetWriter(embeddings_dir)

Reads: ``data/<host>/jsonl/*.jsonl`` (Extractor output).
Writes: ``data/<host>/parquet/embeddings/*.parquet`` with
``doc_name``, ``case_id``, ``text_hash``, ``embedding`` + metadata.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader
from nemo_curator.stages.text.io.writer import ParquetWriter

from packages.datasites.congbobanan._shared import (
    EMBEDDER_JSONL_READ_FIELDS,
    EMBEDDER_PARQUET_FIELDS,
    build_layout,
)
from packages.embedder.stage import build_embedder_stage


def build_embed_pipeline(cfg: Any) -> Pipeline:
    """Return the Embedder :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-embed",
        description="congbobanan Embedder: JSONL -> embeddings parquet.",
        stages=[
            JsonlReader(
                file_paths=str(layout.jsonl_dir),
                fields=list(EMBEDDER_JSONL_READ_FIELDS),
                files_per_partition=int(
                    cfg.get("stage_overrides", {}).get(
                        "embed_files_per_partition", 16
                    )
                ),
            ),
            build_embedder_stage(cfg),
            ParquetWriter(
                path=str(layout.embeddings_dir),
                fields=list(EMBEDDER_PARQUET_FIELDS),
                mode="ignore",
            ),
        ],
        config={
            "host": str(cfg.host),
            "embeddings_dir": str(layout.embeddings_dir),
        },
    )


__all__ = ["build_embed_pipeline"]
