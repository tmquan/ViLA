"""Curator-pipeline factories shared across Curator-based datasites.

The Curator pipeline shape for stages 3-5 (extract / embed / reduce) is
identical across ``anle`` and ``congbobanan``: only the field
projection lists, the per-partition fan-out, and the description
string vary by site. This module exposes one factory per stage that
takes those bits as parameters, so each datasite's
``build_<stage>_pipeline`` becomes a 3-line wrapper that supplies its
schema constants.

These factories assume the upstream contract:

* ``layout`` is built by :func:`packages.common.build_layout` so the
  output directories already exist.
* The extract layer reads ``layout.md_dir`` and writes
  ``layout.jsonl_dir`` keyed by ``doc_name``.
* The embed layer reads ``layout.jsonl_dir`` and writes
  ``layout.embeddings_dir``.
* The reduce layer reads ``layout.embeddings_dir`` and writes
  ``layout.reduced_dir``.

Per-site stage builders (parse, download) stay in their own modules
because they need site-specific iterator / extractor / generator
classes that don't fit a uniform signature.
"""

from __future__ import annotations

from typing import Any, Sequence

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.io.reader import JsonlReader, ParquetReader

from packages.common import SiteLayout
from packages.embedder.stage import build_embedder_stage
from packages.extractor.normalizers import build_normalizer_chain
from packages.extractor.stage import LegalExtractStage
from packages.pipeline.io import (
    JsonlPerDocWriter,
    MarkdownReader,
    ParquetPerDocWriter,
)
from packages.reducer.stage import ReducerStage


# Default ``files_per_partition`` for each stage. Sites with smaller
# corpora typically lower these; sites with larger corpora raise them.
# Override per-site via ``cfg.stage_overrides.<stage>_files_per_partition``.
DEFAULT_FPP: dict[str, int] = {
    "extract": 8,
    "embed": 4,
    "reduce": 64,
}


def _stage_override(cfg: Any, key: str, default: int) -> int:
    """Pull ``cfg.stage_overrides.<key>`` with an int fallback."""
    return int(cfg.get("stage_overrides", {}).get(key, default))


# ----------------------------------------------------- extract


def build_extract_pipeline(
    cfg: Any,
    *,
    site: str,
    layout: SiteLayout,
    jsonl_fields: Sequence[str],
    description: str | None = None,
    files_per_partition: int | None = None,
) -> Pipeline:
    """Return the Extractor :class:`Pipeline`: markdown -> JSONL.

    ``site`` is used to build the canonical pipeline name + default
    description; ``jsonl_fields`` is the projected output schema
    (drives :class:`JsonlPerDocWriter`'s column whitelist).
    """
    fpp = files_per_partition or _stage_override(
        cfg, "extract_files_per_partition", DEFAULT_FPP["extract"],
    )
    # Declarative normalizer chain (wiki.md §3.5). When
    # ``cfg.extractor.normalizers`` is non-empty the chain runs as a
    # Curator ``ProcessingStage`` between the reader and
    # ``LegalExtractStage`` — visible in ``Pipeline.describe()`` and
    # recorded in the per-stage manifest for reproducibility.
    stages: list[Any] = [
        MarkdownReader(
            file_paths=str(layout.md_dir),
            files_per_partition=fpp,
        ),
    ]
    chain = build_normalizer_chain(cfg)
    if chain is not None:
        stages.append(chain)
    stages.extend([
        LegalExtractStage(cfg=cfg),
        JsonlPerDocWriter(
            path=str(layout.jsonl_dir),
            doc_name_field="doc_name",
            fields=list(jsonl_fields),
        ),
    ])
    return Pipeline(
        name=f"{cfg.host}-extract",
        description=(
            description
            or f"{site} Extractor: markdown -> JSONL (legal extract)."
        ),
        stages=stages,
        config={"host": str(cfg.host), "jsonl_dir": str(layout.jsonl_dir)},
    )


# ----------------------------------------------------- embed


def build_embed_pipeline(
    cfg: Any,
    *,
    site: str,
    layout: SiteLayout,
    read_fields: Sequence[str],
    parquet_fields: Sequence[str],
    description: str | None = None,
    files_per_partition: int | None = None,
    jsonl_path: str | None = None,
    parquet_path: str | None = None,
) -> Pipeline:
    """Return the Embedder :class:`Pipeline`: extract → embeddings parquet.

    Input source resolution (highest precedence first):

    * ``parquet_path`` -- canonical wiki.md §3.5 pattern. Reads
      ``parquet/extract/*.parquet`` directly via :class:`ParquetReader`,
      saving one JSONL ↔ parquet hop. Recommended for new sites.
    * ``jsonl_path`` -- legacy single-file JSONL override
      (``data/<host>/jsonl/extract.jsonl``) for sites whose extract
      stage emits a consolidated file.
    * default -- read every ``*.jsonl`` under ``layout.jsonl_dir``
      (per-doc shard pattern, matches anle / congbobanan).
    """
    fpp = files_per_partition or _stage_override(
        cfg, "embed_files_per_partition", DEFAULT_FPP["embed"],
    )
    if parquet_path is not None:
        reader = ParquetReader(
            file_paths=parquet_path,
            fields=list(read_fields),
            files_per_partition=fpp,
        )
        input_path = parquet_path
        input_key = "parquet_path"
    else:
        file_paths = (
            jsonl_path if jsonl_path is not None else str(layout.jsonl_dir)
        )
        reader = JsonlReader(
            file_paths=file_paths,
            fields=list(read_fields),
            files_per_partition=fpp,
        )
        input_path = file_paths
        input_key = "jsonl_path"
    return Pipeline(
        name=f"{cfg.host}-embed",
        description=(
            description
            or f"{site} Embedder: extract → embeddings parquet."
        ),
        stages=[
            reader,
            build_embedder_stage(cfg),
            ParquetPerDocWriter(
                path=str(layout.embeddings_dir),
                doc_name_field="doc_name",
                fields=list(parquet_fields),
            ),
        ],
        config={
            "host": str(cfg.host),
            input_key: input_path,
            "embeddings_dir": str(layout.embeddings_dir),
        },
    )


# ----------------------------------------------------- reduce


def build_reduce_pipeline(
    cfg: Any,
    *,
    site: str,
    layout: SiteLayout,
    embedder_fields: Sequence[str],
    reducer_fields: Sequence[str],
    description: str | None = None,
    files_per_partition: int | None = None,
) -> Pipeline:
    """Return the Reducer :class:`Pipeline`: embeddings -> reduced parquet."""
    fpp = files_per_partition or _stage_override(
        cfg, "reduce_files_per_partition", DEFAULT_FPP["reduce"],
    )
    return Pipeline(
        name=f"{cfg.host}-reduce",
        description=(
            description
            or f"{site} Reducer: embeddings parquet -> reduced parquet."
        ),
        stages=[
            ParquetReader(
                file_paths=str(layout.embeddings_dir),
                fields=list(embedder_fields),
                files_per_partition=fpp,
            ),
            ReducerStage(cfg=cfg),
            ParquetPerDocWriter(
                path=str(layout.reduced_dir),
                doc_name_field="doc_name",
                fields=list(reducer_fields),
            ),
        ],
        config={"host": str(cfg.host), "reduced_dir": str(layout.reduced_dir)},
    )


__all__ = [
    "DEFAULT_FPP",
    "build_embed_pipeline",
    "build_extract_pipeline",
    "build_reduce_pipeline",
]
