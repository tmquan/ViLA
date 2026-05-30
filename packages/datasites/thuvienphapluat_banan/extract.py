"""Extractor pipeline factory for thuvienphapluat_banan: markdown -> JSONL.

Stage chain (built by
:func:`packages.pipeline.factories.build_extract_pipeline`)::

    MarkdownReader(md_dir, recursive=False)
    -> NormalizerChainStage(cfg.extractor.normalizers)
    -> LegalExtractStage(cfg)
    -> JsonlPerDocWriter(jsonl_dir, fields=EXTRACTOR_JSONL_FIELDS)

Reads: ``data/<host>/md/<ban_an_id>.md`` (+ sibling ``.meta.json``).
Writes:

* raw per-doc tier — ``data/<host>/jsonl/<doc>.jsonl`` (one per doc;
  see :data:`packages.datasites.thuvienphapluat_banan._shared.EXTRACTOR_JSONL_FIELDS`).
* parquet consumption tier —
  ``data/<host>/parquet/extract/extract-NNNNN-of-KKKKK.parquet`` (10 K
  rows / shard — thuvienphapluat_banan rows are HTML-derived and lean,
  so the cross-corpus default applies). The coalesce step runs after
  the Curator pipeline finishes, inside
  :func:`packages.datasites.thuvienphapluat_banan.scraper.run_extract`.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.thuvienphapluat_banan._shared import (
    EXTRACTOR_JSONL_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_extract_pipeline as _build


def build_extract_pipeline(cfg: Any) -> Pipeline:
    """Return the thuvienphapluat_banan Extractor :class:`Pipeline`."""
    return _build(
        cfg,
        site="thuvienphapluat_banan",
        layout=build_layout(cfg),
        jsonl_fields=EXTRACTOR_JSONL_FIELDS,
    )


__all__ = ["build_extract_pipeline"]
