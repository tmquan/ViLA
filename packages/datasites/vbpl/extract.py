"""Extractor pipeline factory for vbpl: markdown -> JSONL.

Stage chain (built by
:func:`packages.pipeline.factories.build_extract_pipeline`)::

    MarkdownReader(md_dir, recursive)
    -> NormalizerChainStage(cfg.extractor.normalizers)   # vbpl chain
    -> LegalExtractStage(cfg)                            # generic + structure
    -> JsonlPerDocWriter(jsonl_dir, fields=EXTRACTOR_JSONL_FIELDS)

Reads: ``data/<host>/md/<scope>/<id>.md`` (+ sibling ``.meta.json``;
``FilePartitioningStage`` recurses into ``trung_uong/`` /
``dia_phuong/``).
Writes:

* raw per-doc tier — ``data/<host>/jsonl/<doc>.jsonl`` (one per doc;
  see :data:`packages.datasites.vbpl._shared.EXTRACTOR_JSONL_FIELDS`).
* parquet consumption tier — ``data/<host>/parquet/extract/extract-NNNNN-of-KKKKK.parquet``
  (5 K rows / shard for vbpl; see ``configs/default.yaml`` for the
  ``shards.doc_chunk_size`` override justification). The coalesce step
  runs after the Curator pipeline finishes, inside
  :func:`packages.datasites.vbpl.scraper.run_extract`.

This replaces the previous in-process :class:`VbplDocumentExtractor`
driver (under ``components/extract.py``, kept in the tree as legacy
reference but no longer wired into the CLI). The Curator chain is
the single source of truth for normalization + extraction; the
declarative ``cfg.extractor.normalizers`` list is the only knob
operators flip.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.vbpl._shared import (
    EXTRACTOR_JSONL_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_extract_pipeline as _build


def build_extract_pipeline(cfg: Any) -> Pipeline:
    """Return the vbpl Extractor :class:`Pipeline`."""
    return _build(
        cfg,
        site="vbpl",
        layout=build_layout(cfg),
        jsonl_fields=EXTRACTOR_JSONL_FIELDS,
    )


__all__ = ["build_extract_pipeline"]
