"""Extractor pipeline factory for anle: markdown -> JSONL.

Stage chain (built by
:func:`packages.pipeline.factories.build_extract_pipeline`)::

    MarkdownReader(md_dir)
    -> LegalExtractStage
    -> JsonlPerDocWriter(jsonl_dir, fields=EXTRACTOR_JSONL_FIELDS)

Reads: ``data/<host>/md/*.md`` (+ sibling ``<doc_name>.meta.json``).
Writes: ``data/<host>/jsonl/*.jsonl`` with text + extracted entities +
precedent metadata + structure (see :data:`EXTRACTOR_JSONL_FIELDS`).
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.anle._shared import (
    EXTRACTOR_JSONL_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_extract_pipeline as _build


def build_extract_pipeline(cfg: Any) -> Pipeline:
    """Return the anle Extractor :class:`Pipeline`."""
    return _build(
        cfg,
        site="anle",
        layout=build_layout(cfg),
        jsonl_fields=EXTRACTOR_JSONL_FIELDS,
    )


__all__ = ["build_extract_pipeline"]
