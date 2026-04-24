"""Extractor pipeline: Markdown -> JSONL.

Stage chain::

    MarkdownReader(md_dir)
    -> LegalExtractStage
    -> JsonlWriter(jsonl_dir)

Reads: ``data/<host>/md/*.md`` (+ sibling ``<case_id>.meta.json``).
Writes: ``data/<host>/jsonl/*.jsonl`` with text + regex-based legal
entities + every congbobanan sidebar column carried forward in
:data:`EXTRACTOR_JSONL_FIELDS`.

The precedent normalization layer in :class:`LegalExtractStage` is a
no-op here: congbobanan is a judgment portal, not an án lệ portal, so
``cfg.extractor.run_site_layer`` should stay False and
``precedent_*`` columns stay None.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.congbobanan._shared import (
    EXTRACTOR_JSONL_FIELDS,
    build_layout,
)
from packages.extractor.stage import LegalExtractStage
from packages.pipeline.io import JsonlPerDocWriter, MarkdownReader


def build_extract_pipeline(cfg: Any) -> Pipeline:
    """Return the Extractor :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-extract",
        description="congbobanan Extractor: markdown -> JSONL (legal extract).",
        stages=[
            MarkdownReader(
                file_paths=str(layout.md_dir),
                files_per_partition=int(
                    cfg.get("stage_overrides", {}).get(
                        "extract_files_per_partition", 32
                    )
                ),
            ),
            LegalExtractStage(cfg=cfg),
            JsonlPerDocWriter(
                path=str(layout.jsonl_dir),
                doc_name_field="doc_name",
                fields=list(EXTRACTOR_JSONL_FIELDS),
            ),
        ],
        config={"host": str(cfg.host), "jsonl_dir": str(layout.jsonl_dir)},
    )


__all__ = ["build_extract_pipeline"]
