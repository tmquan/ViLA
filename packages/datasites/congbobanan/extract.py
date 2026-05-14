"""Extractor pipeline factory for congbobanan: markdown -> JSONL.

The precedent normalization layer in :class:`LegalExtractStage` is a
no-op here: congbobanan is a judgment portal, not an án lệ portal, so
``cfg.extractor.run_site_layer`` should stay False and
``precedent_*`` columns stay None.

Stage chain (built by
:func:`packages.pipeline.factories.build_extract_pipeline`)::

    MarkdownReader(md_dir)
    -> LegalExtractStage
    -> JsonlPerDocWriter(jsonl_dir, fields=EXTRACTOR_JSONL_FIELDS)
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline

from packages.datasites.congbobanan._shared import (
    EXTRACTOR_JSONL_FIELDS,
    build_layout,
)
from packages.pipeline.factories import build_extract_pipeline as _build


def build_extract_pipeline(cfg: Any) -> Pipeline:
    """Return the congbobanan Extractor :class:`Pipeline`."""
    return _build(
        cfg,
        site="congbobanan",
        layout=build_layout(cfg),
        jsonl_fields=EXTRACTOR_JSONL_FIELDS,
        files_per_partition=int(
            cfg.get("stage_overrides", {}).get(
                "extract_files_per_partition", 32,
            )
        ),
    )


__all__ = ["build_extract_pipeline"]
