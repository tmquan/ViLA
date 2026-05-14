"""Top-level dispatcher for the vbpl ``extract`` pipeline.

Thin wrapper around
:class:`packages.datasites.vbpl.components.extract.VbplDocumentExtractor`
so :func:`packages.datasites.vbpl.scraper.run_pipeline` can dispatch
``--pipeline extract`` like every other stage.

Reads ``data/<host>/md/<scope>/*.md`` + sibling ``<id>.meta.json``;
writes one ``data/<host>/jsonl/extract.jsonl`` row per document
(schema: :data:`packages.datasites.vbpl._shared.EXTRACTOR_JSONL_FIELDS`).

The three layers (Vietnamese normalization + generic NER + structure)
are gated by ``cfg.extractor.run_text_normalization`` /
``run_generic_layer`` / ``run_structure_layer`` -- see the dataset
README for trade-offs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.datasites.vbpl._shared import build_layout
from packages.datasites.vbpl.components import VbplDocumentExtractor


def run_extract(cfg: Any) -> Path:
    """Run normalize + generic + structure layers. Returns ``extract.jsonl`` path."""
    layout = build_layout(cfg)
    return VbplDocumentExtractor(cfg, layout).run()


__all__ = ["run_extract"]
