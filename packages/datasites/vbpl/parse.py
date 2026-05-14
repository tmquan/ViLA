"""Top-level dispatcher for the vbpl ``parse`` pipeline.

Thin wrapper around
:class:`packages.datasites.vbpl.components.parse.VbplDocumentParser`
so :func:`packages.datasites.vbpl.scraper.run_pipeline` can dispatch
``--pipeline parse`` like every other stage.

Reads ``data/<host>/jsonl/docs.jsonl`` + the on-disk artefacts the
detail stage left behind; writes ``data/<host>/md/<scope>/<id>.md``
+ ``<id>.meta.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.datasites.vbpl._shared import build_layout
from packages.datasites.vbpl.components import VbplDocumentParser


def run_parse(cfg: Any) -> Path:
    """Walk docs.jsonl + cached files, write markdown. Returns ``md_dir``."""
    layout = build_layout(cfg)
    return VbplDocumentParser(cfg, layout).run()


__all__ = ["run_parse"]
