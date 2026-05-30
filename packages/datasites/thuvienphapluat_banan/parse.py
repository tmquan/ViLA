"""Top-level dispatcher for the thuvienphapluat_banan ``parse`` pipeline.

Thin wrapper around
:class:`packages.datasites.thuvienphapluat_banan.components.parse.BananDocumentParser`
so :func:`packages.datasites.thuvienphapluat_banan.scraper.run_pipeline`
can dispatch ``--pipeline parse`` like every other stage.

Reads ``data/<host>/jsonl/docs.jsonl`` (the detail-stage output);
writes ``data/<host>/md/<ban_an_id>.md`` + ``<ban_an_id>.meta.json``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from packages.datasites.thuvienphapluat_banan._shared import build_layout
from packages.datasites.thuvienphapluat_banan.components import (
    BananDocumentParser,
)


def run_parse(cfg: Any) -> Path:
    """Walk docs.jsonl + body_html, write markdown. Returns ``md_dir``."""
    layout = build_layout(cfg)
    return BananDocumentParser(cfg, layout).run()


__all__ = ["run_parse"]
