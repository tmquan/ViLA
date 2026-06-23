"""dichvucong DocumentIterator: explode a cached page into records.

Subclasses :class:`nemo_curator.stages.text.download.base.DocumentIterator`.
A downloaded page file holds a JSON array of procedure records; this
yields **one record per procedure**, tagging each with its source page
file so the extractor can derive provenance.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.download.base import DocumentIterator

logger = logging.getLogger(__name__)


class DichvucongDocumentIterator(DocumentIterator):
    """One record per procedure in a cached search-result page."""

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        p = Path(file_path)
        try:
            rows = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
        except Exception as exc:
            logger.error("bad page json %s: %s", file_path, exc)
            return
        for row in rows:
            if not isinstance(row, dict):
                continue
            yield {"page_path": str(p), "record": row}

    def output_columns(self) -> list[str]:
        return ["page_path", "record"]


__all__ = ["DichvucongDocumentIterator"]
