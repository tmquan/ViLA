"""Read a stored hoi-dap ``<id>.html.gz`` page into a record."""
from __future__ import annotations

import gzip
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.download import DocumentIterator


class TVPLQAIterator(DocumentIterator):
    """Read a stored ``<id>.html.gz`` -> {file_id, url (canonical), html}."""

    _CANON_RE = re.compile(r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)', re.I)

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        p = Path(file_path)
        html = gzip.decompress(p.read_bytes()).decode("utf-8", "ignore") if p.suffix == ".gz" \
            else p.read_text(encoding="utf-8", errors="ignore")
        fid = re.sub(r"\.html(\.gz)?$", "", p.name)
        m = self._CANON_RE.search(html)
        yield {"file_id": fid, "url": (m.group(1) if m else ""), "html": html}

    def output_columns(self) -> list[str]:
        return ["file_id", "url", "html"]


__all__ = ["TVPLQAIterator"]
