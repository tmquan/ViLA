"""Anle DocumentIterator: one downloaded binary -> one record.

Subclasses :class:`nemo_curator.stages.text.download.base.DocumentIterator`.
Given the path to a PDF/DOCX/DOC written by :class:`AnlePDFDownloader` into
``files/``, emits exactly one dict record carrying:

* ``doc_name``    - stable slug (filename stem).
* ``pdf_path``    - absolute path of the downloaded binary.
* ``pdf_bytes``   - raw binary payload for downstream PDF parsing.
* ``detail_html`` - the detail page the downloader saved to
  ``pages/<doc>.html.gz`` (gunzipped here; empty string if absent).
* ``detail_url``  - reconstructed from the detail-URL template.

The detail HTML lives in the sibling ``pages/`` directory (``files/`` and
``pages/`` share a parent), so we look one level up + over.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.download.base import DocumentIterator

from packages.datasites.anle.components.url_generator import DEFAULT_DETAIL_TEMPLATE


class AnleIterator(DocumentIterator):
    """One record per anle document."""

    def __init__(
        self,
        *,
        pages_dir: str | None = None,
        detail_url_template: str = DEFAULT_DETAIL_TEMPLATE,
    ) -> None:
        self._pages_dir = Path(pages_dir) if pages_dir else None
        self._detail_url_template = detail_url_template

    def _detail_html_path(self, binary_path: Path, stem: str) -> Path:
        pages = self._pages_dir or (binary_path.parent.parent / "pages")
        return pages / f"{stem}.html.gz"

    def _read_detail_html(self, path: Path) -> str:
        if not path.exists():
            return ""
        raw = path.read_bytes()
        try:
            return gzip.decompress(raw).decode("utf-8")
        except (OSError, gzip.BadGzipFile):
            return raw.decode("utf-8", errors="replace")

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        p = Path(file_path)
        stem = p.stem
        pdf_bytes = p.read_bytes() if p.exists() else b""
        detail_html = self._read_detail_html(self._detail_html_path(p, stem))
        yield {
            "doc_name": stem,
            "pdf_path": str(p),
            "pdf_bytes": pdf_bytes,
            "detail_html": detail_html,
            "detail_url": self._detail_url_template.format(doc_name=stem),
        }

    def output_columns(self) -> list[str]:
        return ["doc_name", "pdf_path", "pdf_bytes", "detail_html", "detail_url"]


__all__ = ["AnleIterator"]
