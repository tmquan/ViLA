"""Anle DocumentIterator: turn one downloaded file into one record.

Subclasses :class:`nemo_curator.stages.text.download.base.DocumentIterator`.
Given the path to a PDF/DOCX/DOC written by :class:`AnleDocumentDownloader`,
emits exactly one dict record carrying:

* ``doc_name``    - stable slug (filename stem).
* ``pdf_path``    - absolute path of the downloaded binary.
* ``pdf_bytes``   - raw binary payload for downstream PDF parsing.
* ``detail_html`` - sibling ``<stem>.html`` written by the downloader
  (empty string when ``cfg.scraper.fetch_detail_page`` is False).
* ``detail_url``  - sibling ``<stem>.url`` sidecar.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.download.base import DocumentIterator


class AnleDocumentIterator(DocumentIterator):
    """One record per anle document."""

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        p = Path(file_path)
        stem = p.stem
        pdf_bytes = p.read_bytes() if p.exists() else b""
        html_path = p.with_suffix(".html")
        url_path = p.with_suffix(".url")
        detail_html = (
            html_path.read_text(encoding="utf-8") if html_path.exists() else ""
        )
        detail_url = (
            url_path.read_text(encoding="utf-8").strip()
            if url_path.exists()
            else ""
        )
        yield {
            "doc_name": stem,
            "pdf_path": str(p),
            "pdf_bytes": pdf_bytes,
            "detail_html": detail_html,
            "detail_url": detail_url,
        }

    def output_columns(self) -> list[str]:
        return ["doc_name", "pdf_path", "pdf_bytes", "detail_html", "detail_url"]


__all__ = ["AnleDocumentIterator"]
