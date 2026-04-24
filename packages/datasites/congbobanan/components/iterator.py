"""congbobanan DocumentIterator.

One record per downloaded case file. Reads the PDF bytes + the sibling
``<case_id>.html`` and ``<case_id>.url`` sidecars the downloader
cached, then yields a flat dict the extractor can enrich.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.download.base import DocumentIterator


class CongbobananDocumentIterator(DocumentIterator):
    """One record per congbobanan case."""

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        p = Path(file_path)
        case_id = p.stem
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
            # ``doc_name`` is the integer case_id rendered as a string so it
            # can be used as a filesystem key / parquet join key without
            # coercion churn.
            "doc_name": case_id,
            "case_id": case_id,
            "pdf_path": str(p),
            "pdf_bytes": pdf_bytes,
            "detail_html": detail_html,
            "detail_url": detail_url,
        }

    def output_columns(self) -> list[str]:
        return [
            "doc_name",
            "case_id",
            "pdf_path",
            "pdf_bytes",
            "detail_html",
            "detail_url",
        ]


__all__ = ["CongbobananDocumentIterator"]
