"""congbobanan DocumentIterator.

One record per downloaded body file (``files/<case_id>.<ext>``). Reads
the binary bytes plus the sibling detail HTML the downloader cached at
``pages/<case_id>.html.gz``, then yields a flat dict the extractor
enriches. The detail URL is reconstructed from the ``case_id`` via the
detail-page template so no ``.url`` sidecar is needed.
"""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.download.base import DocumentIterator

from packages.datasites.congbobanan.components.url_generator import (
    DEFAULT_DETAIL_URL_TEMPLATE,
)


class CBBADocumentIterator(DocumentIterator):
    """One record per congbobanan case.

    ``pages_dir`` locates the gzipped detail HTML. When ``None`` it is
    derived as ``<body_file>.parent.parent / "pages"`` (the canonical
    ``files/`` + ``pages/`` sibling layout).
    """

    def __init__(
        self,
        pages_dir: str | None = None,
        *,
        detail_template: str = DEFAULT_DETAIL_URL_TEMPLATE,
    ) -> None:
        self.pages_dir = Path(pages_dir) if pages_dir else None
        self.detail_template = detail_template or DEFAULT_DETAIL_URL_TEMPLATE

    def _read_detail_html(self, body_path: Path, case_id: str) -> str:
        pages_dir = self.pages_dir or (body_path.parent.parent / "pages")
        gz = pages_dir / f"{case_id}.html.gz"
        if gz.exists():
            try:
                return gzip.decompress(gz.read_bytes()).decode("utf-8", "replace")
            except OSError:
                return ""
        plain = pages_dir / f"{case_id}.html"
        if plain.exists():
            return plain.read_text(encoding="utf-8", errors="replace")
        return ""

    def iterate(self, file_path: str) -> Iterator[dict[str, Any]]:
        p = Path(file_path)
        case_id = p.stem
        pdf_bytes = p.read_bytes() if p.exists() else b""
        detail_html = self._read_detail_html(p, case_id)
        detail_url = self.detail_template.format(case_id=case_id)
        yield {
            # ``doc_name`` is the integer case_id as a string so it can be a
            # filesystem / parquet join key without coercion churn.
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


__all__ = ["CBBADocumentIterator"]
