"""Local pypdf / docx2txt parser backend.

Pure-Python fallback when a NIM endpoint is unavailable. Supports PDF
(via :mod:`pypdf`) and DOCX (via :mod:`docx2txt`); legacy ``.doc``
binaries are not supported and are skipped with a warning.

Dispatches by magic number so the same :meth:`parse` call handles any
extension::

    %PDF       -> pypdf
    PK\\x03     -> docx2txt  (DOCX is a ZIP)
    else       -> best-effort (log warning, return empty record)
"""

from __future__ import annotations

import logging
from typing import Any

from packages.parser.base import ParserAlgorithm

logger = logging.getLogger(__name__)


# pypdf emits a lot of xref-recovery chatter at WARNING level on
# slightly malformed PDFs ("Ignoring wrong pointing object 69 0
# (offset 0)", "Multiple definitions in dictionary", ...). None of
# these indicate actual parse failures -- pypdf recovers and still
# extracts text. We route them below WARNING so they do not drown
# out real pipeline-level logs. ``--log-level DEBUG`` (or
# ``logging.getLogger("pypdf").setLevel(logging.NOTSET)``) restores
# the full chatter when diagnosing a specific document.
logging.getLogger("pypdf").setLevel(logging.ERROR)
# pypdf.generic emits a second tranche from its object-resolver layer.
logging.getLogger("pypdf.generic").setLevel(logging.ERROR)


class PypdfParser(ParserAlgorithm):
    """Pure-Python local parser (PDF via pypdf, DOCX via docx2txt)."""

    runtime = "local"
    model_id = "local/pypdf"

    def __init__(self) -> None:
        try:
            import pypdf  # noqa: F401
        except ImportError as exc:  # pragma: no cover - import-time check
            raise RuntimeError(
                "runtime=local requires `pypdf`. Install with `pip install pypdf`."
            ) from exc

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        head = pdf_bytes[:4]
        if head.startswith(b"%PDF"):
            return self._parse_pdf(pdf_bytes)
        if head.startswith(b"PK\x03\x04"):
            return self._parse_docx(pdf_bytes)
        logger.warning(
            "PypdfParser: unrecognized magic %r (%d bytes) - skipping",
            head, len(pdf_bytes),
        )
        return {"pages": [], "markdown": "", "confidence": None}

    @staticmethod
    def _parse_pdf(data: bytes) -> dict[str, Any]:
        import io

        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(data))
        pages: list[dict[str, Any]] = []
        md_parts: list[str] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            md = text.strip()
            pages.append({"page_number": i, "markdown": md, "blocks": []})
            if md:
                md_parts.append(f"## Page {i}\n\n{md}")
        return {"pages": pages, "markdown": "\n\n".join(md_parts), "confidence": None}

    @staticmethod
    def _parse_docx(data: bytes) -> dict[str, Any]:
        import io

        try:
            import docx2txt
        except ImportError:  # pragma: no cover - optional dep
            logger.warning("docx2txt not installed - skipping DOCX")
            return {"pages": [], "markdown": "", "confidence": None}
        text = docx2txt.process(io.BytesIO(data)) or ""
        text = text.strip()
        # DOCX has no native paging; treat the whole document as one logical page.
        if not text:
            return {"pages": [], "markdown": "", "confidence": None}
        return {
            "pages": [{"page_number": 1, "markdown": text, "blocks": []}],
            "markdown": f"## Page 1\n\n{text}",
            "confidence": None,
        }


__all__ = ["PypdfParser"]
