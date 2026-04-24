"""Hybrid pypdf -> nemotron-parse parser backend.

Tries the free local :class:`PypdfParser` first. When the local path
returns empty / near-empty markdown -- the tell-tale sign of an
image-only scan that pypdf cannot extract text from -- falls back to
the NIM ``nemotron-parse`` endpoint, which has OCR + layout detection
built in.

Output shape is the :class:`ParserAlgorithm` contract (``pages``,
``markdown``, ``confidence``), so downstream stages (``PdfParseStage``
and onward) never know which backend produced a given document. A
``parser_backend`` key is added to the response for operational
traceability but is not required by the schema.

Why a threshold and not "strictly empty"? pypdf occasionally extracts
a stray header/footer ("Page 1 of 3") from a scan and returns a
handful of characters -- formally non-empty, but not real content.
``min_chars`` (default 50) rejects that and still routes to NIM.
"""

from __future__ import annotations

import logging
from typing import Any

from packages.parser.base import ParserAlgorithm

logger = logging.getLogger(__name__)


class HybridParser(ParserAlgorithm):
    """pypdf first, nemotron-parse fallback on empty / near-empty output."""

    runtime = "hybrid"

    def __init__(
        self,
        local: ParserAlgorithm,
        nim: ParserAlgorithm,
        *,
        min_chars: int = 50,
    ) -> None:
        self.local = local
        self.nim = nim
        self._min_chars = int(min_chars)
        # Advertise both model IDs so the downstream ``parser_model``
        # column captures which backends are in play for this site.
        self.model_id = f"{local.model_id}+{nim.model_id}"

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        local_result = self.local.parse(
            pdf_bytes, preserve_tables=preserve_tables
        )
        local_md = str(local_result.get("markdown") or "").strip()

        if len(local_md) >= self._min_chars:
            # Local path produced real text; keep it.
            local_result["parser_backend"] = getattr(
                self.local, "runtime", "local"
            )
            return local_result

        logger.info(
            "HybridParser: local parser produced %d chars "
            "(threshold=%d); falling back to NIM %s",
            len(local_md), self._min_chars, self.nim.model_id,
        )
        try:
            nim_result = self.nim.parse(
                pdf_bytes, preserve_tables=preserve_tables
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "HybridParser: NIM fallback failed (%s: %s); "
                "keeping local's %d-char output",
                type(exc).__name__, exc, len(local_md),
            )
            local_result["parser_backend"] = getattr(
                self.local, "runtime", "local"
            )
            local_result["nim_fallback_error"] = f"{type(exc).__name__}: {exc}"
            return local_result

        nim_md = str(nim_result.get("markdown") or "").strip()
        if not nim_md:
            logger.warning(
                "HybridParser: both local and NIM returned empty markdown "
                "(this PDF is unreadable; probably corrupted or locked)"
            )
        nim_result["parser_backend"] = getattr(self.nim, "runtime", "nim")
        return nim_result


__all__ = ["HybridParser"]
