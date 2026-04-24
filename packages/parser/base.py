"""Abstract base class for parser algorithms.

Mirrors the NVIDIA NeMo Curator pattern at
``nemo_curator/stages/text/download/html_extractors/base.py``: one tiny
ABC (:class:`ParserAlgorithm`) whose concrete subclasses live in
sibling files named after the backing library (``nemotron.py``,
``pypdf.py``). The :class:`~packages.parser.stage.ParseStage`
selects an implementation at runtime based on ``cfg.parser.runtime``.

The parse contract normalizes every backend's output to the same dict
shape::

    {
        "pages":    list[dict[str, Any]],   # per-page layout records
        "markdown": str,                     # full markdown body
        "confidence": float | None,          # optional parser confidence
    }

Sites consume this via :class:`~packages.parser.stage.ParseResult` and
write one ``md/<doc_id>.md`` + one ``json/<doc_id>.json`` per input.
"""

from __future__ import annotations

import abc
from typing import Any


class ParserAlgorithm(abc.ABC):
    """Backend-agnostic PDF/DOCX parser.

    Concrete subclasses live in :mod:`packages.parser.nemotron`
    (NVIDIA ``nemotron-parse`` NIM endpoint) and
    :mod:`packages.parser.pypdf` (local pypdf / docx2txt fallback).
    """

    #: Human-readable backend name (``"nim"`` / ``"local"`` / ...).
    runtime: str = ""

    #: ``model_id`` used in on-disk records for provenance.
    model_id: str = ""

    @abc.abstractmethod
    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        """Parse one document's raw bytes.

        Returns a dict with keys ``pages`` (list), ``markdown`` (str),
        and ``confidence`` (float or None).
        """


__all__ = ["ParserAlgorithm"]
