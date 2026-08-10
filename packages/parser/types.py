"""Vietnamese legal PDF taxonomy + the shared offline/live classifier.

Formalises the Case A-G taxonomy from ``wiki/PARSING.md`` § 5 into a
code-friendly enum (``wiki/PDFExtractor.md`` slide 25b) and exposes
:func:`classify_pdf`, the single function used by both the offline
triage pass (:mod:`packages.parser.triage`) and any live router, so
the two can never disagree about where a document should go.

Classification is **pypdf-only**: no GPU, no network, no rasterization.
It runs the local :class:`~packages.parser.base.ParserAlgorithm` leg
once and reads four signals off the result:

``local_len``
    Length of the stripped whole-document markdown. Below
    ``min_chars`` (50) the PDF has no usable text layer at all --
    an image-only scan.
``lossy``
    :func:`packages.parser.hybrid.lossy_score` -- the fraction of
    lowercase 1-2-char ASCII fragments embedded in body text. Above
    ``max_lossy`` (0.05) the embedded font's ToUnicode table is
    corrupt badly enough that the extracted text is unreadable.
``empty_pages``
    Count of pages that yielded no text while the document as a
    whole is healthy -- a digital body with a scanned exhibit or
    signature page spliced in.
``cmap_patches``
    Number of Adobe-Vietnamese ToUnicode entries repaired by
    :mod:`packages.parser.cmap_healer`. Non-zero means the text is
    clean *because* of the heal, which is worth recording even
    though the document still routes locally.

Thresholds are the same constants the live
:class:`~packages.parser.hybrid.HybridParser` uses, so an offline tag
predicts the live route exactly.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any

from packages.parser.base import ParserAlgorithm
from packages.parser.hybrid import lossy_score

logger = logging.getLogger(__name__)

#: Minimum stripped-markdown length for a PDF to count as having a
#: text layer at all. Matches ``cfg.parser.min_local_chars``.
DEFAULT_MIN_CHARS = 50

#: Maximum tolerable :func:`lossy_score` before the text layer is
#: considered unreadable. Matches ``cfg.parser.max_local_lossy_score``.
DEFAULT_MAX_LOSSY = 0.05

# Magic numbers dispatched by ``PypdfParser.parse``. Kept here so the
# classifier can tag OFFICE_DOCUMENT without invoking the parser.
_MAGIC_PDF = b"%PDF"
_MAGIC_ZIP = b"PK\x03\x04"  # DOCX is a zip
_MAGIC_OLE2 = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy .doc


class VietnameseLegalPDFType(StrEnum):
    """The eight document shapes a Vietnamese legal corpus contains.

    The first three are **native** -- a text layer exists and is
    trustworthy, so they can be extracted on CPU with no OCR. The
    remaining five are **deferred**: either they need a vision model
    (``FONT_CORRUPTED`` / ``SCANNED_IMAGE`` / ``MIXED_PAGES``) or they
    need byte-level repair before they can even be classified again
    (``ENCRYPTED`` / ``CORRUPTED``).
    """

    NATIVE_DIGITAL = "native_digital"  # was Case A
    CMAP_REPAIRABLE = "cmap_repairable"  # was Case B
    FONT_CORRUPTED = "font_corrupted"  # was Case B'
    SCANNED_IMAGE = "scanned_image"  # was Case C
    MIXED_PAGES = "mixed_pages"  # was Case D
    OFFICE_DOCUMENT = "office_document"  # was Case E (.docx/.doc)
    ENCRYPTED = "encrypted"  # was Case F
    CORRUPTED = "corrupted"  # was Case G


T = VietnameseLegalPDFType

#: Types whose text layer is machine-readable as-is. These are
#: extracted by the ``pdf_native`` pipeline.
NATIVE_TYPES: frozenset[VietnameseLegalPDFType] = frozenset(
    {T.NATIVE_DIGITAL, T.CMAP_REPAIRABLE, T.OFFICE_DOCUMENT}
)

#: Types that need a vision model. Deferred with ``deferred_class="ocr"``.
OCR_TYPES: frozenset[VietnameseLegalPDFType] = frozenset(
    {T.FONT_CORRUPTED, T.SCANNED_IMAGE, T.MIXED_PAGES}
)

#: Types that need byte-level repair before re-triage. Deferred with
#: ``deferred_class="repair"`` -- no OCR model applies until the file
#: can actually be opened.
REPAIR_TYPES: frozenset[VietnameseLegalPDFType] = frozenset(
    {T.ENCRYPTED, T.CORRUPTED}
)

#: Everything the ``pdf_native`` pipeline must not touch.
DEFERRED_TYPES: frozenset[VietnameseLegalPDFType] = OCR_TYPES | REPAIR_TYPES

#: One-line human-readable justification per type, written into the
#: deferred manifest's ``reason`` field.
DETECTION_REASON: dict[VietnameseLegalPDFType, str] = {
    T.NATIVE_DIGITAL: "embedded text layer extracted cleanly by pypdf",
    T.CMAP_REPAIRABLE: (
        "embedded text layer extracted cleanly after repairing broken "
        "Adobe-Vietnamese ToUnicode CMap entries"
    ),
    T.FONT_CORRUPTED: (
        "text layer present but unreadable: embedded font lacks ToUnicode "
        "entries for most glyphs, so extracted text is shredded into short "
        "fragments"
    ),
    T.SCANNED_IMAGE: (
        "no usable text layer: the PDF is an image-only scan, pypdf "
        "extracts nothing"
    ),
    T.MIXED_PAGES: (
        "mixed digital and scanned pages: the body has a text layer but "
        "one or more pages are image-only"
    ),
    T.OFFICE_DOCUMENT: "Office document extracted via docx2txt / antiword",
    T.ENCRYPTED: (
        "PDF is encrypted or password-protected and cannot be opened for "
        "text extraction"
    ),
    T.CORRUPTED: (
        "PDF is malformed and cannot be opened: truncated stream, invalid "
        "xref table, or unrecognized file magic"
    ),
}


def deferred_class(pdf_type: VietnameseLegalPDFType) -> str | None:
    """Return ``"ocr"``, ``"repair"``, or ``None`` for native types."""
    if pdf_type in OCR_TYPES:
        return "ocr"
    if pdf_type in REPAIR_TYPES:
        return "repair"
    return None


def is_native(pdf_type: VietnameseLegalPDFType) -> bool:
    """True when the document can be extracted without OCR or a VLM."""
    return pdf_type in NATIVE_TYPES


def _office_type(head: bytes) -> VietnameseLegalPDFType | None:
    """Tag DOCX / legacy DOC by magic number, or ``None`` if not Office."""
    if head.startswith(_MAGIC_ZIP) or head.startswith(_MAGIC_OLE2):
        return T.OFFICE_DOCUMENT
    return None


def _is_encrypted(pdf_bytes: bytes) -> bool:
    """Probe whether pypdf considers this PDF encrypted.

    Encryption is invisible in the local parser's output. ``PdfReader``
    does not raise on an encrypted file -- it constructs fine and sets
    ``is_encrypted``; the failure surfaces later when
    :meth:`packages.parser.pypdf.PypdfParser._parse_pdf` touches
    ``reader.pages``, and that exception is swallowed by the defensive
    handler at ``pypdf.py:161``. The result is an empty record with no
    ``parse_error`` key, byte-identical to what an image-only scan
    produces.

    That matters because the two need opposite remediation: an
    encrypted file wants ``qpdf --decrypt`` and a re-triage, while a
    scan wants a vision model. So the classifier probes directly
    whenever the text layer comes back empty.

    Returns ``False`` on any failure: an unopenable PDF is
    :attr:`~VietnameseLegalPDFType.CORRUPTED`, not encrypted, and the
    caller's corrupted branch handles it.
    """
    try:
        import io

        import pypdf

        return bool(pypdf.PdfReader(io.BytesIO(pdf_bytes)).is_encrypted)
    except Exception:
        return False


def classify_pdf(
    pdf_bytes: bytes,
    *,
    local: ParserAlgorithm,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_lossy: float = DEFAULT_MAX_LOSSY,
) -> tuple[VietnameseLegalPDFType, dict[str, Any]]:
    """Tag one document and return ``(pdf_type, signals)``.

    ``local`` must be a pypdf-backed :class:`ParserAlgorithm` (i.e.
    :class:`~packages.parser.pypdf.PypdfParser`). Passing an OCR client
    here would defeat the point: the classifier exists precisely to
    decide whether an OCR client is needed.

    Never raises. Any unexpected failure inside the local parser is
    reported as :attr:`~VietnameseLegalPDFType.CORRUPTED` with the
    exception text in ``signals["error"]``, so a single bad file in a
    million-document corpus cannot take down a Ray worker.
    """
    signals: dict[str, Any] = {
        "local_len": 0,
        "lossy": 0.0,
        "cmap_patches": 0,
        "empty_pages": 0,
        "num_pages": 0,
    }

    if not pdf_bytes:
        signals["error"] = "empty file"
        return T.CORRUPTED, signals

    head = pdf_bytes[:8]
    office = _office_type(head)

    try:
        result = local.parse(pdf_bytes)
    except Exception as exc:
        logger.warning(
            "classify_pdf: local parser raised %s: %s",
            type(exc).__name__,
            exc,
        )
        signals["error"] = f"{type(exc).__name__}: {exc}"
        return T.CORRUPTED, signals

    markdown = str(result.get("markdown") or "").strip()
    pages = list(result.get("pages") or [])
    signals["local_len"] = len(markdown)
    signals["lossy"] = lossy_score(markdown)
    signals["cmap_patches"] = int(result.get("cmap_patches") or 0)
    signals["num_pages"] = len(pages)
    signals["empty_pages"] = sum(
        1 for p in pages if not str(p.get("markdown") or "").strip()
    )

    # ``parse_error`` means pypdf could not open the file at all. Split
    # encrypted from corrupted so the deferred manifest can prescribe
    # the right remediation.
    parse_error = result.get("parse_error")
    if parse_error:
        signals["error"] = str(parse_error)
        if head.startswith(_MAGIC_PDF) and _is_encrypted(pdf_bytes):
            return T.ENCRYPTED, signals
        return T.CORRUPTED, signals

    # An Office document with text is native; one without is broken
    # (antiword/catdoc missing, or an unsupported OLE2 subtype such as
    # .xls served with a .doc extension). Rasterizing it would not help,
    # so it is a repair case, not an OCR case.
    if office is not None:
        if signals["local_len"] < min_chars:
            signals["error"] = "Office document yielded no extractable text"
            return T.CORRUPTED, signals
        return T.OFFICE_DOCUMENT, signals

    # Unrecognized magic: PypdfParser logs and returns an empty record
    # rather than raising, so it arrives here indistinguishable from a
    # scan. Catch it on the magic instead.
    if not head.startswith(_MAGIC_PDF):
        signals["error"] = f"unrecognized file magic {head[:4]!r}"
        return T.CORRUPTED, signals

    # An empty text layer is ambiguous: image-only scan, or an
    # encrypted file whose ``reader.pages`` access was swallowed
    # upstream. Probe before defaulting to the OCR route -- rasterizing
    # an encrypted PDF would burn GPU time and still return nothing.
    if signals["local_len"] < min_chars:
        if _is_encrypted(pdf_bytes):
            signals["error"] = "PDF is encrypted; text layer inaccessible"
            return T.ENCRYPTED, signals
        return T.SCANNED_IMAGE, signals
    if signals["lossy"] > max_lossy:
        return T.FONT_CORRUPTED, signals
    if signals["empty_pages"] > 0:
        return T.MIXED_PAGES, signals
    if signals["cmap_patches"] > 0:
        return T.CMAP_REPAIRABLE, signals
    return T.NATIVE_DIGITAL, signals


__all__ = [
    "DEFAULT_MAX_LOSSY",
    "DEFAULT_MIN_CHARS",
    "DEFERRED_TYPES",
    "DETECTION_REASON",
    "NATIVE_TYPES",
    "OCR_TYPES",
    "REPAIR_TYPES",
    "VietnameseLegalPDFType",
    "classify_pdf",
    "deferred_class",
    "is_native",
]
