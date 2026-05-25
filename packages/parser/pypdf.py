"""Local pypdf / docx2txt parser backend.

Pure-Python fallback when a NIM endpoint is unavailable. Supports:

* ``%PDF`` -- via :mod:`pypdf`.
* ``PK\\x03`` (DOCX is a zip)        -- via :mod:`docx2txt`.
* ``\\xD0\\xCF\\x11\\xE0`` (OLE/.doc) -- best-effort via the
  ``antiword`` / ``catdoc`` / ``libreoffice --headless`` CLI tools
  when at least one is on ``PATH``. Pure-Python ``.doc`` extraction
  is hard (the format is OLE-Compound + a piece-table-encoded text
  stream) so we shell out instead of vendoring a partial parser.

Dispatches by magic number so the same :meth:`parse` call handles any
extension::

    %PDF                  -> pypdf
    PK\\x03                -> docx2txt   (DOCX is a ZIP)
    \\xD0\\xCF\\x11\\xE0   -> antiword | catdoc | soffice
    else                  -> log warning, return empty record
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
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
        head = pdf_bytes[:8]
        if head.startswith(b"%PDF"):
            return self._parse_pdf(pdf_bytes)
        if head.startswith(b"PK\x03\x04"):
            return self._parse_docx(pdf_bytes)
        # OLE Compound Document File (legacy MS Office binary). The
        # 8-byte magic also matches .xls / .ppt / .msg, but vbpl /
        # other VN .gov sites only serve .doc through this path -- we
        # let antiword/catdoc decide and fall through to an empty
        # record on failure.
        if head.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
            return self._parse_doc(pdf_bytes)
        logger.warning(
            "PypdfParser: unrecognized magic %r (%d bytes) - skipping",
            head, len(pdf_bytes),
        )
        return {"pages": [], "markdown": "", "confidence": None}

    @staticmethod
    def _parse_pdf(data: bytes) -> dict[str, Any]:
        import io

        import pypdf

        from packages.parser.cmap_healer import heal_pdf_bytes

        # Heal broken Vietnamese ToUnicode CMap entries before pypdf
        # tries to decode glyphs. ~3-5% of the congbobanan / vbpl
        # corpus has one or more <CID> <0020> entries in the Adobe
        # Vietnamese precomposed-vowel block; without the heal,
        # pypdf drops those glyphs as spaces ("đấu" -> "đ u"). The
        # heal is a no-op when no such corruption exists and only
        # incurs pikepdf inspection overhead.
        try:
            healed, patches = heal_pdf_bytes(data)
        except Exception as exc:
            logger.warning(
                "PypdfParser: cmap_healer raised %s: %s; "
                "falling back to raw bytes", type(exc).__name__, exc,
            )
            healed, patches = data, []
        if patches:
            logger.debug(
                "PypdfParser: healed %d Vietnamese CMap entries", len(patches),
            )

        reader = pypdf.PdfReader(io.BytesIO(healed))
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
        result = {
            "pages": pages,
            "markdown": "\n\n".join(md_parts),
            "confidence": None,
        }
        if patches:
            # Surface the heal count for downstream auditing /
            # metrics; the field is opt-in for any consumer that
            # wants to track CMap repair coverage.
            result["cmap_patches"] = len(patches)
        return result

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

    @staticmethod
    def _parse_doc(data: bytes) -> dict[str, Any]:
        """Extract text from a legacy ``.doc`` (Word 97-2003 / OLE) blob.

        Pure-Python ``.doc`` parsing requires walking the OLE compound
        document, decoding the WordDocument stream, and following the
        piece table -- there is no maintained pip-installable library
        that does this end-to-end. We shell out instead.

        Tries, in order:

        1. ``antiword``  (best Vietnamese diacritic preservation,
           apt-installable as ``apt install antiword``).
        2. ``catdoc``    (decent fallback,
           apt-installable as ``apt install catdoc``).
        3. ``soffice``   / ``libreoffice`` headless conversion to
           plain text (heaviest but always works on a host with
           LibreOffice installed).

        If none are present on PATH, logs a one-line install hint and
        returns an empty record so the upstream pipeline can flag the
        document with ``parser_model="local/pypdf"`` + zero markdown
        and a follow-up run can re-parse it once the binary is
        installed.
        """
        text = (
            _try_antiword(data)
            or _try_catdoc(data)
            or _try_libreoffice(data)
        )
        if not text:
            logger.warning(
                "PypdfParser: .doc extraction yielded no text. Install "
                "one of `antiword`, `catdoc`, or `libreoffice` so this "
                "format isn't dropped (apt-get install antiword)."
            )
            return {"pages": [], "markdown": "", "confidence": None}
        # Like DOCX, .doc has no firm page model in the OLE stream.
        return {
            "pages": [{"page_number": 1, "markdown": text, "blocks": []}],
            "markdown": f"## Page 1\n\n{text}",
            "confidence": None,
        }


# ---- .doc subprocess fallbacks --------------------------------------------


def _try_antiword(data: bytes) -> str:
    """Run ``antiword`` over ``data`` from a temp file. Returns the text or ``""``."""
    if shutil.which("antiword") is None:
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            # ``-m UTF-8.txt`` requests a UTF-8 mapping. -w 0 disables
            # column wrapping so paragraphs stay on one line where the
            # downstream extractor can re-split them on sentence
            # boundaries.
            out = subprocess.run(
                ["antiword", "-m", "UTF-8.txt", "-w", "0", path],
                capture_output=True, timeout=60, check=False,
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("antiword crashed: %s", exc)
        return ""
    if out.returncode != 0:
        logger.debug(
            "antiword rc=%d stderr=%r", out.returncode, out.stderr[:200],
        )
        return ""
    return _decode_best_effort(out.stdout).strip()


def _try_catdoc(data: bytes) -> str:
    """Run ``catdoc`` over ``data``. Returns the text or ``""``."""
    if shutil.which("catdoc") is None:
        return ""
    try:
        # catdoc accepts stdin via ``-`` but prefers a real path so
        # the OLE parser can seek; use a temp file.
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as f:
            f.write(data)
            path = f.name
        try:
            out = subprocess.run(
                ["catdoc", "-d", "utf-8", "-w", path],
                capture_output=True, timeout=60, check=False,
            )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("catdoc crashed: %s", exc)
        return ""
    if out.returncode != 0:
        logger.debug(
            "catdoc rc=%d stderr=%r", out.returncode, out.stderr[:200],
        )
        return ""
    return _decode_best_effort(out.stdout).strip()


def _try_libreoffice(data: bytes) -> str:
    """Convert via ``soffice --headless`` to ``.txt`` then read it back.

    This is the heaviest fallback (LibreOffice spins up a one-off
    process per call) but it's the most semantics-preserving for
    Vietnamese diacritics + tables.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        return ""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "in.doc")
            with open(in_path, "wb") as f:
                f.write(data)
            out = subprocess.run(
                [
                    soffice, "--headless", "--convert-to", "txt:Text (encoded):UTF8",
                    "--outdir", tmpdir, in_path,
                ],
                capture_output=True, timeout=120, check=False,
            )
            if out.returncode != 0:
                logger.debug(
                    "soffice rc=%d stderr=%r",
                    out.returncode, out.stderr[:200],
                )
                return ""
            txt_path = os.path.join(tmpdir, "in.txt")
            if not os.path.exists(txt_path):
                return ""
            with open(txt_path, encoding="utf-8", errors="replace") as f:
                return f.read().strip()
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("soffice crashed: %s", exc)
        return ""


def _decode_best_effort(blob: bytes) -> str:
    """Decode a subprocess stdout blob trying UTF-8 then CP1258."""
    if not blob:
        return ""
    for enc in ("utf-8", "cp1258", "latin-1"):
        try:
            return blob.decode(enc)
        except UnicodeDecodeError:
            continue
    return blob.decode("utf-8", errors="replace")


__all__ = ["PypdfParser"]
