"""congbobanan DocumentDownloader.

Given a detail-page URL produced by :class:`CongbobananURLGenerator`,
this downloader:

1. GETs the detail HTML (best-effort: empty sidebars are tolerated).
2. Attempts the body download at ``/3ta{case_id}t1cvn/`` regardless of
   the detail panel's contents. The body endpoint serves real
   documents for ~30 % of IDs whose detail panel is empty, so gating
   the fetch on :func:`page_has_metadata` (as the reference scraper
   did) wrongly discards them. Validation runs *after* the download
   via :func:`_sniff_body_ext`, which classifies the response by
   magic header (``%PDF-`` / ``PK\\x03\\x04`` / OLE2 / ``{\\rtf``) and
   filters out the 6 475-byte HTML 500-error pages the server returns
   for unrecoverable IDs. The server occasionally serves a judgment
   as DOCX / DOC / RTF rather than PDF -- the downloader accepts all
   four formats and saves the body with the sniffed extension.
3. On success writes ``<case_id>.<sniffed-ext>`` plus sibling ``.html``
   and ``.url`` sidecars that the iterator reads back on the next
   stage. The ``.html`` sidecar may be empty when the detail panel was
   a ghost; the extractor stage then emits ``None`` for the affected
   sidebar columns without losing the document.

The base class's :meth:`download` is fully overridden so we own the
atomic ``.tmp -> final`` rename and can cancel the download before any
bytes are written when validation fails. This mirrors the anle
downloader's approach for the same reason (and for the same
``<doc>.pdf.pdf`` bug prevention).

congbobanan.toaan.gov.vn refuses TLS handshakes from non-Vietnamese
source IPs. Set ``cfg.scraper.proxy`` to a Vietnamese egress, run on a
VN VPS, or export ``HTTPS_PROXY`` -- the polite session picks it up.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.download.base import DocumentDownloader

from packages.common.http import PoliteSession, session_from_scraper_cfg
from packages.datasites.congbobanan.components.url_generator import (
    DEFAULT_PDF_URL_TEMPLATE,
    doc_id_from_url,
)

logger = logging.getLogger(__name__)


def page_has_metadata(html: str) -> bool:
    """Return True if the detail HTML has the real sidebar panel.

    Some IDs return HTTP 200 but the body is a feedback-form ghost with
    no metadata. Match the reference scraper's check: either a "Bản án
    số:" or "Quyết định số:" label plus the ``search_left_pub
    details_pub`` sidebar class.
    """
    if not html:
        return False
    has_case_number = ("Bản án số:" in html) or ("Quyết định số:" in html)
    has_sidebar = "search_left_pub details_pub" in html
    return has_case_number and has_sidebar


#: Minimum byte length accepted as a valid PDF body. The 500-error
#: pages congbobanan serves on broken IDs are ~6 475 bytes of HTML, so
#: a size gate alone is insufficient -- :func:`_is_valid_pdf` also
#: requires the canonical ``%PDF`` magic header.
_MIN_VALID_PDF_BYTES = 1_024

#: Minimum byte length per accepted body extension. RTF can be quite
#: small (a handful of escape sequences); PDF/DOC/DOCX bodies always
#: carry kilobytes of header/metadata even for one-page judgments.
_MIN_VALID_BODY_BYTES: dict[str, int] = {
    ".pdf":  _MIN_VALID_PDF_BYTES,
    ".docx": 1_024,
    ".doc":  1_024,
    ".rtf":  100,
}

#: ``(magic header bytes, target extension)``. Longer / more specific
#: magics first so e.g. OLE2 wins over an accidental ``PK`` prefix.
#: These are the four body formats the congbobanan portal is known to
#: serve; everything else (HTML error pages, JSON API stubs, RAR /
#: PE32 garbage observed in the corpus) is rejected by
#: :func:`_sniff_body_ext` returning ``None``.
_BODY_MAGICS: tuple[tuple[bytes, str], ...] = (
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", ".doc"),   # OLE2 (legacy .doc)
    (b"PK\x03\x04",                       ".docx"),  # Office Open XML
    (b"%PDF-",                            ".pdf"),
    (b"{\\rtf",                           ".rtf"),
)

#: Filename extensions the downloader will write and the parser stage
#: reads. Re-exported so the parse pipeline factory and the existence
#: check stay in sync from one declaration.
ACCEPTED_BODY_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".doc", ".rtf")


def _sniff_body_ext(path: str) -> str | None:
    """Return ``.pdf`` / ``.docx`` / ``.doc`` / ``.rtf`` for a valid body, else ``None``.

    Used by :meth:`CongbobananDocumentDownloader.download` to filter
    the response before promoting ``<id>.body.tmp`` to a final
    ``<id>.<ext>``. Steps:

    1. Match the leading bytes against :data:`_BODY_MAGICS`.
    2. Enforce the per-extension minimum size from
       :data:`_MIN_VALID_BODY_BYTES` so 6 475-byte HTML error pages
       and one-line stubs are rejected even when the magic header
       happens to align.
    3. For PDFs only, additionally require ``%%EOF`` somewhere in the
       last 1 KB of the file -- the previous downloader's MIME check
       let through truncated bodies (connection drops mid-stream) that
       are valid PDFs by header but unusable by the parser. DOCX /
       DOC / RTF have container-level integrity that the parse stage
       validates separately, so we don't replicate that here.
    """
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return None

    for magic, ext in _BODY_MAGICS:
        if not head.startswith(magic):
            continue
        if size < _MIN_VALID_BODY_BYTES.get(ext, _MIN_VALID_PDF_BYTES):
            return None
        if ext == ".pdf":
            try:
                with open(path, "rb") as f:
                    f.seek(max(0, size - 1024))
                    if b"%%EOF" not in f.read(1024):
                        return None
            except OSError:
                return None
        return ext
    return None


def _is_valid_pdf(path: str) -> bool:
    """Return True iff ``path`` validates specifically as a PDF body.

    Thin alias over :func:`_sniff_body_ext` preserved for backward
    compatibility with callers / tests that pre-date multi-format
    support. New code should use :func:`_sniff_body_ext` directly.
    """
    return _sniff_body_ext(path) == ".pdf"


class CongbobananDocumentDownloader(DocumentDownloader):
    """Ghost-aware downloader for congbobanan detail + PDF endpoints."""

    def __init__(
        self,
        cfg: Any,
        download_dir: str,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__(download_dir=download_dir, verbose=verbose)
        self.cfg = cfg

        # The schema default for pdf_url_template is ``""``, so fall
        # back with the empty-string-falsy ``or`` pattern rather than
        # ``.get(key, fallback)`` (which only fires when the key is
        # absent, not when it is the empty string).
        self._pdf_url_template: str = (
            str(cfg.scraper.get("pdf_url_template", ""))
            or DEFAULT_PDF_URL_TEMPLATE
        )
        self._retry_empty_detail: bool = bool(
            cfg.scraper.get("retry_empty_detail", True)
        )
        self._num_workers: int | None = (
            int(cfg.scraper.get("num_workers", 4)) or None
        )
        self._extra_headers: dict[str, str] = {
            str(k): str(v)
            for k, v in (cfg.scraper.get("extra_headers", {}) or {}).items()
        }
        # Built on first use inside download().
        self.session: PoliteSession | None = None

    # --------------------------------------------------- Curator contract

    def download(self, url: str) -> str | None:
        """Fetch one congbobanan case. Returns the final on-disk path or None.

        Two-step fetch: the detail HTML at ``/2ta<id>t1cvn/chi-tiet-ban-an``
        and the body at ``/3ta<id>t1cvn/``. The detail panel is best-
        effort -- some IDs (~30 % of the missing tail) serve a real
        body even though the sidebar HTML returns an empty
        ``details_pub`` div with no ``Bản án số:`` / ``Quyết định số:``
        markers. We still write whatever ``detail_html`` came back so the
        extractor stage can pick up any partial sidebar fields; rows with
        a fully-empty panel produce ``None`` sidebar columns in parquet,
        not a missing document.

        Existence check covers every extension in
        :data:`ACCEPTED_BODY_EXTENSIONS` because a previous run may
        have already saved the same case as ``<id>.docx`` / ``.doc`` /
        ``.rtf`` instead of ``.pdf``; we must not redownload in that
        case.

        A body is accepted only when :func:`_sniff_body_ext` classifies
        it as one of the four supported formats. The session
        ``download()`` call additionally enforces the retry budget on
        HTTP 5xx / connection failures; bodies that arrive in any
        other format (HTML error pages, JSON API stubs, etc.) are
        deleted post-download.
        """
        case_id = doc_id_from_url(url)
        if not case_id:
            logger.error("could not derive case_id from url %s", url)
            return None

        download_dir = Path(self._download_dir)
        existing = self._existing_body_path(case_id, download_dir)
        if existing is not None:
            if self._verbose:
                logger.info("file %s exists; not downloading", existing)
            return str(existing)

        self._ensure_session()
        assert self.session is not None

        try:
            # Detail HTML is best-effort: some IDs serve a real body
            # even when the sidebar panel is empty. ``_retry_empty_detail``
            # still applies because the WAF occasionally returns a
            # stub on the first hit even for valid records.
            detail_html = self._fetch_detail_html(url)
            if not page_has_metadata(detail_html) and self._retry_empty_detail:
                detail_html = self._fetch_detail_html(url)

            body_url = self._pdf_url_template.format(case_id=case_id)
            # Generic tmp name -- we don't yet know the extension and
            # don't want to imply ``.pdf`` if the response is DOCX.
            tmp_path = str(download_dir / f"{case_id}.body.tmp")
            try:
                # ``expected_mime`` is left unset: the server lies
                # about Content-Type (claims ``application/pdf`` even
                # for DOCX bodies), so MIME-level gating discards
                # valid documents. Magic-header validation runs
                # post-download via :func:`_sniff_body_ext`.
                self.session.download(body_url, tmp_path)
            except Exception as body_exc:
                # Most "missing" IDs land here: body endpoint 500s
                # consistently after the retry budget.
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                logger.info(
                    "case %s: body unavailable (%s); skipping",
                    case_id, body_exc,
                )
                return None

            ext = _sniff_body_ext(tmp_path)
            if ext is None:
                # 200-with-HTML-payload (a 6 475-byte error page),
                # truncated PDFs, and exotic content (JSON / RAR /
                # PE32 garbage observed in the corpus) fall here.
                os.unlink(tmp_path)
                logger.info(
                    "case %s: body not a recognised document format; skipping",
                    case_id,
                )
                return None

            final_path = download_dir / f"{case_id}{ext}"
            os.replace(tmp_path, final_path)
            final_path.with_suffix(".html").write_text(
                detail_html or "", encoding="utf-8",
            )
            final_path.with_suffix(".url").write_text(url, encoding="utf-8")

            if not page_has_metadata(detail_html):
                logger.info(
                    "case %s: downloaded %s without sidebar metadata "
                    "(detail panel empty)", case_id, ext,
                )
            elif ext != ".pdf":
                logger.info(
                    "case %s: downloaded as %s (non-PDF body)",
                    case_id, ext,
                )
            elif self._verbose:
                logger.info("downloaded %s to %s", url, final_path)
            return str(final_path)

        except Exception as exc:
            logger.error("download failed for %s: %s", url, exc)
            return None

    @staticmethod
    def _existing_body_path(case_id: str, download_dir: Path) -> Path | None:
        """Return the on-disk body for ``case_id`` if one exists in any accepted ext."""
        for ext in ACCEPTED_BODY_EXTENSIONS:
            candidate = download_dir / f"{case_id}{ext}"
            try:
                if candidate.stat().st_size > 0:
                    return candidate
            except FileNotFoundError:
                continue
            except OSError:
                continue
        return None

    # Abstract on the base class; we satisfy them but never dispatch
    # through this path since :meth:`download` is overridden.
    def _get_output_filename(self, url: str) -> str:
        case_id = doc_id_from_url(url) or "unknown"
        return f"{case_id}.pdf"

    def _download_to_path(  # pragma: no cover - bypassed by download()
        self, url: str, path: str
    ) -> tuple[bool, str | None]:
        raise NotImplementedError(
            "CongbobananDocumentDownloader.download() is overridden; "
            "_download_to_path is never invoked."
        )

    def num_workers_per_node(self) -> int | None:
        """Cap per-node downloader concurrency against congbobanan's WAF."""
        return self._num_workers

    # --------------------------------------------------- internals

    def _ensure_session(self) -> None:
        if self.session is None:
            self.session = session_from_scraper_cfg(self.cfg)
            if self._extra_headers:
                self.session._session.headers.update(self._extra_headers)

    def _fetch_detail_html(self, url: str) -> str:
        assert self.session is not None
        try:
            resp = self.session.get(url)
        except Exception as exc:
            logger.warning("detail fetch failed for %s: %s", url, exc)
            return ""
        if resp.status_code != 200:
            return ""
        return resp.text


__all__ = [
    "ACCEPTED_BODY_EXTENSIONS",
    "CongbobananDocumentDownloader",
    "page_has_metadata",
]
