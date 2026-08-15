"""congbobanan PDF/binary downloader (self-contained curl_cffi).

Given a detail-page URL produced by :class:`CBBADocumentURLGenerator`,
this downloader:

1. GETs the detail HTML (best-effort: empty sidebars are tolerated) and
   stores it gzipped to ``pages/<case_id>.html.gz``.
2. Attempts the body download at ``/3ta{case_id}t1cvn/`` regardless of
   the detail panel's contents. The body endpoint serves real documents
   for ~30% of IDs whose detail panel is empty, so gating the fetch on
   :func:`page_has_metadata` (as the reference scraper did) wrongly
   discards them. Validation runs *after* the download via
   :func:`_sniff_body_ext`, which classifies the response by magic
   header (``%PDF-`` / ``PK\\x03\\x04`` / OLE2 / ``{\\rtf``) and filters
   out the ~6475-byte HTML 500-error pages the server returns for
   unrecoverable IDs. The server occasionally serves a judgment as
   DOCX / DOC / RTF rather than PDF -- all four formats are accepted and
   the body is saved with the sniffed extension to
   ``files/<case_id>.<ext>``.

congbobanan.toaan.gov.vn refuses TLS handshakes from non-Vietnamese
source IPs and ships a ``.gov.vn`` cert chain not in the Mozilla bundle,
so the session is built with ``make_session(verify=False)`` and an
optional VN-egress proxy. The base :class:`PDFDownloader.download` is
overridden here because the server lies about ``Content-Type`` (claims
``application/pdf`` even for DOCX bodies), so the base's HEAD-probe
extension selection would misfile documents -- magic-byte sniffing is
authoritative instead.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from loguru import logger

from packages.datasites._curator.base import (
    PDFDownloader,
    is_challenge,
    make_session,
)
from packages.datasites.congbobanan.components.url_generator import (
    DEFAULT_PDF_URL_TEMPLATE,
    doc_id_from_url,
)


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


#: Minimum byte length accepted as a valid PDF body. The 500-error pages
#: congbobanan serves on broken IDs are ~6475 bytes of HTML, so a size
#: gate alone is insufficient -- :func:`_sniff_body_ext` also requires
#: the canonical ``%PDF`` magic header.
_MIN_VALID_PDF_BYTES = 1_024

#: Minimum byte length per accepted body extension. RTF can be quite
#: small; PDF/DOC/DOCX bodies always carry kilobytes of header/metadata.
_MIN_VALID_BODY_BYTES: dict[str, int] = {
    ".pdf": _MIN_VALID_PDF_BYTES,
    ".docx": 1_024,
    ".doc": 1_024,
    ".rtf": 100,
}

#: ``(magic header bytes, target extension)``. Longer / more specific
#: magics first so e.g. OLE2 wins over an accidental ``PK`` prefix. These
#: are the four body formats the congbobanan portal is known to serve;
#: everything else (HTML error pages, JSON stubs, RAR / PE32 garbage
#: observed in the corpus) is rejected by :func:`_sniff_body_ext`.
_BODY_MAGICS: tuple[tuple[bytes, str], ...] = (
    (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", ".doc"),   # OLE2 (legacy .doc)
    (b"PK\x03\x04", ".docx"),                        # Office Open XML
    (b"%PDF-", ".pdf"),
    (b"{\\rtf", ".rtf"),
)

#: Filename extensions the downloader will write and the iterator reads.
ACCEPTED_BODY_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".doc", ".rtf")


def _sniff_body_ext(path: str) -> str | None:
    """Return ``.pdf`` / ``.docx`` / ``.doc`` / ``.rtf`` for a valid body, else None.

    1. Match the leading bytes against :data:`_BODY_MAGICS`.
    2. Enforce the per-extension minimum size from
       :data:`_MIN_VALID_BODY_BYTES` so ~6475-byte HTML error pages and
       one-line stubs are rejected even when the magic header aligns.
    3. For PDFs only, additionally require ``%%EOF`` somewhere in the
       last 1KB -- guards against truncated bodies (connection drops
       mid-stream) that are valid PDFs by header but unusable by the
       parser.
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
    """Return True iff ``path`` validates specifically as a PDF body."""
    return _sniff_body_ext(path) == ".pdf"


class CBBADocumentPDFDownloader(PDFDownloader):
    """Ghost-aware downloader for congbobanan detail + body endpoints.

    Subclasses :class:`packages.datasites._curator.base.PDFDownloader`:
    detail HTML -> ``pages/<case_id>.html.gz``, binary body ->
    ``files/<case_id>.<ext>``. ``download()`` is overridden to replace
    the base MIME-based extension selection with post-download magic-byte
    sniffing (the server lies about ``Content-Type``).
    """

    def __init__(
        self,
        download_dir: str,
        *,
        pages_dir: str | None = None,
        proxy: str | None = None,
        user_agent: str | None = None,
        pdf_url_template: str = DEFAULT_PDF_URL_TEMPLATE,
        retry_empty_detail: bool = True,
        verbose: bool = False,
        timeout: int = 60,
        max_retries: int = 2,
        cooldown: float = 6.0,
        pace: float = 0.5,
        num_workers: int | None = 4,
    ) -> None:
        super().__init__(
            download_dir,
            pages_dir=pages_dir,
            verbose=verbose,
            timeout=timeout,
            max_retries=max_retries,
            cooldown=cooldown,
            pace=pace,
            num_workers=num_workers,
        )
        self.proxy = proxy
        self.user_agent = user_agent
        self.pdf_url_template = pdf_url_template or DEFAULT_PDF_URL_TEMPLATE
        self.retry_empty_detail = retry_empty_detail

    # -- subclass hooks ---------------------------------------------------- #
    def _new_session(self):
        # verify=False: the .gov.vn cert chain is not in the Mozilla bundle.
        return make_session(ua=self.user_agent, proxy=self.proxy, verify=False)

    def _doc_name(self, url: str) -> str | None:
        return doc_id_from_url(url)

    def _resolve_binary_url(self, url: str, detail_html: str, doc_name: str) -> str | None:
        return self.pdf_url_template.format(case_id=doc_name)

    # -- helpers ----------------------------------------------------------- #
    def _existing_body_path(self, case_id: str) -> Path | None:
        download_dir = Path(self._download_dir)
        for ext in ACCEPTED_BODY_EXTENSIONS:
            candidate = download_dir / f"{case_id}{ext}"
            try:
                if candidate.stat().st_size > 0:
                    return candidate
            except OSError:
                continue
        return None

    def _fetch_detail_html(self, url: str) -> str:
        # Shared resilience policy (throttle + Cloudflare challenge + transient
        # network faults) lives in PDFDownloader._get_with_retry; text response,
        # so challenge-probing stays on.
        resp = self._get_with_retry(self._session(), url, check_challenge=True)
        if resp is not None and resp.status_code == 200 and not is_challenge(resp.text):
            return resp.text
        return ""

    # -- Curator contract -------------------------------------------------- #
    def download(self, url: str) -> str | None:
        """Fetch one congbobanan case. Returns the final on-disk path or None.

        Two-step fetch: the detail HTML at ``/2ta<id>t1cvn/chi-tiet-ban-an``
        (saved to ``pages/``) and the body at ``/3ta<id>t1cvn/`` (saved to
        ``files/``). The detail panel is best-effort -- some IDs serve a
        real body even though the sidebar is an empty ``details_pub`` div,
        so we never gate the body fetch on it. A body is accepted only
        when :func:`_sniff_body_ext` classifies it as one of the four
        supported formats.
        """
        case_id = self._doc_name(url)
        if not case_id:
            logger.error(f"could not derive case_id from url {url}")
            return None

        existing = self._existing_body_path(case_id)
        if existing is not None:
            if self._verbose:
                logger.info(f"file {existing} exists; not downloading")
            return str(existing)

        download_dir = Path(self._download_dir)
        s = self._session()
        try:
            # Best-effort detail HTML; retry once on an empty panel because
            # the WAF occasionally returns a stub on the first hit even for
            # valid records.
            detail_html = self._fetch_detail_html(url)
            if not page_has_metadata(detail_html) and self.retry_empty_detail:
                detail_html = self._fetch_detail_html(url)
            self._save_html(case_id, detail_html or "")

            body_url = self._resolve_binary_url(url, detail_html, case_id)
            if not body_url:
                logger.warning(f"no binary url for {url}")
                return None

            tmp_path = str(download_dir / f"{case_id}.body.tmp")
            got = False
            # Binary response: same shared retry policy, challenge-probing OFF
            # (raw bytes must not be decoded as text to sniff a challenge page).
            r = self._get_with_retry(s, body_url, check_challenge=False)
            if r is not None and r.status_code == 200 and r.content:
                with open(tmp_path, "wb") as f:
                    f.write(r.content)
                got = True

            if not got:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                logger.info(f"case {case_id}: body unavailable; skipping")
                return None

            ext = _sniff_body_ext(tmp_path)
            if ext is None:
                # 200-with-HTML error page, truncated PDFs, or exotic
                # content (JSON / RAR / PE32 garbage) land here.
                os.unlink(tmp_path)
                logger.info(
                    f"case {case_id}: body not a recognised document format; skipping"
                )
                return None

            final_path = download_dir / f"{case_id}{ext}"
            os.replace(tmp_path, final_path)
            time.sleep(self.pace)

            if not page_has_metadata(detail_html):
                logger.info(
                    f"case {case_id}: downloaded {ext} without sidebar metadata "
                    "(detail panel empty)"
                )
            elif ext != ".pdf":
                logger.info(f"case {case_id}: downloaded as {ext} (non-PDF body)")
            elif self._verbose:
                logger.info(f"downloaded {url} to {final_path}")
            return str(final_path)

        except Exception as exc:  # noqa: BLE001
            logger.error(f"download failed for {url}: {exc}")
            return None


__all__ = [
    "ACCEPTED_BODY_EXTENSIONS",
    "CBBADocumentPDFDownloader",
    "page_has_metadata",
]
