"""luutru DocumentDownloader: fetch detail HTML + PDF binary.

Subclasses :class:`nemo_curator.stages.text.download.base.DocumentDownloader`.
Given a detail-page URL produced by :class:`LuutruURLGenerator`, this
downloader:

1. GETs the detail HTML (``xemchitietvanban.htm?id=<GUID>``) for
   metadata parsing downstream,
2. derives the binary URL from the "Tệp đính kèm" anchor in the HTML
   (the attachment lives on ``dms.luutru.gov.vn``),
3. streams the binary to
   ``<download_dir>/<doc_name>.{pdf,docx,doc}`` via an atomic
   ``.tmp -> final`` rename, and
4. writes sibling ``<doc_name>.html`` + ``<doc_name>.url`` caches the
   iterator reads back on the next stage.

The base class's :meth:`download` is fully overridden so we own the
atomic rename and pick the final extension from the HEAD probe before
any bytes are written.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from nemo_curator.stages.text.download.base import DocumentDownloader

from packages.common.http import PoliteSession
from packages.datasites.luutru.components.url_generator import (
    _session_from_cfg,
    absolutize,
    extract_doc_name_from_url,
)

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://luutru.gov.vn/"

_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
}

#: Extensions we consider a completed download. Used by the idempotent
#: skip check so re-running after a ``.docx`` download does not fetch
#: the same document again as ``.pdf``.
_KNOWN_EXTS: tuple[str, ...] = (".pdf", ".docx", ".doc")

#: Attachment-anchor selectors, tried in order. The detail page links
#: the source file on the ``dms.luutru.gov.vn`` document store.
_PDF_LINK_SELECTORS: tuple[str, ...] = (
    "a[href*='dms.luutru.gov.vn']",
    "a[href$='.pdf']",
    "a[href*='.pdf']",
    "a[href$='.docx']",
    "a[href$='.doc']",
)


class LuutruDocumentDownloader(DocumentDownloader):
    """Detail HTML + binary attachment fetcher for one luutru document.

    Holds only pickle-safe state on the driver (``cfg`` + knobs). The
    :class:`PoliteSession` is constructed lazily inside :meth:`download`
    (per Ray worker) because it owns a :class:`threading.Lock` that
    cannot be serialised across workers.
    """

    def __init__(
        self,
        cfg: Any,
        download_dir: str,
        *,
        verbose: bool = False,
    ) -> None:
        super().__init__(download_dir=download_dir, verbose=verbose)
        self.cfg = cfg
        self._base_url: str = str(cfg.scraper.get("base_url", DEFAULT_BASE_URL))
        self._fetch_detail: bool = bool(cfg.scraper.get("fetch_detail_page", True))
        self._fetch_head: bool = bool(
            cfg.scraper.get("fetch_head_before_download", True)
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
        """Fetch one luutru document. Returns the final on-disk path."""
        doc_name = extract_doc_name_from_url(url)
        if not doc_name:
            logger.error("could not derive doc_name from url %s", url)
            return None

        # Idempotent skip: if any known-extension binary already exists
        # and is non-empty, treat the document as already downloaded.
        for ext in _KNOWN_EXTS:
            existing = Path(self._download_dir) / f"{doc_name}{ext}"
            if existing.exists() and existing.stat().st_size > 0:
                if self._verbose:
                    logger.info("file %s exists; not downloading", existing)
                return str(existing)

        self._ensure_session()
        assert self.session is not None

        try:
            detail_html = self._fetch_detail_html(url) if self._fetch_detail else ""
            pdf_url = self._resolve_pdf_url(detail_html)
            if not pdf_url:
                logger.warning("no attachment link found on %s; skipping", url)
                return None
            ext, expected_mime = self._pick_extension(pdf_url)

            final_path = Path(self._download_dir) / f"{doc_name}{ext}"
            tmp_path = str(final_path) + ".tmp"

            self.session.download(pdf_url, tmp_path, expected_mime=expected_mime)
            os.replace(tmp_path, final_path)

            # Sidecars keyed on the final stem so the iterator can
            # recover detail HTML / source URL without a second HTTP.
            if detail_html:
                final_path.with_suffix(".html").write_text(
                    detail_html, encoding="utf-8"
                )
            final_path.with_suffix(".url").write_text(url, encoding="utf-8")

            if self._verbose:
                logger.info("downloaded %s to %s", url, final_path)
            return str(final_path)

        except Exception as exc:  # log + skip a single doc
            logger.error("download failed for %s: %s", url, exc)
            return None

    # ``_get_output_filename`` / ``_download_to_path`` are abstract on
    # the base class; we implement them but they are never called
    # because we override :meth:`download` above.
    def _get_output_filename(self, url: str) -> str:
        doc_name = extract_doc_name_from_url(url) or "unknown"
        return f"{doc_name}.pdf"

    def _download_to_path(  # pragma: no cover - bypassed by download()
        self, url: str, path: str
    ) -> tuple[bool, str | None]:
        raise NotImplementedError(
            "LuutruDocumentDownloader.download() is overridden; "
            "_download_to_path is never invoked."
        )

    def num_workers_per_node(self) -> int | None:
        """Rate-limit HTTP fan-out against the luutru.gov.vn host."""
        return self._num_workers

    # --------------------------------------------------- internals

    def _ensure_session(self) -> None:
        if self.session is None:
            self.session = _session_from_cfg(self.cfg)
            if self._extra_headers:
                self.session._session.headers.update(self._extra_headers)

    def _fetch_detail_html(self, url: str) -> str:
        assert self.session is not None
        resp = self.session.get(url)
        resp.raise_for_status()
        return resp.text

    def _resolve_pdf_url(self, detail_html: str) -> str | None:
        if not detail_html:
            return None
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(detail_html, "html.parser")
        for selector in _PDF_LINK_SELECTORS:
            anchor = soup.select_one(selector)
            if anchor and anchor.get("href"):
                return absolutize(self._base_url, str(anchor["href"]))
        return None

    def _pick_extension(self, url: str) -> tuple[str, str]:
        if not self._fetch_head:
            return ".pdf", "application/pdf"
        assert self.session is not None
        try:
            head = self.session._session.head(
                url,
                timeout=self.session._timeout,
                allow_redirects=True,
                verify=self.session._session.verify,
            )
            content_type = head.headers.get("Content-Type", "").split(";")[0].strip()
        except Exception:  # default to PDF on a flaky HEAD
            content_type = "application/pdf"
        ext = _MIME_TO_EXT.get(content_type)
        if ext is None:
            # Fall back to the URL suffix, then PDF.
            lower = url.lower().split("?", 1)[0]
            for known in _KNOWN_EXTS:
                if lower.endswith(known):
                    ext = known
                    break
            else:
                ext = ".pdf"
        expected = (
            content_type if content_type in _MIME_TO_EXT else "application/pdf"
        )
        return ext, expected


__all__ = ["DEFAULT_BASE_URL", "LuutruDocumentDownloader"]
