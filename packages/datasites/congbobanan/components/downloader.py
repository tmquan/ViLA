"""congbobanan DocumentDownloader.

Given a detail-page URL produced by :class:`CongbobananURLGenerator`,
this downloader:

1. GETs the detail HTML.
2. Runs :func:`page_has_metadata` to filter out ghost records (IDs that
   return HTTP 200 with an empty placeholder page). Ghost pages short-
   circuit: no PDF is fetched and :meth:`download` returns ``None`` so
   Curator's :class:`DocumentDownloadStage` skips the row.
3. Streams the binary PDF from ``/3ta{case_id}t1cvn/`` to
   ``<download_dir>/<case_id>.pdf``.
4. Writes sibling ``<case_id>.html`` + ``<case_id>.url`` sidecars that
   the iterator reads back on the next stage.

The base class's :meth:`download` is fully overridden so we own the
atomic ``.tmp -> final`` rename and can cancel the download before any
bytes are written when the ghost-page check fails. This mirrors the
anle downloader's approach for the same reason (and for the same
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

        self._pdf_url_template: str = str(
            cfg.scraper.get("pdf_url_template", DEFAULT_PDF_URL_TEMPLATE)
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
        """Fetch one congbobanan case. Returns the final on-disk path or None."""
        case_id = doc_id_from_url(url)
        if not case_id:
            logger.error("could not derive case_id from url %s", url)
            return None

        final_path = Path(self._download_dir) / f"{case_id}.pdf"
        if final_path.exists() and final_path.stat().st_size > 0:
            if self._verbose:
                logger.info("file %s exists; not downloading", final_path)
            return str(final_path)

        self._ensure_session()
        assert self.session is not None

        try:
            detail_html = self._fetch_detail_html(url)
            if not page_has_metadata(detail_html) and self._retry_empty_detail:
                # Ghost pages sometimes fill in on a second request
                # (WAF / cache warm-up). One extra attempt is cheap.
                detail_html = self._fetch_detail_html(url)

            if not page_has_metadata(detail_html):
                logger.debug("case %s: ghost page; skipping", case_id)
                return None

            pdf_url = self._pdf_url_template.format(case_id=case_id)
            tmp_path = str(final_path) + ".tmp"
            self.session.download(
                pdf_url, tmp_path, expected_mime="application/pdf"
            )
            if os.path.getsize(tmp_path) < 100:
                # Sub-100-byte response is almost always an HTML error
                # body the MIME check missed; drop it.
                os.unlink(tmp_path)
                logger.warning("case %s: PDF payload < 100 bytes; skipping", case_id)
                return None
            os.replace(tmp_path, final_path)

            final_path.with_suffix(".html").write_text(
                detail_html, encoding="utf-8"
            )
            final_path.with_suffix(".url").write_text(url, encoding="utf-8")

            if self._verbose:
                logger.info("downloaded %s to %s", url, final_path)
            return str(final_path)

        except Exception as exc:  # noqa: BLE001 - Curator boundary
            logger.error("download failed for %s: %s", url, exc)
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
        except Exception as exc:  # noqa: BLE001 - tolerate transient fails
            logger.warning("detail fetch failed for %s: %s", url, exc)
            return ""
        if resp.status_code != 200:
            return ""
        return resp.text


__all__ = ["CongbobananDocumentDownloader", "page_has_metadata"]
