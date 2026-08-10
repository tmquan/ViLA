"""Anle PDF downloader: detail HTML -> pages/, binary -> files/.

Subclasses the shared :class:`packages.datasites._curator.base.PDFDownloader`,
which already implements the resumable/atomic ``download(url)`` flow:

    1. GET the detail HTML          -> pages/<doc>.html.gz
    2. resolve the binary URL       (``_resolve_binary_url``)
    3. HEAD-probe the MIME          -> .pdf / .docx / .doc
    4. stream the binary            -> files/<doc>.<ext>

This subclass only supplies the anle specifics: a curl_cffi Chrome-JA3
session with ``verify=False`` (anle's cert is signed by a VN CA outside
the Mozilla bundle) and the ``Accept: */*`` header Oracle ADF needs, the
``dDocName`` -> doc-name mapping, and the ShowProperty binary-URL
resolution (anchor in the detail HTML, else the UCMServer template).
"""

from __future__ import annotations

from packages.datasites._curator.base import PDFDownloader, make_session
from packages.datasites.anle.components.url_generator import (
    ADF_HEADERS,
    absolutize,
    extract_doc_name,
)

DEFAULT_PDF_URL_TEMPLATE = (
    "https://anle.toaan.gov.vn/webcenter/ShowProperty"
    "?nodeId=/UCMServer/{doc_name}"
)
_BASE_URL = "https://anle.toaan.gov.vn/"


class AnlePDFDownloader(PDFDownloader):
    """Detail HTML + binary attachment fetcher for one anle document."""

    def __init__(
        self,
        download_dir: str,
        *,
        pages_dir: str | None = None,
        pdf_url_template: str = DEFAULT_PDF_URL_TEMPLATE,
        proxy: str | None = None,
        fetch_detail: bool = True,
        verbose: bool = False,
        timeout: int = 60,
        max_retries: int = 2,
        cooldown: float = 30.0,
        pace: float = 0.5,
        num_workers: int | None = 4,
    ) -> None:
        super().__init__(
            download_dir,
            pages_dir=pages_dir,
            verbose=verbose,
            fetch_detail=fetch_detail,
            timeout=timeout,
            max_retries=max_retries,
            cooldown=cooldown,
            pace=pace,
            num_workers=num_workers,
        )
        self._pdf_url_template = pdf_url_template
        self._proxy = proxy

    # -- subclass hooks ---------------------------------------------------- #
    def _new_session(self):
        s = make_session(proxy=self._proxy, verify=False)
        s.headers.update(ADF_HEADERS)
        return s

    def _doc_name(self, url: str) -> str | None:
        return extract_doc_name(url)

    def _resolve_binary_url(self, url: str, detail_html: str, doc_name: str) -> str | None:
        if detail_html:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(detail_html, "html.parser")
            anchor = soup.select_one("a[href$='.pdf'], a[href*='.pdf']")
            if anchor and anchor.get("href"):
                return absolutize(_BASE_URL, str(anchor["href"]))
        return self._pdf_url_template.format(doc_name=doc_name)


__all__ = ["DEFAULT_PDF_URL_TEMPLATE", "AnlePDFDownloader"]
