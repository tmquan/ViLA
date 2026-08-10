"""Anle DocumentExtractor: detail HTML -> structured row fields.

Subclasses :class:`nemo_curator.stages.text.download.base.DocumentExtractor`.
Adds the columns the downstream pipeline needs (``precedent_number``,
``adopted_date``, ``applied_article``, ``principle_text``, ``court``,
``pdf_url``, ``source``) while preserving the iterator's keys. Curator
runs this per iterated record; the returned dict becomes one row of the
produced :class:`DocumentBatch`.

Self-contained: the CSS-selector map that used to live in the Hydra
config is inlined here as :data:`DEFAULT_SELECTORS`; callers may override
any subset via the ``selectors`` constructor arg.
"""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup
from nemo_curator.stages.text.download.base import DocumentExtractor

from packages.datasites.anle.components.url_generator import absolutize

DEFAULT_HOST = "anle.toaan.gov.vn"
_BASE_URL = "https://anle.toaan.gov.vn/"

DEFAULT_SELECTORS: dict[str, list[str]] = {
    "precedent_number": ["h1.al-title", "h1", ".al-header"],
    "adopted_date": [".al-adopted-date", "span.date"],
    "applied_article": [".al-applied-article", ".al-article"],
    "principle_text": [".al-principle", ".al-body", "article"],
    "court": [".al-court", ".court"],
    "pdf_link": ["a[href$='.pdf']", "a[href*='.pdf']"],
}


class AnleExtractor(DocumentExtractor):
    """Parse detail HTML into the anle row shape."""

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        selectors: dict[str, list[str]] | None = None,
    ) -> None:
        self._host = host
        self._selectors = _merge_selectors(DEFAULT_SELECTORS, selectors)

    def extract(self, record: dict[str, Any]) -> dict[str, Any] | None:
        header = self._parse_detail(record.get("detail_html", ""))
        return {
            "doc_name": record["doc_name"],
            "source": self._host,
            "detail_url": record.get("detail_url", ""),
            "pdf_path": record.get("pdf_path", ""),
            "pdf_bytes": record.get("pdf_bytes", b""),
            "precedent_number": header.get("precedent_number"),
            "adopted_date": header.get("adopted_date"),
            "applied_article": header.get("applied_article"),
            "principle_text": header.get("principle_text"),
            "court": header.get("court"),
            "pdf_url": header.get("pdf_url"),
        }

    def input_columns(self) -> list[str]:
        return ["doc_name", "pdf_path", "pdf_bytes", "detail_html", "detail_url"]

    def output_columns(self) -> list[str]:
        return [
            "doc_name",
            "source",
            "detail_url",
            "pdf_path",
            "pdf_bytes",
            "precedent_number",
            "adopted_date",
            "applied_article",
            "principle_text",
            "court",
            "pdf_url",
        ]

    # ------------------------------------------------------ internals
    def _parse_detail(self, html: str) -> dict[str, Any]:
        if not html:
            return {}
        soup = BeautifulSoup(html, "html.parser")
        header: dict[str, Any] = {
            "precedent_number": _first_text(soup, self._selectors["precedent_number"]),
            "adopted_date": _first_text(soup, self._selectors["adopted_date"]),
            "applied_article": _first_text(soup, self._selectors["applied_article"]),
            "principle_text": _first_text(soup, self._selectors["principle_text"]),
            "court": _first_text(soup, self._selectors["court"]),
            "pdf_url": _first_href(soup, self._selectors["pdf_link"]),
        }
        if header["pdf_url"]:
            header["pdf_url"] = absolutize(_BASE_URL, header["pdf_url"])
        return header


def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for s in selectors:
        node = soup.select_one(s)
        if node is not None:
            text = (node.get_text(separator=" ") or "").strip()
            if text:
                return text
    return None


def _first_href(soup: BeautifulSoup, selectors: list[str]) -> str | None:
    for s in selectors:
        node = soup.select_one(s)
        if node is not None and node.get("href"):
            return str(node["href"])
    return None


def _merge_selectors(
    base: dict[str, list[str]], override: dict[str, list[str]] | None
) -> dict[str, list[str]]:
    out = {k: list(v) for k, v in base.items()}
    if override:
        for key, sels in override.items():
            out[key] = list(sels)
    return out


__all__ = ["DEFAULT_SELECTORS", "AnleExtractor"]
