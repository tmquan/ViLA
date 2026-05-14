"""Pure-function parsers for vbpl.

Two surfaces, both safely processable without any network or browser:

* Sitemap XML (the public ``/sitemap.xml`` index + every
  ``sitemap-trung-uong-N.xml`` / ``sitemap-dia-phuong-N.xml`` shard).
  We parse the index into a list of shard URLs and each urlset into a
  list of :class:`SitemapEntry`.
* Captured API JSON (one or more responses to
  ``/api/qtdc/public/doc/...`` collected by
  :class:`packages.datasites.vbpl.components.detail.VbplDetailDownloader`
  via Playwright network interception). The shape is best-effort
  documented; unknown keys are tolerated.

Both parsers are intentionally I/O-free and Playwright-free so they
are unit-testable against fixtures and re-runnable against a stored
``html/<scope>/<id>.html`` + sibling JSON dump.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)


# ---- sitemap -------------------------------------------------------------


_SM_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"

#: ``/van-ban/chi-tiet/<slug>--<id>``. The slug is lowercase kebab-case
#: Vietnamese transliteration; the trailing ``--<id>`` is vbpl's stable
#: primary key and comes in three observed flavours:
#:
#: * ``--186739``                pure-digit current-id (post-2026 portal)
#: * ``--vbpqta_6629``           legacy "Văn Bản Pháp Quy Toàn Văn" id
#: * ``--vbpqdinhchinh_88``      legacy "VBPQ Đính Chính" (corrigendum) id
#:
#: We accept any ``[A-Za-z0-9_]+`` after the final ``--`` so all three
#: flavours round-trip. The greedy slug match works because vbpl slugs
#: themselves only contain single ``-`` separators; the ``--`` only
#: appears as the slug<->id boundary.
_DETAIL_URL_RE = re.compile(
    r"/van-ban/chi-tiet/(?P<slug>.+)--(?P<id>[A-Za-z0-9_]+)$"
)

#: A path segment of the sitemap-shard URL classifies it into a scope.
#: ``sitemap-trung-uong-1.xml`` -> ``trung_uong``,
#: ``sitemap-dia-phuong-21.xml`` -> ``dia_phuong``.
_SHARD_SCOPE_RE = re.compile(r"sitemap-(trung-uong|dia-phuong)-\d+\.xml$")


@dataclass
class SitemapEntry:
    """One ``<url>`` row in a sitemap shard.

    ``item_id`` is a string because vbpl mixes pure-digit modern IDs
    (``186739``) with legacy alphanumeric IDs (``vbpqta_6629``,
    ``vbpqdinhchinh_88``). String form round-trips both shapes through
    JSON, filesystem paths, and URL templates without coercion.
    """

    item_id: str
    scope: str
    slug: str
    url: str
    lastmod: str | None = None
    changefreq: str | None = None
    priority: str | None = None


def parse_sitemap_index(xml_text: str) -> list[str]:
    """Return the list of shard URLs from a ``<sitemapindex>`` document.

    The order is preserved (vbpl serves shards roughly chronologically;
    keeping order makes the harvest log easy to scan).
    """
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("sitemap index parse failed: %s", exc)
        return []
    out: list[str] = []
    for sm in root.findall(f"{_SM_NS}sitemap"):
        loc = sm.findtext(f"{_SM_NS}loc")
        if loc:
            out.append(loc.strip())
    return out


def parse_sitemap_urlset(xml_text: str, *, scope: str) -> list[SitemapEntry]:
    """Parse one ``<urlset>`` shard into :class:`SitemapEntry` rows.

    Rows that do not match the ``/van-ban/chi-tiet/<slug>--<id>``
    pattern (e.g. legacy index pages bundled into ``sitemap-static``)
    are silently dropped; they contain no document content.
    """
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning("sitemap urlset parse failed: %s", exc)
        return []

    out: list[SitemapEntry] = []
    for u in root.findall(f"{_SM_NS}url"):
        loc = (u.findtext(f"{_SM_NS}loc") or "").strip()
        if not loc:
            continue
        info = item_id_from_detail_url(loc)
        if info is None:
            continue
        slug, item_id = info
        out.append(
            SitemapEntry(
                item_id=item_id,
                scope=scope,
                slug=slug,
                url=loc,
                lastmod=_text_or_none(u.findtext(f"{_SM_NS}lastmod")),
                changefreq=_text_or_none(u.findtext(f"{_SM_NS}changefreq")),
                priority=_text_or_none(u.findtext(f"{_SM_NS}priority")),
            ),
        )
    return out


def item_id_from_detail_url(url: str) -> tuple[str, str] | None:
    """Return ``(slug, item_id)`` parsed from a ``/van-ban/chi-tiet/<slug>--<id>`` URL.

    Returns ``None`` for URLs that don't match the detail shape (e.g.
    listing or static pages mixed into ``sitemap-static.xml``).
    Both elements are strings; see :class:`SitemapEntry` for why
    ``item_id`` is not coerced to int.
    """
    if not url:
        return None
    m = _DETAIL_URL_RE.search(url.strip())
    if m is None:
        return None
    return m.group("slug"), m.group("id")


def scope_from_shard_url(url: str) -> str | None:
    """Return ``"trung_uong"`` / ``"dia_phuong"`` for a shard URL.

    Returns ``None`` for unrecognised shards (e.g.
    ``sitemap-static.xml``); the caller can choose to skip them.
    """
    if not url:
        return None
    m = _SHARD_SCOPE_RE.search(url)
    if m is None:
        return None
    return m.group(1).replace("-", "_")


# ---- detail-API JSON -----------------------------------------------------


@dataclass
class FilePath:
    """One downloadable attachment exposed by the detail API."""

    file_url: str
    file_name: str | None = None
    file_type: str | None = None    # extension, lowercased: pdf | doc | docx ...
    local_path: str | None = None   # set by the downloader after success


@dataclass
class DetailRecord:
    """Mapped vbpl document detail.

    ``item_id`` is a string for the same reason as in
    :class:`SitemapEntry` (mixed pure-digit + legacy alphanumeric IDs).
    ``raw_api_json`` keeps the union of every captured API response so
    a future pipeline can re-extract fields we don't pull out yet
    without re-running the slow Playwright fetch.
    """

    item_id: str
    scope: str
    source_url: str
    api_url: str | None = None
    title: str = ""
    so_hieu: str | None = None
    doc_type: str | None = None
    ngay_ban_hanh: str | None = None
    co_quan_ban_hanh: str | None = None
    trich_yeu: str | None = None
    body_html: str = ""
    body_text: str = ""
    file_paths: list[FilePath] = field(default_factory=list)
    raw_api_json: dict[str, Any] = field(default_factory=dict)


# Candidate JSON keys we probe in priority order. vbpl's API is
# Spring Boot + Vietnamese-domain field names; the actual schema
# isn't published, so we try the obvious Vietnamese-snake names first
# and fall back to common English-camelCase. The first non-empty
# value wins.
_TITLE_KEYS = ("tieuDe", "tenVanBan", "title", "name")
_SO_HIEU_KEYS = ("soHieu", "soHieuVanBan", "documentNumber", "code")
_DOC_TYPE_KEYS = (
    "loaiVanBanText", "loaiVanBan", "docType", "type", "tenLoaiVanBan",
)
_NGAY_KEYS = (
    "ngayBanHanh", "ngayKy", "ngayHieuLuc", "issuedDate", "signedDate",
)
_AGENCY_KEYS = (
    "coQuanBanHanh", "tenCoQuanBanHanh", "noiBanHanh",
    "issuingAgency", "agency",
)
_TRICH_KEYS = ("trichYeu", "summary", "abstract")
_BODY_HTML_KEYS = (
    "noiDung", "toanVan", "noiDungVanBan", "body", "content",
    "htmlContent", "bodyHtml",
)
_FILES_KEYS = (
    "tepDinhKem", "fileDinhKem", "danhSachFile", "files",
    "attachments", "tepTin",
)
_FILE_URL_KEYS = ("filePath", "url", "downloadUrl", "fileUrl", "path")
_FILE_NAME_KEYS = ("tenFile", "fileName", "name")


def detail_record_from_api_json(
    *,
    item_id: str,
    scope: str,
    source_url: str,
    api_responses: Iterable[tuple[str, Any]],
) -> DetailRecord:
    """Fold the captured API responses into a :class:`DetailRecord`.

    ``api_responses`` is an iterable of ``(api_url, parsed_json)``
    pairs. The body / metadata response is typically a single dict
    keyed by the document; ``related-file`` and similar endpoints
    return list payloads with one row per attachment. We merge them
    all into one record, taking the first non-empty value for each
    field across responses.
    """
    rec = DetailRecord(
        item_id=item_id,
        scope=scope,
        source_url=source_url,
    )
    raw: dict[str, Any] = {}
    for api_url, payload in api_responses:
        if payload is None:
            continue
        # Index by URL substring so multiple responses don't clobber
        # each other (e.g. /preview-by-target/ vs /related-file).
        raw[api_url] = payload

        # Recursively scan dicts for our candidate keys; lists of
        # files get flattened in a separate pass.
        for d in _walk_dicts(payload):
            if not rec.title:
                rec.title = _str_or_empty(_first_nonempty(d, _TITLE_KEYS))
            if rec.so_hieu is None:
                rec.so_hieu = _none_or_str(_first_nonempty(d, _SO_HIEU_KEYS))
            if rec.doc_type is None:
                rec.doc_type = _none_or_str(_first_nonempty(d, _DOC_TYPE_KEYS))
            if rec.ngay_ban_hanh is None:
                rec.ngay_ban_hanh = _iso_date(_first_nonempty(d, _NGAY_KEYS))
            if rec.co_quan_ban_hanh is None:
                rec.co_quan_ban_hanh = _none_or_str(
                    _first_nonempty(d, _AGENCY_KEYS),
                )
            if rec.trich_yeu is None:
                rec.trich_yeu = _none_or_str(_first_nonempty(d, _TRICH_KEYS))
            if not rec.body_html:
                rec.body_html = _str_or_empty(
                    _first_nonempty(d, _BODY_HTML_KEYS),
                )

        rec.file_paths.extend(_extract_files(payload))

    rec.body_text = _html_to_text(rec.body_html)
    rec.raw_api_json = raw
    if rec.api_url is None and raw:
        rec.api_url = next(iter(raw.keys()), None)
    return rec


# ---- internals -----------------------------------------------------------


def _text_or_none(v: str | None) -> str | None:
    if v is None:
        return None
    s = v.strip()
    return s or None


def _str_or_empty(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _none_or_str(v: Any) -> str | None:
    s = _str_or_empty(v)
    return s or None


def _first_nonempty(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d:
            v = d[k]
            if v is None:
                continue
            if isinstance(v, str) and not v.strip():
                continue
            return v
    return None


def _walk_dicts(payload: Any) -> Iterable[dict[str, Any]]:
    """Yield every dict inside a possibly-nested JSON payload."""
    if isinstance(payload, dict):
        yield payload
        for v in payload.values():
            yield from _walk_dicts(v)
    elif isinstance(payload, list):
        for v in payload:
            yield from _walk_dicts(v)


def _extract_files(payload: Any) -> list[FilePath]:
    """Pull every attachment-like row out of a payload.

    Looks for any dict (at any nesting level) that matches one of the
    file-list keys, then flattens its rows into :class:`FilePath`. A
    row qualifies if it has any URL-shaped value under the candidate
    file-URL keys.
    """
    out: list[FilePath] = []
    seen: set[str] = set()
    for d in _walk_dicts(payload):
        for fk in _FILES_KEYS:
            v = d.get(fk)
            if not isinstance(v, list):
                continue
            for row in v:
                if not isinstance(row, dict):
                    continue
                url_val = _first_nonempty(row, _FILE_URL_KEYS)
                if not url_val:
                    continue
                url = str(url_val)
                if url in seen:
                    continue
                seen.add(url)
                name = _none_or_str(_first_nonempty(row, _FILE_NAME_KEYS))
                ext = _ext_from(name) or _ext_from(url)
                out.append(FilePath(file_url=url, file_name=name, file_type=ext))
    return out


_EXT_RE = re.compile(r"\.([A-Za-z0-9]{1,5})(?:[?#].*)?$")


def _ext_from(s: str | None) -> str | None:
    if not s:
        return None
    m = _EXT_RE.search(s)
    return m.group(1).lower() if m else None


_DATE_DDMMYYYY_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")
_DATE_ISO_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})")


def _iso_date(raw: Any) -> str | None:
    """Coerce common Vietnamese / ISO date encodings to ``YYYY-MM-DD``."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    m = _DATE_ISO_RE.match(s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = _DATE_DDMMYYYY_RE.search(s)
    if m:
        d, mo, y = (int(m.group(i)) for i in (1, 2, 3))
        try:
            return datetime(y, mo, d).date().isoformat()
        except ValueError:
            return None
    return None


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _html_to_text(html: str) -> str:
    """Cheap HTML -> text. Good enough for a JSONL preview column.

    bs4 would be cleaner but parser.py is intentionally dep-light
    (the rest of the parser is re/dataclass only) so the test suite
    can run without installing the full crawler stack. The downstream
    extractor can re-render from ``body_html`` for fidelity.
    """
    if not html:
        return ""
    s = html.replace("\xa0", " ").replace("\u00a0", " ")
    s = _TAG_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


__all__ = [
    "DetailRecord",
    "FilePath",
    "SitemapEntry",
    "detail_record_from_api_json",
    "item_id_from_detail_url",
    "parse_sitemap_index",
    "parse_sitemap_urlset",
    "scope_from_shard_url",
]
