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
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

from packages.datasites.vbpl.codes import (
    CANONICAL_CODE_TO_NAME,
    code_from_slug,
    doc_type_slug,
    legal_area_label,
    legal_type_name,
)

# Pure text-cleanup helpers now live in ``normalizers``; re-exported so
# external importers (e.g. ``packages.datasites.vbpl.normalizers``) and this
# module's ``__all__`` keep resolving them from ``...components.parser``.
from packages.datasites.vbpl.components.normalizers import (  # noqa: F401
    clean_title,
    normalise_doc_number,
    normalise_doc_number_list,
    normalise_issuing_authority,
    normalise_label,
    normalise_text,
    normalise_title,
    strip_doctype_docnum_crossrefs,
    strip_markdown_junk,
    strip_redundant_title_prefix,
)

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

    ``doc_type`` is the **canonical short code** (e.g. ``"QĐ"``,
    ``"NĐ"``, ``"CT"`` -- not the legacy ``"CThi"``). ``legal_type``
    is the canonical Vietnamese full name (e.g. ``"Chỉ thị"``).
    ``legal_area`` is the first non-empty Vietnamese area label
    pulled from ``documentFields[]``, defaulting to
    ``"Chưa phân loại"`` when the doc isn't tagged. See
    :mod:`packages.datasites.vbpl.codes`.
    """

    item_id: str
    scope: str
    source_url: str
    api_url: str | None = None
    title: str | None = ""
    #: Document number(s). Now a *list* because a small minority of
    #: vbpl rows pack multiple identifiers into one cell separated by
    #: ``" và "`` ("and") or ``,``. Single-value docs (99%+) ship
    #: with a 1-element list; an empty list (mapped to ``null`` in
    #: parquet) means "no number on source".
    doc_number: list[str] = field(default_factory=list)
    doc_type: str | None = None
    legal_type: str | None = None
    legal_area: str | None = None
    issue_date: str | None = None
    issuing_authority: str | None = None
    summary: str | None = None
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
_DOC_NUMBER_KEYS = ("soHieu", "soHieuVanBan", "documentNumber", "docNum")
# Document-type discovery. We prefer the structured ``docType`` block
# (``{name, code, parentCode, ...}``) because it carries both the
# Vietnamese full name AND the short code; the older Spring-Boot
# string-only fields are accepted as fallbacks.
_DOC_TYPE_KEYS = (
    "docType", "loaiVanBanText", "loaiVanBan", "type", "tenLoaiVanBan",
)
_ISSUE_DATE_KEYS = (
    "issueDate", "ngayBanHanh", "ngayKy", "ngayHieuLuc",
    "issuedDate", "signedDate",
)
_AGENCY_KEYS = (
    "agencyName", "coQuanBanHanh", "tenCoQuanBanHanh", "noiBanHanh",
    "issuingAgency", "agency",
)
_SUMMARY_KEYS = ("trichYeu", "summary", "abstract")
_BODY_HTML_KEYS = (
    "content", "noiDung", "toanVan", "noiDungVanBan", "body",
    "htmlContent", "bodyHtml",
)
_LEGAL_AREA_KEYS = ("documentFields", "linhVuc", "domain", "category")
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
        #
        # Field-level **presentation normalization** is deliberately
        # NOT done here -- it moved to the declarative normalizer
        # chain that runs inside the Curator extract stage (wiki/DATASITES.md
        # §3.5 + ``cfg.extractor.normalizers``). The detail stage's
        # only job is to extract raw values from the API JSON and
        # coerce types where the dataclass needs them (date parsing,
        # slug → canonical-code lookup, doc_number split-on-CSV). The
        # normalizer chain then NFC-canonicalises, strips smart
        # quotes, peels redundant title prefixes, etc. -- once,
        # idempotently, recorded in the manifest.
        for d in _walk_dicts(payload):
            if not rec.title:
                rec.title = _none_or_str(_first_nonempty(d, _TITLE_KEYS)) or ""
            if not rec.doc_number:
                # Split-on-CSV is a typing concern (the dataclass
                # field is ``list[str]``), not presentation
                # normalization. The per-token cleanup is handled
                # by the ``vbpl_doc_number_list`` chain entry.
                rec.doc_number = normalise_doc_number_list(
                    _none_or_str(_first_nonempty(d, _DOC_NUMBER_KEYS))
                )
            if rec.doc_type is None or rec.legal_type is None:
                raw_dt = _first_nonempty(d, _DOC_TYPE_KEYS)
                if raw_dt is not None:
                    if rec.doc_type is None:
                        # ``doc_type`` is the self-describing snake_case
                        # slug ("quyet_dinh", "thong_tu_lien_tich",
                        # ...). The compact short code ("QĐ", "TTLT")
                        # is still exposed via the doc_number field
                        # itself and via ``codes.SLUG_TO_CANONICAL_CODE``.
                        rec.doc_type = doc_type_slug(raw_dt)
                    if rec.legal_type is None:
                        rec.legal_type = legal_type_name(raw_dt)
            if rec.legal_area is None:
                raw_area = _first_nonempty(d, _LEGAL_AREA_KEYS)
                if raw_area is not None:
                    # ``legal_area_label`` is a typing / lookup
                    # concern (it maps raw API codes to the canonical
                    # area name); ``normalise_label`` (presentation
                    # cleanup) now lives in the chain.
                    rec.legal_area = legal_area_label(raw_area)
            if rec.issue_date is None:
                rec.issue_date = _iso_date(_first_nonempty(d, _ISSUE_DATE_KEYS))
            if rec.issuing_authority is None:
                rec.issuing_authority = _none_or_str(
                    _first_nonempty(d, _AGENCY_KEYS),
                )
            if rec.summary is None:
                rec.summary = _none_or_str(
                    _first_nonempty(d, _SUMMARY_KEYS),
                )
            if not rec.body_html:
                rec.body_html = _str_or_empty(
                    _first_nonempty(d, _BODY_HTML_KEYS),
                )

        rec.file_paths.extend(_extract_files(payload))

    rec.body_text = _html_to_text(rec.body_html)
    rec.raw_api_json = raw
    if rec.api_url is None and raw:
        rec.api_url = next(iter(raw.keys()), None)

    # Defensive fallback for stale-sitemap rows: the gateway returned
    # ``invalid.document.entity.not.found`` for ~3% of corpus URLs, so
    # docType / documentFields never made it into the captured JSON.
    # The URL slug still encodes the Vietnamese doc-type name -- recover
    # the canonical code from the slug front (``thong-tu-so-...`` -> TT).
    if (rec.doc_type is None or rec.legal_type is None) and source_url:
        slug_tail = source_url.rsplit("/", 1)[-1]
        if "--" in slug_tail:
            slug = slug_tail.rsplit("--", 1)[0]
        else:
            slug = slug_tail
        inferred_code = code_from_slug(slug)
        if inferred_code is not None:
            if rec.doc_type is None:
                rec.doc_type = inferred_code
            if rec.legal_type is None:
                rec.legal_type = CANONICAL_CODE_TO_NAME.get(inferred_code)
    if rec.legal_area is None:
        rec.legal_area = "Chưa phân loại"

    # Drop the redundant ``"<legal_type> số <doc_number>"`` head + any
    # cross-references of the shape ``<DocType> <DocNum>`` from the
    # title. :func:`clean_title` runs the full chain (normalise +
    # prefix-strip + crossref-strip) and may return ``None`` when
    # the title is degenerate (e.g. the whole title is just a
    # doc-num token like ``"1938/QĐ-UBND"``).
    if rec.title:
        rec.title = clean_title(rec.title, rec.legal_type, rec.doc_number)
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


# text-cleanup helpers moved to :mod:`normalizers`; re-exported here
# so existing ``from ...parser import clean_title`` call-sites keep working.
# (see grep of the vbpl package for external importers).

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
    "clean_title",
    "detail_record_from_api_json",
    "item_id_from_detail_url",
    "normalise_doc_number",
    "normalise_doc_number_list",
    "normalise_issuing_authority",
    "normalise_label",
    "normalise_text",
    "normalise_title",
    "parse_sitemap_index",
    "parse_sitemap_urlset",
    "scope_from_shard_url",
    "strip_doctype_docnum_crossrefs",
    "strip_markdown_junk",
    "strip_redundant_title_prefix",
]
