"""Canonical Vietnamese legal-document codes for vbpl.

The vbpl.vn gateway exposes a ``docType`` block on every detail
response::

    {"id": ..., "name": "Chỉ thị", "code": "CThi", "parentCode": null}

…and a ``documentFields[]`` list of legal-area labels::

    {"id": ..., "name": "Đất đai", "code": null}

The raw ``code`` values are inconsistent (legacy CMS prefixes like
``LVB-``, ``LVBM-``, and ``_TW``; weird abbreviations like ``CThi``
and ``Lu``), so downstream consumers see noise where there should be a
clean enumeration. This module is the single source of truth for the
canonical Vietnamese abbreviation of each document type, plus helpers
that map the raw API blocks onto

* :func:`canonical_code` -- short Vietnamese abbreviation that
  matches the abbreviations used in the document number itself
  (``43/2026/NĐ-CP`` -> ``NĐ``).
* :func:`legal_type_name` -- the canonical Vietnamese full name
  (``"Nghị định"``).
* :func:`legal_area_label` -- the canonical Vietnamese area name
  pulled from ``documentFields[]`` (``"Đất đai"`` etc.), defaulting
  to the literal ``"Chưa phân loại"`` ("uncategorised") when the
  list is empty or all-null.

The mapping was derived from a full audit of all 158,822 captured
API responses on 2026-05-19 (see ``/tmp/audit_codes.json``); every
observed ``(raw_code, name)`` pair is covered. New codes that
appear in future crawls fall through ``canonical_code`` unchanged
so the table can be extended additively without dropping rows.
"""

from __future__ import annotations

from typing import Any, Iterable

#: Map of *raw* ``docType.code`` -> canonical short code. Built from
#: a full audit of the corpus. Codes already in canonical form
#: (``QĐ``, ``NQ``, ``TT``, ...) are listed as identity entries so
#: :func:`canonical_code` can short-circuit on a single dict lookup
#: instead of an if-else chain.
RAW_CODE_TO_CANONICAL: dict[str, str] = {
    # Already canonical
    "QĐ":   "QĐ",     # Quyết định
    "NQ":   "NQ",     # Nghị quyết
    "TT":   "TT",     # Thông tư
    "BD":   "BD",     # Bản dịch văn bản
    "NĐ":   "NĐ",     # Nghị định
    "VBHN": "VBHN",   # Văn bản hợp nhất
    "PL":   "PL",     # Pháp lệnh
    "VBHC": "VBHC",   # Văn bản hành chính liên quan
    "NQLT": "NQLT",   # Nghị quyết liên tịch
    "BL":   "BL",     # Bộ luật
    "HP":   "HP",     # Hiến pháp
    # Legacy noise: strip prefixes / fix abbreviations
    "CThi":      "CT",     # Chỉ thị (user request: CThi -> CT)
    "Lu":        "L",      # Luật (single-letter canonical)
    "L-CTN":     "L-CTN",  # Lệnh Chủ tịch nước (kept distinct from "L" Luật)
    "TTLT_TW":   "TTLT",   # Thông tư liên tịch
    "LVB-SLe":   "SL",     # Sắc lệnh
    "LVB-VBLQ":  "VBLQ",   # Văn bản liên quan
    "LVB-CTr":   "CTr",    # Chương trình
    "LVB-VBK":   "VBK",    # Văn bản khác
    "LVBM-CV":   "CV",     # Công văn
    "LVBM-HD":   "HĐ",     # Hiệp định (tone-mark canonical)
    "LVBM-TB":   "TB",     # Thông báo
    "LVBM-TTLB": "TTLB",   # Thông tư liên bộ (legacy)
    "LVBM-SL":   "SLT",    # Sắc luật (distinct from "SL" Sắc lệnh)
    "LVBM-NDT":  "NĐT",    # Nghị định thư
    "LVBM-BGN":  "BGN",    # Bản ghi nhớ
    "LVBM-TT":   "ThT",    # Thỏa thuận (kept distinct from "TT" Thông tư)
    "CHUA_XAC_DINH": "KXĐ",  # Chưa xác định
}


#: Canonical ASCII snake_case slug keyed by canonical short code.
#: This is the value written into the ``doc_type`` column of the
#: published parquet -- a self-describing identifier that a reader
#: can interpret without consulting a separate codebook. The slug
#: is the diacritic-stripped, lowercase, underscore-joined form of
#: :data:`CANONICAL_CODE_TO_NAME` (e.g. ``"Quyết định" -> "quyet_dinh"``,
#: ``"Thông tư liên tịch" -> "thong_tu_lien_tich"``) so each value
#: round-trips losslessly to the Vietnamese name via
#: :func:`legal_type_name`. Short codes like ``QĐ`` / ``TTLT`` are
#: still exposed in :data:`CANONICAL_CODE_TO_NAME` and in
#: ``so_hieu`` itself (``"143/QĐ-KHTC"``), so consumers who need the
#: compact form can recover it from :data:`SLUG_TO_CANONICAL_CODE`.
CANONICAL_CODE_TO_SLUG: dict[str, str] = {
    "HP":    "hien_phap",
    "BL":    "bo_luat",
    "L":     "luat",
    "PL":    "phap_lenh",
    "L-CTN": "lenh",
    "NQ":    "nghi_quyet",
    "NĐ":    "nghi_dinh",
    "QĐ":    "quyet_dinh",
    "TT":    "thong_tu",
    "CT":    "chi_thi",
    "SL":    "sac_lenh",
    "SLT":   "sac_luat",
    "VBHN":  "van_ban_hop_nhat",
    "TTLT":  "thong_tu_lien_tich",
    "NQLT":  "nghi_quyet_lien_tich",
    "TTLB":  "thong_tu_lien_bo",
    "CV":    "cong_van",
    "TB":    "thong_bao",
    "HĐ":    "hiep_dinh",
    "NĐT":   "nghi_dinh_thu",
    "BGN":   "ban_ghi_nho",
    "ThT":   "thoa_thuan",
    "VBHC":  "van_ban_hanh_chinh_lien_quan",
    "VBK":   "van_ban_khac",
    "VBLQ":  "van_ban_lien_quan",
    "CTr":   "chuong_trinh",
    "BD":    "ban_dich_van_ban",
    "KXĐ":   "chua_xac_dinh",
}

#: Reverse lookup ``slug -> canonical short code`` so legacy
#: consumers and downstream re-aggregators can recover the
#: abbreviated form (or canonical code) from the parquet's
#: ``doc_type`` slug.
SLUG_TO_CANONICAL_CODE: dict[str, str] = {
    v: k for k, v in CANONICAL_CODE_TO_SLUG.items()
}


#: Canonical Vietnamese full name keyed by canonical short code.
#: Used by :func:`legal_type_name` when the raw API response is
#: missing the ``name`` field, and as a documentation anchor.
CANONICAL_CODE_TO_NAME: dict[str, str] = {
    "HP":   "Hiến pháp",
    "BL":   "Bộ luật",
    "L":    "Luật",
    "PL":   "Pháp lệnh",
    "L-CTN": "Lệnh",
    "NQ":   "Nghị quyết",
    "NĐ":   "Nghị định",
    "QĐ":   "Quyết định",
    "TT":   "Thông tư",
    "CT":   "Chỉ thị",
    "SL":   "Sắc lệnh",
    "SLT":  "Sắc luật",
    "VBHN": "Văn bản hợp nhất",
    "TTLT": "Thông tư liên tịch",
    "NQLT": "Nghị quyết liên tịch",
    "TTLB": "Thông tư liên bộ",
    "CV":   "Công văn",
    "TB":   "Thông báo",
    "HĐ":   "Hiệp định",
    "NĐT":  "Nghị định thư",
    "BGN":  "Bản ghi nhớ",
    "ThT":  "Thỏa thuận",
    "VBHC": "Văn bản hành chính liên quan",
    "VBK":  "Văn bản khác",
    "VBLQ": "Văn bản liên quan",
    "CTr":  "Chương trình",
    "BD":   "Bản dịch văn bản",
    "KXĐ":  "Chưa xác định",
}


#: Reverse lookup name -> canonical code, used to backfill code when
#: the raw API response carries a name but the ``code`` field is
#: empty (~40 docs in the corpus). Case-insensitive lookup.
_NAME_TO_CANONICAL: dict[str, str] = {
    v.lower(): k for k, v in CANONICAL_CODE_TO_NAME.items()
}

#: Sentinel for documents whose API response did not expose a
#: ``documentFields[]`` entry (or every entry was ``null``). Matches
#: the literal vbpl.vn uses when an editor hasn't tagged a doc yet.
UNCATEGORISED_AREA: str = "Chưa phân loại"


#: Map of URL-slug prefix -> canonical code, used as a defensive
#: fallback for ~3% of corpus rows where the detail-API returned
#: ``invalid.document.entity.not.found`` (the document was removed
#: from the gateway but its URL still appears in the public
#: sitemap). The slug encodes the Vietnamese name of the doc type
#: at the very front (``thong-tu-so-30-2014-tt-bca-...``,
#: ``nghi-dinh-so-89-2016-nd-cp-...``); matching it against the
#: ``CANONICAL_CODE_TO_NAME`` reverse table recovers the doc type
#: for these rows without re-mining anything from the body.
#:
#: Order matters: longer prefixes must come first so
#: ``van-ban-hop-nhat`` matches before the no-such-prefix
#: ``van-ban-...`` ones.
_SLUG_PREFIX_TO_CODE: tuple[tuple[str, str], ...] = (
    ("van-ban-hop-nhat-", "VBHN"),
    ("thong-tu-lien-tich-", "TTLT"),
    ("thong-tu-lien-bo-", "TTLB"),
    ("nghi-quyet-lien-tich-", "NQLT"),
    ("nghi-dinh-thu-", "NĐT"),
    ("phap-lenh-", "PL"),
    ("nghi-quyet-", "NQ"),
    ("nghi-dinh-", "NĐ"),
    ("quyet-dinh-", "QĐ"),
    ("thong-tu-", "TT"),
    ("chi-thi-", "CT"),
    ("sac-lenh-", "SL"),
    ("sac-luat-", "SLT"),
    ("hien-phap-", "HP"),
    ("bo-luat-", "BL"),
    ("luat-",   "L"),
    ("lenh-",   "L-CTN"),
    ("cong-van-", "CV"),
    ("thong-bao-", "TB"),
    ("hiep-dinh-", "HĐ"),
    ("ban-ghi-nho-", "BGN"),
    ("thoa-thuan-", "ThT"),
    ("ban-dich-", "BD"),
    ("chuong-trinh-", "CTr"),
    ("van-ban-khac-", "VBK"),
    ("van-ban-lien-quan-", "VBLQ"),
    ("van-ban-hanh-chinh-", "VBHC"),
)


def code_from_slug(slug: Any) -> str | None:
    """Recover canonical code from a vbpl URL slug.

    Vbpl URL slugs encode the Vietnamese doc-type name as a kebab-
    case prefix (``thong-tu-so-30-2014-tt-bca-...`` ->
    ``Thông tư`` -> ``TT``). About 3% of corpus rows have no API
    metadata (gateway returned ``invalid.document.entity.not.found``
    for stale sitemap URLs); the slug is the only signal left for
    those rows. Returns ``None`` when no prefix matches.
    """
    if not isinstance(slug, str):
        return None
    s = slug.strip().lower()
    if not s:
        return None
    for prefix, code in _SLUG_PREFIX_TO_CODE:
        if s.startswith(prefix):
            return code
    return None


def doc_type_slug(raw: Any) -> str | None:
    """Map a raw ``docType`` block onto the canonical snake_case slug.

    The slug is the value written into the ``doc_type`` parquet
    column. Inputs are the same shapes :func:`canonical_code`
    accepts (dict, raw string code, canonical short code, ``None``,
    or stringified-dict legacy form). Returns ``None`` only when
    the input is truly unrecognised. Unknown short codes are
    converted on-the-fly via :func:`slugify_vi` so a new code that
    appears in a future crawl gets a sensible slug even if
    :data:`CANONICAL_CODE_TO_SLUG` hasn't been extended yet.

    >>> doc_type_slug({"code": "QĐ", "name": "Quyết định"})
    'quyet_dinh'
    >>> doc_type_slug({"code": "TTLT_TW", "name": "Thông tư liên tịch"})
    'thong_tu_lien_tich'
    >>> doc_type_slug("CThi")
    'chi_thi'
    >>> doc_type_slug(None) is None
    True
    """
    code = canonical_code(raw)
    if code is None:
        return None
    slug = CANONICAL_CODE_TO_SLUG.get(code)
    if slug is not None:
        return slug
    # Unknown short code -- best-effort: derive the slug from the
    # canonical Vietnamese name (or from the code itself if even
    # that is missing). Keeps the column populated for codes added
    # to vbpl after this audit was taken.
    name = CANONICAL_CODE_TO_NAME.get(code)
    return slugify_vi(name) if name else slugify_vi(code)


def slugify_vi(text: str) -> str:
    """Vietnamese-aware ``snake_case`` slug for the doc-type column.

    Strips diacritics (NFD then drop combining marks), maps the
    bare ``đ`` / ``Đ`` to ``d``, lowercases, replaces every
    non-alphanumeric run with a single underscore, and trims
    underscores from both ends.

    >>> slugify_vi('Quyết định')
    'quyet_dinh'
    >>> slugify_vi('Thông tư liên tịch')
    'thong_tu_lien_tich'
    >>> slugify_vi('Văn bản hành chính liên quan')
    'van_ban_hanh_chinh_lien_quan'
    >>> slugify_vi('L-CTN')
    'l_ctn'
    """
    import re
    import unicodedata
    # NFD decomposes "ế" -> "e" + combining acute; ``Mn`` filters
    # out the combining marks, leaving plain ASCII letters. The
    # Vietnamese ``đ``/``Đ`` is its own codepoint (no combining
    # form) so handle it explicitly before the ASCII pass.
    s = (text or "").replace("Đ", "D").replace("đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s


def canonical_code(raw: Any) -> str | None:
    """Map a raw ``docType`` block (dict or string) onto the canonical code.

    Accepts:

    * A dict that looks like ``{"name": ..., "code": ...}`` — the
      shape the vbpl gateway returns.
    * A plain string that is either a raw code (``"CThi"``) or a
      canonical code (``"QĐ"``).
    * A stringified Python dict (the shape that landed in the
      first cut of the published parquet); parsed leniently.
    * ``None`` — returns ``None``.

    Returns ``None`` for inputs that don't match any known shape so
    the caller can decide whether to drop the row or leave the
    field null.
    """
    parsed = _coerce_to_dict(raw)
    if not isinstance(parsed, dict):
        if isinstance(parsed, str) and parsed.strip():
            s = parsed.strip()
            if s in RAW_CODE_TO_CANONICAL:
                return RAW_CODE_TO_CANONICAL[s]
            # Unknown raw code: keep as-is so future codes don't drop.
            return s
        return None

    code = parsed.get("code")
    name = parsed.get("name")
    if isinstance(code, str) and code.strip():
        c = code.strip()
        return RAW_CODE_TO_CANONICAL.get(c, c)
    if isinstance(name, str) and name.strip():
        return _NAME_TO_CANONICAL.get(name.strip().lower())
    return None


def legal_type_name(raw: Any) -> str | None:
    """Return the canonical Vietnamese full name for a doc type block.

    Prefers the raw ``name`` field when present (vbpl is consistent
    on names even when its codes drift); falls back to looking the
    canonical name up by code. Returns ``None`` when neither
    resolves.
    """
    parsed = _coerce_to_dict(raw)
    if isinstance(parsed, dict):
        name = parsed.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    code = canonical_code(raw)
    if code is None:
        return None
    return CANONICAL_CODE_TO_NAME.get(code)


def legal_area_label(fields: Any) -> str | None:
    """Return a single canonical legal-area label from ``documentFields[]``.

    vbpl tags every doc with zero or more legal areas (``Đất đai``,
    ``Đường bộ``, ``Lĩnh vực giá``, ...). About two thirds of the
    corpus is tagged ``"Chưa phân loại"`` ("uncategorised"); we
    preserve that literal so the downstream viz can show the
    uncategorised bucket explicitly instead of silently nulling it.

    When a document carries multiple areas (rare), we take the first
    non-null one — matching how the vbpl SPA renders the sidebar.
    Pass ``None`` or an empty list to get ``"Chưa phân loại"``.
    """
    if fields is None:
        return UNCATEGORISED_AREA
    if isinstance(fields, str):
        s = fields.strip()
        return s or UNCATEGORISED_AREA
    if not isinstance(fields, Iterable):
        return UNCATEGORISED_AREA
    for f in fields:
        if not isinstance(f, dict):
            continue
        name = f.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return UNCATEGORISED_AREA


# ---------------------------------------------------------------- helpers


def _coerce_to_dict(raw: Any) -> Any:
    """Best-effort: parse stringified-dict payloads emitted by the v1 parquet.

    The first cut of ``documents.parquet`` stored ``doc_type`` as a
    repr'd Python dict (``"{'name': 'Quyết định', 'code': 'QĐ', ...}"``).
    Use :func:`ast.literal_eval` to round-trip those without pulling
    a JSON dep — it tolerates the Python ``None`` / single-quote
    encoding that ``json.loads`` would choke on.
    """
    if raw is None or isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                pass
    return raw


__all__ = [
    "CANONICAL_CODE_TO_NAME",
    "CANONICAL_CODE_TO_SLUG",
    "RAW_CODE_TO_CANONICAL",
    "SLUG_TO_CANONICAL_CODE",
    "UNCATEGORISED_AREA",
    "canonical_code",
    "code_from_slug",
    "doc_type_slug",
    "legal_area_label",
    "legal_type_name",
    "slugify_vi",
]
