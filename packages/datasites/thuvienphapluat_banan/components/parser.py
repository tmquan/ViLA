"""Pure-function HTML → dataclass extractors for thuvienphapluat_banan.

Two surfaces:

* :func:`parse_listing_page` reads a paginated search result
  (``/banan/tim-ban-an?type_q=0&sortType=1&page=N``) and yields one
  :class:`ListingEntry` per ``/banan/ban-an/<slug>-<id>`` card.
* :func:`parse_detail_page` reads a single detail page
  (``/banan/ban-an/<slug>-<id>``) and returns a :class:`DetailRecord`
  with the sidebar metadata (``Tên bản án``, ``Cơ quan ban hành``,
  ``Số hiệu``, ``Cấp xét xử``, ``Lĩnh vực``, ``Ngày ban hành``,
  ``Từ khóa``) plus the active-tab judgment body.

Both functions are side-effect-free + cheap to unit-test against
fixture HTML; the harvester / downloader own the rate-limited HTTP +
disk cache layers around them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from packages.datasites.thuvienphapluat_banan._shared import (
    CASE_KIND_VI_TO_EN,
    PROCEDURE_VI_TO_EN,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------


@dataclass
class ListingEntry:
    """One judgment card from a paginated listing page."""

    ban_an_id: int
    slug: str
    url: str
    title: str | None = None
    summary: str | None = None
    doc_number: str | None = None
    court: str | None = None
    issue_date: str | None = None  # raw "dd/mm/yyyy" as shown on the card
    case_kind: str | None = None
    procedure: str | None = None


@dataclass
class DetailRecord:
    """One judgment's sidebar metadata + body, extracted from detail HTML."""

    ban_an_id: int
    source_url: str
    slug: str | None = None
    title: str | None = None
    court: str | None = None
    doc_number: str | None = None
    trial_level: str | None = None
    legal_area: str | None = None
    issue_date_raw: str | None = None
    issue_date: str | None = None
    keywords: list[str] = field(default_factory=list)
    related_doc_ids: list[int] = field(default_factory=list)
    body_html: str | None = None
    body_text: str | None = None


# -----------------------------------------------------------------------
# Detail-URL helpers
# -----------------------------------------------------------------------

#: Regex for ``/banan/ban-an/<slug>-<id>`` where ``<id>`` is the
#: trailing 5–8-digit integer. We capture the slug + id; the slug may
#: contain hyphens itself so we anchor on the final ``-<digits>``.
_DETAIL_URL_RE = re.compile(
    r"^(?P<prefix>.*?/banan/ban-an/)(?P<slug>.+?)-(?P<id>\d+)(?:\?[^#]*)?(?:#.*)?$",
)


def parse_detail_url(url: str) -> tuple[int, str] | None:
    """Pull ``(ban_an_id, slug)`` out of a detail URL. Returns ``None`` on miss."""
    if not url:
        return None
    m = _DETAIL_URL_RE.match(url.strip())
    if not m:
        return None
    try:
        return int(m.group("id")), m.group("slug")
    except (TypeError, ValueError):
        return None


# -----------------------------------------------------------------------
# Listing-page parser
# -----------------------------------------------------------------------

#: Cards on the search listing live in ``<a href="/banan/ban-an/...">``.
#: We dedup by ``ban_an_id`` because the same card can appear twice on
#: a page (cover image + title) and the page chrome links to related
#: pages with the same URL shape.

# Pull the "Số hiệu: 39/2021/HS-ST", "Tòa án nhân dân ...", date out of
# a textual snippet. The listing-card layout is a flat block of text;
# we grep for the labelled fields.
_NUM_RE  = re.compile(r"\b(\d+/\d{4}/[A-Z\u0110\u0111\u00C1-\u1EF9]+(?:-[A-Z\u0110\u0111\u00C1-\u1EF9]+)*)\b")
_DATE_RE = re.compile(r"\bng\u00e0y\s*(\d{1,2}/\d{1,2}/\d{4})\b", re.I)


#: CSS class on the listing-card outer container. Each judgment card
#: on ``/banan/tim-ban-an?page=N`` is wrapped in a
#: ``<div class="list-group-item-action flex-column align-items-start ...">``;
#: walking by these containers (rather than by raw ``<a>`` tags) keeps
#: per-card sidebar metadata correctly scoped without bleeding across
#: cards.
_LISTING_CARD_SELECTOR = ".list-group-item-action"


def parse_listing_page(
    html: str,
    *,
    page_url: str = "",
) -> list[ListingEntry]:
    """Extract one :class:`ListingEntry` per judgment card on the page."""
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    seen: dict[int, ListingEntry] = {}

    cards = soup.select(_LISTING_CARD_SELECTOR)
    for card in cards:
        anchors = [
            a for a in card.find_all("a", href=True)
            if "/banan/ban-an/" in a["href"]
        ]
        if not anchors:
            continue
        # Pick the first /banan/ban-an/<slug>-<id> anchor with a non-empty
        # text and a non-trivial slug; that's the card title.
        title_anchor = next(
            (a for a in anchors if (a.get_text(strip=True) or "").strip()),
            anchors[0],
        )
        parsed = parse_detail_url(title_anchor["href"])
        if parsed is None:
            continue
        ban_an_id, slug = parsed
        if ban_an_id in seen:
            continue
        title = (title_anchor.get_text(" ", strip=True) or "").strip() or None
        url = _absolute_url(title_anchor["href"], page_url=page_url)
        entry = ListingEntry(
            ban_an_id=ban_an_id, slug=slug, url=url, title=title,
        )
        # Scope all metadata extraction to the card container so doc
        # numbers / courts / dates can't bleed across cards.
        _populate_listing_meta(entry, card.get_text(" ", strip=True))
        seen[ban_an_id] = entry

    out = sorted(seen.values(), key=lambda e: e.ban_an_id, reverse=True)
    return out


def _populate_listing_meta(entry: ListingEntry, text: str) -> None:
    """Fill doc_number / court / issue_date / case_kind / procedure / summary."""
    if not entry.doc_number:
        m = _NUM_RE.search(text)
        if m:
            entry.doc_number = m.group(1)
            ck, proc = _split_case_kind_procedure(entry.doc_number)
            entry.case_kind = ck
            entry.procedure = proc
    if not entry.issue_date:
        m = _DATE_RE.search(text)
        if m:
            entry.issue_date = m.group(1)
    if not entry.court:
        # "Tòa án nhân dân ..." is the canonical prefix.
        m = re.search(r"(T\u00f2a [^.]+?)(?:\s*[\u2014\u2013\-]|\s*S\u1ed1\s*hi\u1ec7u|\s*$)", text, re.I)
        if m:
            entry.court = m.group(1).strip()
    if not entry.summary:
        # Strip the title from the head and use the remainder, capped.
        s = text
        if entry.title and entry.title in s:
            s = s.replace(entry.title, "", 1).strip()
        entry.summary = s[:400] if s else None


# -----------------------------------------------------------------------
# Detail-page parser
# -----------------------------------------------------------------------

#: Sidebar key → ``DetailRecord`` attribute. Right-hand side is the
#: dataclass field name; left is the literal Vietnamese label the
#: portal renders inside ``.list-group.detail-item .list-group-item``.
_DETAIL_SIDEBAR_KEYS: dict[str, str] = {
    "Tên bản án":       "title",
    "Cơ quan ban hành": "court",
    "Số hiệu":          "doc_number",
    "Cấp xét xử":       "trial_level",
    "Lĩnh vực":         "legal_area",
    "Ngày ban hành":    "issue_date_raw",
    "Từ khóa":          "_keywords_raw",
}


def parse_detail_page(html: str, *, source_url: str = "") -> DetailRecord | None:
    """Parse a detail HTML body and return a :class:`DetailRecord`.

    Returns ``None`` when the document is not a recognisable detail
    page (e.g. soft-404 from the WAF, or the ``pagenotfound.htm``
    redirect target).
    """
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")

    # Page-not-found / soft-404 surfaces — short-circuit early so the
    # downloader can tag the row as ``not_found`` without burning the
    # full parser.
    text_blob = soup.get_text(" ", strip=True)
    if not text_blob or len(text_blob) < 50:
        return None
    if "pagenotfound" in (source_url or "").lower():
        return None
    # Some not-found pages render the homepage banner with no
    # ``.tab-content`` block. We detect that by the absence of any
    # ``.detail-item`` sidebar AND a missing tab-content.
    tab_content = soup.find(class_="tab-content")
    sidebar_items = soup.select(".list-group.detail-item .list-group-item, .detail-item .list-group-item")
    if not tab_content and not sidebar_items:
        return None

    parsed = parse_detail_url(source_url) if source_url else None
    ban_an_id = parsed[0] if parsed else 0
    slug = parsed[1] if parsed else None

    rec = DetailRecord(ban_an_id=ban_an_id, source_url=source_url, slug=slug)

    # ---- sidebar metadata ------------------------------------------------
    # The portal renders each metadata row as a ``<div class="row w-100">``
    # with two children: ``.col-xl-3`` containing the ``<b>Label:</b>``
    # and ``.col-xl-9`` containing the value. The enclosing ``<li>``
    # tags are nested inside each other rather than siblings (the
    # source HTML never closes ``</li>`` before opening the next one),
    # so naive ``get_text()`` on the outer ``<li>`` swallows the entire
    # sidebar. Walking ``.row.w-100`` direct row elements avoids the
    # nesting trap entirely.
    keywords_raw: str | None = None
    for row in soup.select(".detail-item .row.w-100"):
        cols = row.find_all("div", recursive=False)
        if len(cols) < 2:
            continue
        label_col, value_col = cols[0], cols[1]
        label_el = label_col.find(["label", "strong", "b"]) or label_col
        label = (label_el.get_text(" ", strip=True) or "").rstrip(":").strip()
        if not label:
            continue
        # The keyword row renders the values as ``<a>``-anchored chips
        # (one anchor per keyword). For every other row a plain text
        # readout is the most faithful representation.
        if label == "Từ khóa":
            anchors = value_col.find_all("a")
            if anchors:
                keywords_raw = ", ".join(
                    (a.get_text(" ", strip=True) or "").strip()
                    for a in anchors
                    if a.get_text(strip=True)
                )
            else:
                keywords_raw = value_col.get_text(" ", strip=True)
            continue
        attr = _DETAIL_SIDEBAR_KEYS.get(label)
        if attr is None:
            continue
        value_text = value_col.get_text(" ", strip=True)
        setattr(rec, attr, value_text or None)

    rec.keywords = _split_keywords(keywords_raw) if keywords_raw else []
    rec.issue_date = _iso_date(rec.issue_date_raw) or None

    # ---- body (the first / active tab is "Nội dung bản án") -------------
    if tab_content is not None:
        active = (
            tab_content.find(class_=["tab-pane", "active"], recursive=False)
            or tab_content.find(class_="active")
            or tab_content.find(class_="tab-pane")
        )
        if active is not None:
            rec.body_html = str(active)
            rec.body_text = active.get_text("\n", strip=True)

    # ---- related document ids (xét lại / dẫn chiếu / căn cứ) ------------
    rec.related_doc_ids = _extract_related_ids(soup, current_id=rec.ban_an_id)

    return rec


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

#: Strip 1+ punctuation / whitespace runs into a single space.
_WS_RE = re.compile(r"\s+", re.UNICODE)


def _absolute_url(href: str, *, page_url: str) -> str:
    if href.startswith(("http://", "https://")):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        # Build origin from page_url; fall back to the site origin.
        origin = _origin(page_url) or "https://thuvienphapluat.vn"
        return origin.rstrip("/") + href
    return href


def _origin(url: str) -> str | None:
    if not url:
        return None
    m = re.match(r"^(https?://[^/]+)", url)
    return m.group(1) if m else None


def _split_keywords(text: str) -> list[str]:
    if not text:
        return []
    # The portal uses commas + " , " around each token.
    parts = [p.strip() for p in re.split(r"\s*[,;]\s*", text) if p.strip()]
    # Dedup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        key = p.casefold()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _iso_date(raw: str | None) -> str | None:
    """Convert ``"26/07/2021"`` → ``"2021-07-26"``. Returns ``None`` on miss."""
    if not raw:
        return None
    m = re.match(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$", raw)
    if not m:
        return None
    d, mo, y = (int(g) for g in m.groups())
    if not (1 <= mo <= 12 and 1 <= d <= 31 and 1900 <= y <= 2100):
        return None
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _split_case_kind_procedure(doc_number: str) -> tuple[str | None, str | None]:
    """``"39/2021/HS-ST"`` → ``("HS", "ST")``; returns ``(None, None)`` on miss."""
    if not doc_number:
        return None, None
    m = re.match(r"^\s*\d+/\d{4}/(?P<suffix>[A-Z\u0110\u0111][A-Z\u0110\u0111-]+)\s*$", doc_number)
    if not m:
        return None, None
    suffix = m.group("suffix")
    if "-" in suffix:
        kind, proc = suffix.split("-", 1)
    else:
        kind, proc = suffix, None
    # Normalise against the known maps (pass through verbatim otherwise).
    kind_norm = kind if kind in CASE_KIND_VI_TO_EN else kind
    proc_norm = proc if proc and proc in PROCEDURE_VI_TO_EN else proc
    return kind_norm, proc_norm


def _extract_related_ids(soup: BeautifulSoup, *, current_id: int) -> list[int]:
    """Collect ``ban_an_id`` of every ``/banan/ban-an/...`` link on the page."""
    out: list[int] = []
    seen: set[int] = {current_id}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/banan/ban-an/" not in href:
            continue
        parsed = parse_detail_url(href)
        if parsed is None:
            continue
        bid = parsed[0]
        if bid in seen:
            continue
        seen.add(bid)
        out.append(bid)
    return out


__all__ = [
    "DetailRecord",
    "ListingEntry",
    "parse_detail_page",
    "parse_detail_url",
    "parse_listing_page",
]
