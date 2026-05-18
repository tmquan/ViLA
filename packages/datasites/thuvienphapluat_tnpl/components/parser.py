"""HTML parsers for thuvienphapluat.vn/tnpl/.

Three surfaces, all served from the same Vietnamese ASP.NET WebForms
application:

* **Homepage** (``/tnpl/home``) -- carries the 47-entry LinhVuc
  taxonomy in a ``<select name="ctl00$Content$SearchTNPL$ddlField">``
  block, the total term count (``Tìm thấy <b>N</b> thuật ngữ``), and
  the latest ~20 term ids as ``<a class='tnpl' href='/tnpl/{id}/...'>``
  anchors. No real listing pagination -- those ~20 are all the homepage
  exposes.

* **Detail page** (``/tnpl/{id}/{slug}?tab=0``) -- the slug is
  decorative; arbitrary slug returns the same content. Two anchor blocks
  matter:

  - ``<div id="Tab1" class="tabtnpl noidungbaiviet">`` holds the
    Vietnamese term name (first ``<b class='tnpl'>``), an optional
    English label (``<b>Tiếng Anh: </b><b class='tnpl'>...</b>``), the
    definition body, and a trailing ``<p>Lĩnh vực: <b>...</b></p>
    <p>Tình trạng: <b>...</b></p>``.
  - ``<div id="Tab4" class="tabtnpl">`` (history) holds the last-update
    audit line ``Cập nhật bởi <b class='tnpl'>USER</b> <i>HH:mm dd/MM/yyyy</i>``.

* **Soft-404 body** -- missing or retracted IDs return HTTP 200 with
  ``Không tìm thấy thuật ngữ này!`` in the body and no Tab1 block. The
  detail parser returns ``None`` for these so the caller can tag the
  record as ``fetch_status="not_found"``.

All parsers are pure functions; no I/O. Vietnamese text is preserved
verbatim (no lowercasing, no diacritic stripping). The date is
best-effort parsed from ``HH:mm dd/MM/yyyy`` into ISO 8601 stored as
``cập_nhật_lúc``; the original string is kept in ``cập_nhật_lúc_gốc``
so a stricter consumer can re-parse with its own locale.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)


# ---- regexes -----------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_TOTAL_COUNT_RE = re.compile(r"Tìm thấy\s*<b[^>]*>\s*(\d+)\s*</b>", re.IGNORECASE)
_TNPL_HREF_RE = re.compile(r"^/tnpl/(\d+)/", re.IGNORECASE)
#: Detail-page footer line.
_UPDATE_LINE_RE = re.compile(
    r"Cập nhật bởi\s*(?:<[^>]+>)?\s*(?P<who>[^<]+?)\s*</?\w[^>]*>\s*"
    r"(?:<[^>]+>)?\s*(?P<when>\d{1,2}:\d{2}\s+\d{1,2}/\d{1,2}/\d{4})",
)
_DATETIME_RE = re.compile(r"(?P<hh>\d{1,2}):(?P<mm>\d{2})\s+(?P<d>\d{1,2})/(?P<mo>\d{1,2})/(?P<y>\d{4})")
_NOT_FOUND_SENTINEL = "Không tìm thấy thuật ngữ này"


# ----------------------------------------------------------------------
# Homepage parsers
# ----------------------------------------------------------------------


def parse_taxonomy(index_html: str) -> dict[int, str]:
    """Extract the LinhVuc id→Vietnamese-name map from the homepage.

    The taxonomy lives in the ``<select name="ctl00$Content$SearchTNPL$ddlField">``
    block. The first option (``value="0"`` / ``Tất cả``) is the
    placeholder and is dropped. Returns ``{id: name}`` with 47 entries
    for the canonical taxonomy (as of 2026-05).
    """
    if not index_html:
        return {}
    soup = BeautifulSoup(index_html, "html.parser")
    sel = soup.find(
        "select",
        attrs={"name": "ctl00$Content$SearchTNPL$ddlField"},
    )
    if sel is None:
        return {}
    out: dict[int, str] = {}
    for opt in sel.find_all("option"):
        try:
            v = int(str(opt.get("value")))
        except (TypeError, ValueError):
            continue
        if v <= 0:
            continue
        name = _clean_text(opt.get_text(" "))
        if name:
            out[v] = name
    return out


def parse_total_count(index_html: str) -> int | None:
    """Extract the ``Tìm thấy <b>N</b> thuật ngữ`` total term count.

    Returns ``None`` when the marker is absent (e.g. fetch failure
    yielded an error page); callers may fall back to the largest
    bootstrap id + a safety buffer.
    """
    if not index_html:
        return None
    m = _TOTAL_COUNT_RE.search(index_html)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def parse_homepage_ids(index_html: str) -> list[int]:
    """Extract every ``/tnpl/{id}/...`` term id linked from the homepage.

    The homepage server-renders ~20 most-recently-updated terms as
    ``<a class='tnpl' href='/tnpl/{id}/slug?tab=0'>``. We harvest the
    full set of distinct ids in document order; the caller uses
    ``max(ids)`` as the upper bound for the brute-force ID probe range.
    """
    if not index_html:
        return []
    soup = BeautifulSoup(index_html, "html.parser")
    seen: set[int] = set()
    out: list[int] = []
    for a in soup.select("a.tnpl[href]"):
        href = str(a.get("href") or "")
        m = _TNPL_HREF_RE.match(href)
        if not m:
            continue
        try:
            tid = int(m.group(1))
        except ValueError:
            continue
        if tid in seen:
            continue
        seen.add(tid)
        out.append(tid)
    return out


# ----------------------------------------------------------------------
# Detail parser
# ----------------------------------------------------------------------


@dataclass
class DetailRecord:
    """One ``<div id="Tab1">`` block, parsed.

    Attribute names match the JSONL column names in
    :data:`packages.datasites.thuvienphapluat_tnpl._shared.DETAIL_JSONL_FIELDS`
    one-to-one (Python 3 identifiers accept non-ASCII so the diacritics
    round-trip without a rename layer).
    """

    tên_thuật_ngữ: str = ""
    tên_thuật_ngữ_gốc_tiếng_anh: str | None = None
    định_nghĩa: str = ""
    lĩnh_vực: str | None = None
    tình_trạng: str | None = None
    cập_nhật_bởi: str | None = None
    cập_nhật_lúc_gốc: str | None = None
    cập_nhật_lúc: str | None = None
    thuật_ngữ_liên_quan_ids: list[int] = field(default_factory=list)
    thuật_ngữ_liên_quan: list[str] = field(default_factory=list)


def parse_detail_fragment(html: str) -> DetailRecord | None:
    """Parse a ``/tnpl/{id}/...`` detail page.

    Returns ``None`` for the soft-404 case (body contains
    ``Không tìm thấy thuật ngữ này`` or no ``#Tab1`` block). The
    caller writes a stub row with ``fetch_status="not_found"`` so the
    gap is auditable from the JSONL alone.
    """
    if not html:
        return None
    if _NOT_FOUND_SENTINEL in html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    tab1 = soup.select_one("#Tab1.tabtnpl")
    if tab1 is None:
        return None

    rec = DetailRecord()

    # Term name + optional English label. The first <b class='tnpl'> is
    # always the Vietnamese term. When an English label follows, the
    # next-but-one bold is preceded by a ``<b>Tiếng Anh: </b>`` literal
    # so we anchor on that text.
    bold_tnpls = tab1.find_all("b", class_="tnpl")
    if bold_tnpls:
        rec.tên_thuật_ngữ = _clean_text(bold_tnpls[0].get_text(" "))
    en_label = tab1.find(
        "b",
        string=lambda s: bool(s) and "Tiếng Anh" in str(s),
    )
    if en_label is not None:
        # The English term name is the next sibling <b class='tnpl'>.
        en_bold = en_label.find_next("b", class_="tnpl")
        if en_bold is not None:
            v = _clean_text(en_bold.get_text(" "))
            if v:
                rec.tên_thuật_ngữ_gốc_tiếng_anh = v

    # LinhVuc + status come from inline <p> blocks at the bottom of Tab1.
    # We use a single pass over Tab1's <p> tags rather than CSS
    # selectors because the structure is fragile (no stable id/class).
    for p in tab1.find_all("p"):
        text = _clean_text(p.get_text(" "))
        if not text:
            continue
        if text.startswith("Lĩnh vực:"):
            val = text[len("Lĩnh vực:") :].strip()
            if val:
                rec.lĩnh_vực = val
        elif text.startswith("Tình trạng:"):
            val = text[len("Tình trạng:") :].strip()
            if val:
                rec.tình_trạng = val

    # Definition body = everything inside Tab1 after the term-name
    # block(s) and before the Lĩnh vực / Tình trạng line. We build it
    # by cloning Tab1, dropping the now-known structural children, and
    # serialising what's left -- robust to the wide variety of
    # ``<span>``/``<sub>``/``<p>`` formatting the source uses.
    rec.định_nghĩa = _extract_definition_text(tab1)

    # Cross-references = every /tnpl/N/... <a> inside the definition.
    seen: set[int] = set()
    for a in tab1.select("a[href]"):
        href = str(a.get("href") or "")
        m = _TNPL_HREF_RE.match(href)
        if not m:
            continue
        try:
            tid = int(m.group(1))
        except ValueError:
            continue
        if tid in seen:
            continue
        seen.add(tid)
        rec.thuật_ngữ_liên_quan_ids.append(tid)
        rec.thuật_ngữ_liên_quan.append(_clean_text(a.get_text(" ")))

    # Last-update audit line lives in Tab4 (history). Fall back to a
    # whole-page regex when Tab4 is empty or absent.
    tab4 = soup.select_one("#Tab4.tabtnpl")
    audit_html = str(tab4) if tab4 is not None else html
    m = _UPDATE_LINE_RE.search(audit_html)
    if m:
        rec.cập_nhật_bởi = _clean_text(m.group("who"))
        rec.cập_nhật_lúc_gốc = m.group("when").strip()
        rec.cập_nhật_lúc = _iso_datetime(rec.cập_nhật_lúc_gốc)

    # Sanity: empty parse -> treat as not-found so the caller can tag
    # it cleanly instead of writing a half-populated record.
    if not (rec.tên_thuật_ngữ or rec.định_nghĩa):
        return None
    return rec


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _extract_definition_text(tab1: Tag) -> str:
    """Return the non-HTML definition text from a #Tab1 block.

    Strategy: walk Tab1's direct children, accumulate the text for
    everything *between* the term-name block and the
    ``Lĩnh vực: ... Tình trạng: ...`` trailer. Persisted JSONL never
    carries source HTML; ``bs4`` gives us decoded text via ``.get_text()``.
    """
    keep_text: list[str] = []
    started = False
    for child in tab1.children:
        if isinstance(child, Tag):
            cls = " ".join(child.get("class") or [])
            text = _clean_text(child.get_text(" "))
            # Skip the leading term-name `<b class='tnpl'>` and the
            # optional English-label `<b>Tiếng Anh: </b><b class='tnpl'>...</b>`
            # sequence before the definition body starts.
            if not started:
                if child.name == "b" and "tnpl" in cls:
                    continue
                if child.name == "b" and text.startswith("Tiếng Anh"):
                    continue
                if child.name == "br":
                    continue
                # An empty `<div class='clr px10'>` separates the
                # title block from the definition body; the FIRST one
                # is the signal that body content follows.
                if child.name == "div" and "clr" in cls and not text:
                    started = True
                    continue
                # Any other tag (a non-empty <p> or <div>) is the start
                # of the body itself. Keep it.
                started = True
            # We've started the body. Trailing classifiers are <p> tags
            # beginning with "Lĩnh vực:" / "Tình trạng:". Drop them so
            # the definition string is just the definition.
            if child.name == "p" and (
                text.startswith("Lĩnh vực:") or text.startswith("Tình trạng:")
            ):
                continue
            # Drop the empty trailing <div class='clr px10'> separator.
            if child.name == "div" and "clr" in cls and not text:
                continue
            if text:
                keep_text.append(text)
        else:
            # NavigableString -- keep raw text payload in both views.
            s = str(child)
            if started:
                cleaned = _clean_text(s)
                if cleaned:
                    keep_text.append(cleaned)
    return " ".join(keep_text).strip()


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).replace("\xa0", " ").replace("\u00a0", " ")
    return _WS_RE.sub(" ", s).strip()


def _iso_datetime(raw: str | None) -> str | None:
    """Parse ``HH:mm dd/MM/yyyy`` (Vietnamese locale) into ISO 8601.

    Returns ``None`` on any malformed input rather than raising so the
    crawler can keep the original string in ``cập_nhật_lúc_gốc`` for
    re-parsing by a stricter consumer.
    """
    if not raw:
        return None
    m = _DATETIME_RE.search(raw)
    if not m:
        return None
    try:
        hh = int(m.group("hh"))
        mm = int(m.group("mm"))
        d = int(m.group("d"))
        mo = int(m.group("mo"))
        y = int(m.group("y"))
        return datetime(y, mo, d, hh, mm).isoformat(timespec="seconds")
    except ValueError:
        return None


__all__ = [
    "DetailRecord",
    "parse_detail_fragment",
    "parse_homepage_ids",
    "parse_taxonomy",
    "parse_total_count",
]
