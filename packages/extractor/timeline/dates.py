"""Vietnamese date-surface-form → ISO normaliser.

The NER extractor emits ``date`` entities verbatim (the LLM is
instructed to keep the source substring). For visual analytics we
need a sortable anchor. This module converts surface forms to a
:class:`packages.extractor.timeline.schema.WhenAnchor` deterministically,
without locale libraries (so the output is stable across hosts).

Coverage for the corpus shipped with this repo (1607 ``date`` entities
across the 140-doc ``samplebanan`` sample):

* ``DD/MM/YYYY`` / ``DD-MM-YYYY`` / ``DD.MM.YYYY`` — full ISO.
* ``DD tháng MM năm YYYY`` (ASCII or with the leading "ngày", any
  case) — full ISO.
* ``tháng MM năm YYYY`` — ``iso=None``, ``iso_partial="YYYY-MM"``.
* ``MM/YYYY`` — ``iso=None``, ``iso_partial="YYYY-MM"``.
* ``năm YYYY`` / four-digit year alone — ``iso=None``,
  ``iso_partial="YYYY"``.

OCR-noise variants ("12/1 2/2022", "30 -12-2016", "Ngày: 19-9-2017",
"từ thán 6/2012 đến thán 4/2015") are folded by:

1. NFC + lowercasing.
2. Stripping a leading "ngày[:]?\\s*" prefix.
3. Collapsing all whitespace inside the candidate.
4. Trying each pattern in order of specificity.

If nothing matches, the surface text is preserved on the anchor and
``iso``/``iso_partial`` are both ``None`` — the event then sorts to
the end-of-corpus bucket (``sort_key = "9999-99-99"``).

Sort-key construction (lexicographically stable, ASCII-only):

* Full ``YYYY-MM-DD`` → ``"YYYY-MM-DD"``.
* Partial ``YYYY-MM``   → ``"YYYY-MM-99"``.
* Partial ``YYYY``      → ``"YYYY-99-99"``.
* Unresolvable          → ``"9999-99-99"``.

Within the same sort_key, events are tie-broken by the event_id
(which the builder constructs in document-offset order).
"""

from __future__ import annotations

import re
import unicodedata

from packages.extractor.timeline.schema import WhenAnchor

# --------------------------------------------------------------------- regexes

# Order matters: most specific first.
#
# We accept either '/', '-', or '.' as numeric separators, with
# optional whitespace around them — OCR sometimes produces
# "30 -12-2016" or "12/1 2/2022".
_NUMERIC_FULL = re.compile(
    r"^\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\s*$",
)

# "DD tháng MM năm YYYY"  — the canonical Vietnamese long form.
_VN_FULL = re.compile(
    r"^\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})\s*$",
)

# "tháng MM năm YYYY"  — month + year only.
_VN_MONTH = re.compile(
    r"^\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})\s*$",
)

# "MM/YYYY"  — numeric month + year only. Must come AFTER _NUMERIC_FULL
# to avoid eating "DD/MM/YYYY".
_NUMERIC_MONTH = re.compile(r"^\s*(\d{1,2})\s*[/\-.]\s*(\d{4})\s*$")

# "năm YYYY"  — explicit year-only.
_VN_YEAR = re.compile(r"^\s*năm\s*(\d{4})\s*$")

# Bare 4-digit year as a fallback.
_BARE_YEAR = re.compile(r"^\s*(\d{4})\s*$")

#: Strip a leading "ngày" / "Ngày:" / "NGÀY" before pattern matching.
_LEADING_NGAY = re.compile(r"^ngày\s*:?\s*", flags=re.IGNORECASE)

#: Sentinel for unresolvable dates. Sorts after every real date.
_UNRESOLVED_SORT_KEY = "9999-99-99"


# --------------------------------------------------------------------- helpers


def _normalise(s: str) -> str:
    """NFC + lowercase + strip leading 'ngày' marker.

    Folding to lowercase is safe — every regex above operates on
    lowercase 'tháng' / 'năm' literals.
    """
    s = unicodedata.normalize("NFC", s)
    s = s.strip()
    s = s.lower()
    s = _LEADING_NGAY.sub("", s)
    return s


def _resolve_year(yy: int) -> int | None:
    """Map a 2- or 4-digit year to a 4-digit year.

    Two-digit years are extremely rare in court judgments but show
    up in OCR'd scans. Use a sliding pivot at 70 — ``00``-``69`` →
    ``2000``-``2069``, ``70``-``99`` → ``1970``-``1999``. Numbers
    outside ``1900..2099`` are rejected as not-a-date.
    """
    if yy < 100:
        return 2000 + yy if yy < 70 else 1900 + yy
    if 1900 <= yy <= 2099:
        return yy
    return None


def _build_full_iso(day: int, month: int, year: int) -> str | None:
    """Return ``YYYY-MM-DD`` if the triple is a valid Gregorian date."""
    y = _resolve_year(year)
    if y is None:
        return None
    if not (1 <= month <= 12):
        return None
    # Per-month day cap; conservative leap-year handling for Feb.
    last_day = (
        31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    )[month - 1]
    if not (1 <= day <= last_day):
        return None
    return f"{y:04d}-{month:02d}-{day:02d}"


def _build_partial_ym(month: int, year: int) -> str | None:
    """Return ``YYYY-MM`` if the pair is plausible, else ``None``."""
    y = _resolve_year(year)
    if y is None:
        return None
    if not (1 <= month <= 12):
        return None
    return f"{y:04d}-{month:02d}"


def _build_partial_y(year: int) -> str | None:
    """Return ``YYYY`` if the year is plausible, else ``None``."""
    y = _resolve_year(year)
    return f"{y:04d}" if y is not None else None


# --------------------------------------------------------------------- API


def parse_date_to_anchor(
    raw: str,
    *,
    page: int | None = None,
) -> WhenAnchor:
    """Convert a surface-form date string to a :class:`WhenAnchor`.

    The function never raises on bad input — it always returns a
    valid :class:`WhenAnchor`. Unresolvable forms keep ``raw`` and
    set ``iso`` and ``iso_partial`` to ``None`` so downstream
    consumers can show the surface text without inventing an
    anchor.
    """
    norm = _normalise(raw)
    iso: str | None = None
    iso_partial: str | None = None

    # 1. Full numeric DD?/MM?/YYYY (slash / hyphen / dot).
    m = _NUMERIC_FULL.match(norm)
    if m:
        d, mo, y = (int(m.group(i)) for i in (1, 2, 3))
        iso = _build_full_iso(d, mo, y)

    # 2. Vietnamese long form.
    if iso is None:
        m = _VN_FULL.match(norm)
        if m:
            d, mo, y = (int(m.group(i)) for i in (1, 2, 3))
            iso = _build_full_iso(d, mo, y)

    # 3. Vietnamese month-year.
    if iso is None:
        m = _VN_MONTH.match(norm)
        if m:
            mo, y = (int(m.group(i)) for i in (1, 2))
            iso_partial = _build_partial_ym(mo, y)

    # 4. Numeric MM/YYYY.
    if iso is None and iso_partial is None:
        m = _NUMERIC_MONTH.match(norm)
        if m:
            mo, y = (int(m.group(i)) for i in (1, 2))
            iso_partial = _build_partial_ym(mo, y)

    # 5. năm YYYY.
    if iso is None and iso_partial is None:
        m = _VN_YEAR.match(norm)
        if m:
            iso_partial = _build_partial_y(int(m.group(1)))

    # 6. Bare 4-digit year (last-resort).
    if iso is None and iso_partial is None:
        m = _BARE_YEAR.match(norm)
        if m:
            iso_partial = _build_partial_y(int(m.group(1)))

    # Sort key construction.
    if iso is not None:
        sort_key = iso  # YYYY-MM-DD
    elif iso_partial is not None:
        # YYYY-MM → "YYYY-MM-99"; YYYY → "YYYY-99-99".
        sort_key = (
            f"{iso_partial}-99" if len(iso_partial) == 7 else f"{iso_partial}-99-99"
        )
    else:
        sort_key = _UNRESOLVED_SORT_KEY

    return WhenAnchor(
        iso=iso,
        iso_partial=iso_partial,
        raw=raw,
        page=page,
        sort_key=sort_key,
    )


__all__ = ["parse_date_to_anchor"]
