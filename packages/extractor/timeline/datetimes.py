"""Vietnamese date/time-surface-form → ISO normaliser.

Renamed from ``dates.py`` in :data:`SCHEMA_VERSION` ``v2`` to reflect
the fact that the module now resolves *both* dates and times. Three
parser paths:

1. :func:`parse_date_to_anchor` — the *absolute* parser, used on
   every NER ``date`` entity. Returns a fully populated
   :class:`packages.extractor.timeline.schema.WhenAnchor` (or an
   unresolvable sentinel anchor) for inputs like

   * ``"21/3/2018"``                              — date only
   * ``"13 tháng 10 năm 2021"``                   — date only
   * ``"tháng 5 năm 2021"`` / ``"năm 2018"``      — partial
   * ``"22 giờ 30 phút ngày 14/3/2023"``          — date + time
   * ``"khoảng 22 giờ ngày 14/3/2023"``           — date + hour-only
   * ``"14 giờ 25 phút"`` / ``"lúc 14:25"``        — time only

2. :func:`parse_relative_to_anchor` — the *relative* parser, used
   on inputs like ``"05 phút sau"``, ``"Trước đó 3 ngày"``,
   ``"Cùng ngày"``, ``"Hôm qua"``. Resolves the surface form
   against a previously established absolute anchor and returns a
   new :class:`WhenAnchor` whose ``iso``/``iso_time`` are the
   resolved date and (when the unit is sub-day) clock time, and
   whose relative-provenance fields (``is_relative``, ``magnitude``,
   ``unit``, ``direction``, ``anchor_event_id``) record the
   resolution. Returns ``None`` when the surface form is not a
   relative expression at all.

3. :func:`find_relative_expressions` — source-text scanner that
   walks an entire NFC-normalised source markdown and yields every
   relative span with its character offsets. Lets the timeline
   builder synthesise ``date_relative`` entities even when the
   upstream NER pass missed them (the regex is more reliable than
   the LLM for this narrow templated task — see
   ``wiki/TIMELINE.md § 3a``).

Coverage for the corpus shipped with this repo (1607 ``date``
entities across the 140-doc ``samplebanan`` sample):

* ``DD/MM/YYYY`` / ``DD-MM-YYYY`` / ``DD.MM.YYYY`` — full ISO.
* ``DD tháng MM năm YYYY`` (ASCII or with the leading "ngày", any
  case) — full ISO.
* ``tháng MM năm YYYY`` — ``iso=None``, ``iso_partial="YYYY-MM"``.
* ``MM/YYYY`` — ``iso=None``, ``iso_partial="YYYY-MM"``.
* ``năm YYYY`` / four-digit year alone — ``iso=None``,
  ``iso_partial="YYYY"``.
* ``HH giờ MM phút ngày D/M/Y`` — full ISO + ``iso_time="HH:MM:00"``.
* ``HH giờ ngày D/M/Y`` — full ISO + ``iso_time="HH:00:00"``.
* ``HH giờ MM phút`` (no date) — ``iso=None`` + ``iso_time="HH:MM:00"``.

OCR-noise variants ("12/1 2/2022", "30 -12-2016", "Ngày: 19-9-2017",
"từ thán 6/2012 đến thán 4/2015") are folded by:

1. NFC + lowercasing.
2. Stripping leading prefixes (``khoảng``, ``vào``, ``lúc``,
   ``ngày[:]?``).
3. Collapsing internal whitespace inside the candidate.
4. Trying each pattern in order of specificity.

If nothing matches, the surface text is preserved on the anchor and
``iso``/``iso_partial``/``iso_time`` are all ``None`` — the event
then sorts to the end-of-corpus bucket
(``sort_key = "9999-99-99T99:99:99"``).

Sort-key construction (lexicographically stable, ASCII-only,
date-then-time so a timed event sorts before an untimed event on
the same calendar day):

* Date + time   → ``"YYYY-MM-DDTHH:MM:SS"``.
* Date only     → ``"YYYY-MM-DDT99:99:99"`` (sorts AFTER all timed
  events on the same calendar day).
* ``YYYY-MM``   → ``"YYYY-MM-99T99:99:99"``.
* ``YYYY``      → ``"YYYY-99-99T99:99:99"``.
* Time only     → ``"9999-99-99THH:MM:SS"``.
* Unresolvable  → ``"9999-99-99T99:99:99"``.

Within the same sort_key, events are tie-broken by the event_id
(which the builder constructs in document-offset order).
"""

from __future__ import annotations

import datetime as _dt
import re
import unicodedata

from packages.extractor.timeline.schema import (
    RelativeDirection,
    RelativeUnit,
    WhenAnchor,
)

# --------------------------------------------------------------------- regexes

# Date-only regexes. Order matters: most specific first.
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

#: Strip leading helper words from the candidate before pattern
#: matching. ``vào`` ("at / on") and ``lúc`` ("at the time of")
#: introduce a clock or a date without changing its meaning;
#: ``ngày`` / ``Ngày:`` is the date marker. ``khoảng``
#: ("approximately") is intentionally NOT stripped here because
#: the relative-temporal classifier uses it as the F5-vague
#: signal (``"khoảng 5 tuần sau"`` → ±25% spread); the absolute
#: date patterns instead accept ``khoảng`` as an optional inline
#: prefix on the time / date components.
_LEADING_PREFIX = re.compile(
    r"^(?:vào|lúc|ngày)\s*:?\s*",
    flags=re.IGNORECASE,
)

#: Optional ``khoảng`` prefix accepted inline in absolute
#: datetime patterns. Kept separate so the relative classifier can
#: still detect ``khoảng`` for F5.
_OPT_KHOANG = r"(?:khoảng\s+)?"

#: Sentinel for unresolvable dates. Sorts after every real date.
_UNRESOLVED_SORT_KEY = "9999-99-99T99:99:99"

#: All-unknown sort-key date component. Used to construct
#: time-only sort keys (``"9999-99-99T14:25:00"``).
_UNKNOWN_DATE_SORT = "9999-99-99"


# Time-only regex group fragments (no anchors). Bound to ``_TIME_*``
# patterns below for the standalone time parser and re-used in the
# combined date+time patterns.
_TIME_VN_HMS = (
    r"(\d{1,2})\s*giờ\s*(\d{1,2})\s*phút\s*(\d{1,2})\s*giây"
)
_TIME_VN_HM = r"(\d{1,2})\s*giờ\s*(\d{1,2})\s*phút"
_TIME_VN_H = r"(\d{1,2})\s*giờ(?:\s|$)"  # avoid eating "giờ MM" prefix
_TIME_NUM_HMS = r"(\d{1,2}):(\d{2}):(\d{2})"
_TIME_NUM_HM = r"(\d{1,2}):(\d{2})"

# Top-level time-only matchers (full string, after _normalise).
# Optional leading "khoảng" accepted inline so the time-only
# parser still works when the surface form is e.g. "khoảng 14 giờ
# 25 phút". The relative classifier uses a separate path so this
# does not collide with F5_KHOANG.
_TIME_ONLY_RES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*" + _OPT_KHOANG + _TIME_VN_HMS + r"\s*$"),
    re.compile(r"^\s*" + _OPT_KHOANG + _TIME_VN_HM + r"\s*$"),
    re.compile(r"^\s*" + _OPT_KHOANG + _TIME_VN_H + r"\s*$"),
    re.compile(r"^\s*" + _OPT_KHOANG + _TIME_NUM_HMS + r"\s*$"),
    re.compile(r"^\s*" + _OPT_KHOANG + _TIME_NUM_HM + r"\s*$"),
)

# Combined "TIME ngày DATE" forms (Vietnamese clock + Vietnamese date).
# Optional leading "khoảng" is accepted inline; "vào " / "lúc " /
# leading "ngày " are already stripped by _normalise. Optional
# inter-token "ngày" prefix on the date side.
_DATETIME_VN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # HH giờ MM phút SS giây ngày D/M/Y
    re.compile(
        r"^\s*" + _OPT_KHOANG + _TIME_VN_HMS
        + r"\s*(?:ngày\s*)?"
        + r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\s*$",
    ),
    # HH giờ MM phút ngày D/M/Y
    re.compile(
        r"^\s*" + _OPT_KHOANG + _TIME_VN_HM
        + r"\s*(?:ngày\s*)?"
        + r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\s*$",
    ),
    # HH giờ ngày D/M/Y
    re.compile(
        r"^\s*" + _OPT_KHOANG + _TIME_VN_H
        + r"\s*(?:ngày\s*)?"
        + r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\s*$",
    ),
    # HH giờ MM phút ngày D tháng M năm Y
    re.compile(
        r"^\s*" + _OPT_KHOANG + _TIME_VN_HM
        + r"\s*(?:ngày\s*)?"
        + r"(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})\s*$",
    ),
    # HH giờ ngày D tháng M năm Y
    re.compile(
        r"^\s*" + _OPT_KHOANG + _TIME_VN_H
        + r"\s*(?:ngày\s*)?"
        + r"(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})\s*$",
    ),
)

# "DD/MM/YYYY HH:MM[:SS]" or "DD/MM/YYYY HH giờ MM phút".
_DATE_THEN_TIME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\s+"
        + _TIME_NUM_HMS + r"\s*$",
    ),
    re.compile(
        r"^\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\s+"
        + _TIME_NUM_HM + r"\s*$",
    ),
    re.compile(
        r"^\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\s+"
        + _TIME_VN_HM + r"\s*$",
    ),
    re.compile(
        r"^\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})\s+"
        + _TIME_VN_H + r"\s*$",
    ),
)


# --------------------------------------------------------------------- helpers


def _normalise(s: str) -> str:
    """NFC + lowercase + strip leading marker words.

    Folding to lowercase is safe — every regex above operates on
    lowercase ``tháng`` / ``năm`` / ``giờ`` / ``phút`` / ``giây``
    literals. Strips ``khoảng`` / ``vào`` / ``lúc`` / ``ngày[:]?``
    so the bare candidate falls through to the pattern table.
    """
    s = unicodedata.normalize("NFC", s)
    s = s.strip()
    s = s.lower()
    # Strip prefixes greedily — "khoảng 22 giờ ngày 14/3/2023" → "22 giờ ngày 14/3/2023";
    # "lúc 14:25" → "14:25".
    while True:
        new = _LEADING_PREFIX.sub("", s)
        if new == s:
            break
        s = new
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


def _validate_hms(h: int, m: int, s: int) -> tuple[int, int, int] | None:
    """Return ``(h, m, s)`` iff every component is in valid range."""
    if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
        return (h, m, s)
    return None


def _parse_time(s: str) -> tuple[int, int, int] | None:
    """Recognise a Vietnamese clock-time surface form.

    Accepts ``HH giờ MM phút SS giây``, ``HH giờ MM phút``,
    ``HH giờ``, ``HH:MM``, ``HH:MM:SS``. Returns a validated
    ``(h, m, s)`` triple, or ``None`` if no match or any component
    is out of range.
    """
    norm = _normalise(s)
    for pat in _TIME_ONLY_RES:
        m = pat.match(norm)
        if m is None:
            continue
        groups = m.groups()
        h = int(groups[0])
        mi = int(groups[1]) if len(groups) >= 2 else 0
        se = int(groups[2]) if len(groups) >= 3 else 0
        return _validate_hms(h, mi, se)
    return None


def _build_iso_time(h: int, m: int, s: int) -> str:
    """Format ``(h, m, s)`` as canonical ``HH:MM:SS``."""
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_sort_key(
    *,
    iso: str | None,
    iso_partial: str | None,
    iso_time: str | None,
) -> str:
    """Compose the lexicographic sort key.

    See module docstring for the exact rules. Date and time
    components are separated by a literal ``"T"`` so the lex order
    matches ISO-8601 datetime order, and an unknown sub-component
    is filled with ``99`` so it sorts AFTER all known values at
    its level (a timed event therefore sorts before an untimed
    event on the same date — exactly the visual order analysts
    expect).
    """
    date_part: str
    if iso is not None:
        date_part = iso  # YYYY-MM-DD
    elif iso_partial is not None:
        date_part = (
            f"{iso_partial}-99" if len(iso_partial) == 7
            else f"{iso_partial}-99-99"
        )
    else:
        date_part = _UNKNOWN_DATE_SORT
    time_part = iso_time if iso_time is not None else "99:99:99"
    return f"{date_part}T{time_part}"


def _build_iso_datetime(iso: str | None, iso_time: str | None) -> str | None:
    """Return ``"YYYY-MM-DDTHH:MM[:SS]"`` when both halves are present."""
    if iso is None or iso_time is None:
        return None
    return f"{iso}T{iso_time}"


def _whenanchor(
    *,
    iso: str | None,
    iso_partial: str | None,
    iso_time: str | None,
    raw: str,
    page: int | None,
    is_relative: bool = False,
    anchor_event_id: str | None = None,
    magnitude: float | None = None,
    unit: RelativeUnit | None = None,
    direction: RelativeDirection | None = None,
    iso_max: str | None = None,
) -> WhenAnchor:
    """Construct a :class:`WhenAnchor` with derived sort_key + iso_datetime."""
    sort_key = _build_sort_key(iso=iso, iso_partial=iso_partial, iso_time=iso_time)
    iso_datetime = _build_iso_datetime(iso, iso_time)
    return WhenAnchor(
        iso=iso,
        iso_partial=iso_partial,
        iso_time=iso_time,
        iso_datetime=iso_datetime,
        raw=raw,
        page=page,
        sort_key=sort_key,
        is_relative=is_relative,
        anchor_event_id=anchor_event_id,
        magnitude=magnitude,
        unit=unit,
        direction=direction,
        iso_max=iso_max,
    )


# --------------------------------------------------------------------- API


def parse_date_to_anchor(
    raw: str,
    *,
    page: int | None = None,
) -> WhenAnchor:
    """Convert a surface-form date / time string to a :class:`WhenAnchor`.

    The function never raises on bad input — it always returns a
    valid :class:`WhenAnchor`. Unresolvable forms keep ``raw`` and
    set ``iso`` / ``iso_partial`` / ``iso_time`` all to ``None`` so
    downstream consumers can show the surface text without
    inventing an anchor.

    Recognised shapes (in order of attempt):

    1. ``HH giờ MM phút (giây) ngày D/M/Y`` and ``D tháng M năm Y``
       → date + time.
    2. ``D/M/Y HH:MM[:SS]`` or ``D/M/Y HH giờ MM phút`` → date + time.
    3. ``D/M/Y``, ``D-M-Y``, ``D.M.Y`` → date only.
    4. ``D tháng M năm Y`` → date only.
    5. ``tháng M năm Y`` / ``M/Y`` → partial ``YYYY-MM``.
    6. ``năm Y`` / bare ``Y`` → partial ``YYYY``.
    7. ``HH giờ MM phút`` / ``HH:MM`` / ``HH giờ`` → time only
       (``iso`` stays ``None``).
    """
    norm = _normalise(raw)
    iso: str | None = None
    iso_partial: str | None = None
    iso_time: str | None = None

    # 1. Combined Vietnamese clock + date forms.
    for pat in _DATETIME_VN_PATTERNS:
        m = pat.match(norm)
        if m is None:
            continue
        groups = m.groups()
        # Time groups occupy the leading positions; the trailing 3
        # are always (day, month, year). HMS forms have 6 groups,
        # HM forms have 5, H-only forms have 4.
        n = len(groups)
        if n == 6:
            h, mi, se, d, mo, y = (int(g) for g in groups)
        elif n == 5:
            h, mi, d, mo, y = (int(g) for g in groups)
            se = 0
        elif n == 4:
            h, d, mo, y = (int(g) for g in groups)
            mi, se = 0, 0
        else:  # pragma: no cover — defensive
            continue
        hms = _validate_hms(h, mi, se)
        iso_candidate = _build_full_iso(d, mo, y)
        if hms is None or iso_candidate is None:
            continue
        iso = iso_candidate
        iso_time = _build_iso_time(*hms)
        break

    # 2. "DD/MM/YYYY HH:MM[:SS]" or "DD/MM/YYYY HH giờ MM phút".
    if iso is None:
        for pat in _DATE_THEN_TIME_PATTERNS:
            m = pat.match(norm)
            if m is None:
                continue
            groups = m.groups()
            d, mo, y = (int(groups[i]) for i in (0, 1, 2))
            n = len(groups)
            if n == 6:
                h, mi, se = (int(groups[i]) for i in (3, 4, 5))
            elif n == 5:
                h, mi = (int(groups[i]) for i in (3, 4))
                se = 0
            elif n == 4:
                h = int(groups[3])
                mi, se = 0, 0
            else:  # pragma: no cover
                continue
            hms = _validate_hms(h, mi, se)
            iso_candidate = _build_full_iso(d, mo, y)
            if hms is None or iso_candidate is None:
                continue
            iso = iso_candidate
            iso_time = _build_iso_time(*hms)
            break

    # 3. Full numeric DD?/MM?/YYYY (date only).
    if iso is None:
        m = _NUMERIC_FULL.match(norm)
        if m:
            d, mo, y = (int(m.group(i)) for i in (1, 2, 3))
            iso = _build_full_iso(d, mo, y)

    # 4. Vietnamese long form (date only).
    if iso is None:
        m = _VN_FULL.match(norm)
        if m:
            d, mo, y = (int(m.group(i)) for i in (1, 2, 3))
            iso = _build_full_iso(d, mo, y)

    # 5. Vietnamese month-year.
    if iso is None:
        m = _VN_MONTH.match(norm)
        if m:
            mo, y = (int(m.group(i)) for i in (1, 2))
            iso_partial = _build_partial_ym(mo, y)

    # 6. Numeric MM/YYYY.
    if iso is None and iso_partial is None:
        m = _NUMERIC_MONTH.match(norm)
        if m:
            mo, y = (int(m.group(i)) for i in (1, 2))
            iso_partial = _build_partial_ym(mo, y)

    # 7. năm YYYY.
    if iso is None and iso_partial is None:
        m = _VN_YEAR.match(norm)
        if m:
            iso_partial = _build_partial_y(int(m.group(1)))

    # 8. Bare 4-digit year (last-resort, but only when nothing
    #    timing-like matched — we check time-only after this to
    #    avoid eating "2025" as a year when the surface was
    #    intentionally meant as a year).
    if iso is None and iso_partial is None:
        m = _BARE_YEAR.match(norm)
        if m:
            iso_partial = _build_partial_y(int(m.group(1)))

    # 9. Time-only (no date). Only attempted if no date matched.
    if iso is None and iso_partial is None:
        hms = _parse_time(raw)
        if hms is not None:
            iso_time = _build_iso_time(*hms)

    return _whenanchor(
        iso=iso,
        iso_partial=iso_partial,
        iso_time=iso_time,
        raw=raw,
        page=page,
    )


# =====================================================================
# Relative temporal parser  (F1..F5 in wiki/TIMELINE.md § 3a)
# =====================================================================
#
# The five families:
#
#   F1. Forward delta    — "X <unit> sau", "Sau X <unit>", "Sau đó X <unit>"
#   F2. Backward delta   — "Trước đó X <unit>", "X <unit> trước",
#                          "Cách đó X <unit>"
#   F3. Same-day deixis  — "Cùng ngày", "Hôm đó", "Cùng lúc",
#                          "Lúc đó"
#   F4. Calendar deixis  — "Hôm qua" (-1d), "Hôm sau" / "Ngày hôm
#                          sau" (+1d), "Hôm nay" (0d), "Ngày mai"
#                          (+1d), "Tuần trước" (-1w), "Tuần sau"
#                          (+1w), "Năm ngoái" / "Năm trước" (-1y),
#                          "Năm sau" (+1y)
#   F5. Vague magnitude  — "Vài ngày sau" (~3d), "Khoảng X tuần",
#                          "Trong vòng X ngày"
#
# Parsing produces a (magnitude, unit, direction) triple which the
# resolver applies to the anchor date (and clock-time when sub-day).
# Unit conversions:
#
#   giây / phút / giờ                    → sub-day; add to anchor's
#                                          iso_time, may roll over.
#   ngày / tuần                          → exact ``timedelta`` on the
#                                          date part (iso_time
#                                          preserved if present).
#   tháng / năm                          → calendar arithmetic
#                                          (clamp Feb-29 -> Feb-28),
#                                          iso_time preserved.
#
# Vague magnitudes (F5) carry a ``magnitude_max`` so the resolver
# can stamp ``iso`` (lower bound) and ``iso_max`` (upper bound) on
# the WhenAnchor.

#: Closed unit alphabet (ASCII ``y/m/w/d/h/min/s`` codes for the
#: internal calculator) → Vietnamese surface label. The regex
#: accepts the Vietnamese label only; this map drives the resolver.
_UNIT_LABEL: dict[str, str] = {
    "giây":  "s",
    "phút":  "min",
    "giờ":   "h",
    "ngày":  "d",
    "tuần":  "w",
    "tháng": "M",
    "năm":   "Y",
}

#: Sub-day units. These add a clock-time delta to the anchor's
#: ``iso_time`` (when present); day-or-larger units leave it
#: unchanged.
_SUB_DAY_UNITS: frozenset[str] = frozenset({"giây", "phút", "giờ"})


_VN_UNIT_RE = r"(giây|phút|giờ|ngày|tuần|tháng|năm)"

#: F1 forward variants. Magnitude allows decimal-comma OCR forms
#: ("1,2 phút sau"). The "sau khi" tail is stripped before matching
#: so "5 ngày sau khi xảy ra" still matches as "5 ngày sau".
_F1_NUM_TRAILING = re.compile(
    r"^(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\s+sau\b",
)
_F1_SAU_X = re.compile(
    r"^sau(?:\s+(?:đó|khi))?\s+(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\b",
)

#: F2 backward variants. ``trước đó X N`` is the most common
#: ban-án phrasing; ``X N trước`` is also seen.
_F2_TRUOC_DO_X = re.compile(
    r"^trước(?:\s+đó)?\s+(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\b",
)
_F2_X_TRUOC = re.compile(
    r"^(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\s+trước(?!\s+đó)\b",
)
_F2_CACH_DO = re.compile(
    r"^cách(?:\s+đó)?\s+(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\b",
)

#: F3 same-day / same-time deixis. No magnitude.
_F3_RAW_FORMS: tuple[str, ...] = (
    "cùng ngày", "cùng buổi", "cùng lúc", "cùng thời điểm",
    "hôm đó", "lúc đó", "ngay sau đó", "ngay lúc đó",
)

#: F4 calendar deixis. Map of surface form → (delta, unit, direction).
#: Anchor is the *most recent* preceding absolute date, not the
#: judgment date — this matches narrative usage in ban-án (the
#: deixis is centred on the previously discussed event, not on
#: "today").
_F4_DEIXIS: dict[str, tuple[float, RelativeUnit, RelativeDirection]] = {
    "hôm nay":          (0.0, "ngày",  "same"),
    "hôm qua":          (1.0, "ngày",  "before"),
    "hôm kia":          (2.0, "ngày",  "before"),
    "hôm sau":          (1.0, "ngày",  "after"),
    "ngày hôm sau":     (1.0, "ngày",  "after"),
    "ngày mai":         (1.0, "ngày",  "after"),
    "ngày kia":         (2.0, "ngày",  "after"),
    "tuần trước":       (1.0, "tuần",  "before"),
    "tuần qua":         (1.0, "tuần",  "before"),
    "tuần sau":         (1.0, "tuần",  "after"),
    "tháng trước":      (1.0, "tháng", "before"),
    "tháng sau":        (1.0, "tháng", "after"),
    "năm ngoái":        (1.0, "năm",   "before"),
    "năm trước":        (1.0, "năm",   "before"),
    "năm sau":          (1.0, "năm",   "after"),
}

#: F5 vague magnitudes — produce a (mag_low, mag_high) pair.
_F5_VAI = re.compile(r"^(?:vài|một\s+vài|mấy)\s*" + _VN_UNIT_RE + r"\s+sau\b")
_F5_KHOANG = re.compile(
    r"^khoảng\s+(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\s+sau\b",
)
_F5_VAI_TRUOC = re.compile(r"^(?:vài|một\s+vài|mấy)\s*" + _VN_UNIT_RE + r"\s+trước\b")

#: Source-text scanner — tighter, ASCII-tolerant variants that match
#: substrings inside running prose, not just normalised candidates.
#: Each pattern's match is post-processed by :func:`_classify_match`
#: to produce a (direction, magnitude, magnitude_max, unit) tuple.
_SCAN_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Backward-relative — try first; "trước" lone form matches both
    # F2 and the "tuần trước" deixis below, so order matters.
    re.compile(
        r"\btrước\s+đó\s+(?:khoảng\s+)?(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bcách\s+đó\s+(?:khoảng\s+)?(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\b",
        flags=re.IGNORECASE,
    ),
    # Forward-relative.
    re.compile(
        r"\b(?:khoảng\s+)?(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\s+sau\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bsau\s+đó\s+(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\b",
        flags=re.IGNORECASE,
    ),
    # F2 trailing-trước (must come AFTER the trước-đó pattern above
    # so "trước đó N <unit>" is not eaten as "<unit> trước"). The
    # negative lookbehind avoids misfiring on "trước đó 3 ngày".
    re.compile(
        r"(?<!đó\s)\b(\d+(?:[.,]\d+)?)\s*" + _VN_UNIT_RE + r"\s+trước\b(?!\s+đó)",
        flags=re.IGNORECASE,
    ),
    # F4 deixis bare forms (anchored on word boundaries).
    re.compile(
        r"\b(?:ngày\s+hôm\s+sau|hôm\s+(?:nay|qua|kia|sau|đó)|"
        r"ngày\s+(?:mai|kia)|tuần\s+(?:trước|qua|sau)|"
        r"tháng\s+(?:trước|sau)|năm\s+(?:ngoái|trước|sau))\b",
        flags=re.IGNORECASE,
    ),
    # F3 same-day deixis.
    re.compile(
        r"\bcùng\s+(?:ngày|buổi|lúc|thời\s+điểm)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bngay\s+(?:sau\s+đó|lúc\s+đó)\b|\blúc\s+đó\b",
        flags=re.IGNORECASE,
    ),
    # F5 vague forward-relative.
    re.compile(
        r"\b(?:vài|mấy|một\s+vài)\s+" + _VN_UNIT_RE + r"\s+sau\b",
        flags=re.IGNORECASE,
    ),
)


def _to_float(s: str) -> float | None:
    """Parse a magnitude allowing decimal-comma (``"1,2"`` -> ``1.2``)."""
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def _add_delta(
    *,
    anchor_iso: str,
    anchor_time: str | None,
    magnitude: float,
    unit: RelativeUnit,
    direction: RelativeDirection,
) -> tuple[str | None, str | None]:
    """Compute ``anchor_iso(+anchor_time) ± (magnitude * unit)``.

    Returns a ``(new_iso, new_iso_time)`` pair. ``new_iso_time`` is
    propagated through day-or-larger arithmetic (so the time of day
    survives a day-shift) and is recomputed for sub-day arithmetic
    (which can also shift the calendar day via midnight rollover).

    Sub-day units (``giây/phút/giờ``):
        * If ``anchor_time`` is present: arithmetic is exact on the
          full datetime; both date and time can change.
        * If ``anchor_time`` is absent: the anchor is date-only;
          the delta cannot be applied with day-resolution accuracy,
          so we keep the date unchanged and return ``new_iso_time
          = None`` (the caller may treat this as a degraded-quality
          resolution; the relative-provenance fields are still
          stamped onto the anchor so renderers can show the raw
          surface form).

    Day-or-larger units (``ngày/tuần``):
        * Add an exact ``timedelta`` to the date.
        * Preserve ``anchor_time`` on the result.

    Calendar units (``tháng/năm``):
        * Add months / years with calendar arithmetic; clamp Feb-29
          → Feb-28 in non-leap target years.
        * Preserve ``anchor_time`` on the result.

    Returns ``(None, None)`` if the anchor isn't a parseable ISO
    date or the result would fall outside ``1900..2099``.
    """
    try:
        y, m, d = (int(x) for x in anchor_iso.split("-"))
        d0 = _dt.date(y, m, d)
    except (ValueError, TypeError):
        return None, None

    if direction == "same":
        return anchor_iso, anchor_time

    sign = -1 if direction == "before" else 1
    code = _UNIT_LABEL[unit]

    # Sub-day arithmetic.
    if code in {"s", "min", "h"}:
        seconds_per_unit = {"s": 1, "min": 60, "h": 3600}[code]
        total_seconds = sign * magnitude * seconds_per_unit
        if anchor_time is None:
            # Cannot apply sub-day delta with day resolution; leave
            # the date untouched and report no resolved time.
            return anchor_iso, None
        try:
            th, tm, ts = (int(x) for x in anchor_time.split(":"))
            base_dt = _dt.datetime(y, m, d, th, tm, ts)
        except (ValueError, TypeError):
            return anchor_iso, None
        try:
            new_dt = base_dt + _dt.timedelta(seconds=total_seconds)
        except OverflowError:
            return None, None
        if not (1900 <= new_dt.year <= 2099):
            return None, None
        return new_dt.date().isoformat(), _build_iso_time(
            new_dt.hour, new_dt.minute, new_dt.second,
        )

    # Day / week — exact day arithmetic; preserve clock time.
    if code in {"d", "w"}:
        seconds_per_unit = {"d": 86400, "w": 604800}[code]
        total_seconds = sign * magnitude * seconds_per_unit
        try:
            new_d = d0 + _dt.timedelta(seconds=total_seconds)
        except OverflowError:
            return None, None
        if not (1900 <= new_d.year <= 2099):
            return None, None
        return new_d.isoformat(), anchor_time

    # Month / year — calendar arithmetic; preserve clock time.
    months = 0
    years = 0
    if code == "M":
        months = sign * round(magnitude)
    elif code == "Y":
        years = sign * round(magnitude)
    else:
        return None, None

    new_year = y + years + (m - 1 + months) // 12
    new_month = (m - 1 + months) % 12 + 1
    if not (1900 <= new_year <= 2099):
        return None, None
    last_day = (
        31, 29 if (
            new_year % 4 == 0
            and (new_year % 100 != 0 or new_year % 400 == 0)
        ) else 28,
        31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    )[new_month - 1]
    new_day = min(d, last_day)
    return f"{new_year:04d}-{new_month:02d}-{new_day:02d}", anchor_time


def is_relative_form(raw: str) -> bool:
    """Return ``True`` iff ``raw`` looks like a relative expression.

    Cheap structural check used by the source-scanner / dedup
    pipeline; the actual parsing happens in
    :func:`parse_relative_to_anchor`. False positives here are
    benign — the parser will simply return ``None`` and the surface
    is treated as an absolute date.
    """
    return _classify_normalised(_normalise(raw)) is not None


def _classify_normalised(
    norm: str,
) -> tuple[RelativeDirection, float, float | None, RelativeUnit] | None:
    """Classify a *normalised* surface form into ``(dir, mag, mag_max, unit)``.

    Returns ``None`` if the input is not a recognised relative
    surface form. ``mag_max`` is non-None only for F5 vague
    magnitudes (``"vài ngày sau"`` → ``mag=1, mag_max=5, unit=ngày``);
    otherwise the resolver produces a single-point anchor.
    """
    # F3 — closed-set same-time deixis (no magnitude).
    for tok in _F3_RAW_FORMS:
        if norm == tok:
            return ("same", 0.0, None, "ngày")

    # F4 — closed-set calendar deixis.
    if norm in _F4_DEIXIS:
        mag, unit, direction = _F4_DEIXIS[norm]
        return (direction, mag, None, unit)

    # F1 — forward delta with magnitude.
    for pat in (_F1_NUM_TRAILING, _F1_SAU_X):
        m = pat.match(norm)
        if m is None:
            continue
        mag = _to_float(m.group(1))
        if mag is None:
            return None
        return ("after", mag, None, m.group(2))  # type: ignore[return-value]

    # F2 — backward delta with magnitude.
    for pat in (_F2_TRUOC_DO_X, _F2_CACH_DO, _F2_X_TRUOC):
        m = pat.match(norm)
        if m is None:
            continue
        mag = _to_float(m.group(1))
        if mag is None:
            return None
        return ("before", mag, None, m.group(2))  # type: ignore[return-value]

    # F5 — vague forward / backward magnitude.
    m = _F5_VAI.match(norm)
    if m is not None:
        return ("after", 1.0, 5.0, m.group(1))  # type: ignore[return-value]
    m = _F5_KHOANG.match(norm)
    if m is not None:
        mag = _to_float(m.group(1))
        if mag is None:
            return None
        # ±25% spread for "khoảng" magnitudes.
        return ("after", mag, mag * 1.25, m.group(2))  # type: ignore[return-value]
    m = _F5_VAI_TRUOC.match(norm)
    if m is not None:
        return ("before", 1.0, 5.0, m.group(1))  # type: ignore[return-value]

    return None


def parse_relative_to_anchor(
    raw: str,
    *,
    anchor: WhenAnchor | None,
    anchor_event_id: str | None = None,
    page: int | None = None,
) -> WhenAnchor | None:
    """Resolve ``raw`` against ``anchor`` and return a :class:`WhenAnchor`.

    Returns ``None`` if the surface form is not a relative
    expression at all (caller should then try the absolute parser
    or treat the entity as ambient).

    If ``raw`` *is* a relative expression but ``anchor`` is ``None``
    or the anchor's ``iso`` is unparseable, the function returns a
    :class:`WhenAnchor` with ``iso=None`` and ``is_relative=True`` —
    the relative-provenance fields are still filled so downstream
    consumers know what was extracted (and can render it as an
    unresolved tooltip).

    When the anchor is fully dated and the unit is sub-day, the
    resolved anchor carries both the (possibly-shifted) ``iso``
    date and a fresh ``iso_time``; when the unit is day-or-larger,
    the anchor's ``iso_time`` is propagated through unchanged so a
    "5 ngày sau" against a 22:30 anchor still lands at 22:30 on the
    target date.
    """
    norm = _normalise(raw)
    cls = _classify_normalised(norm)
    if cls is None:
        return None
    direction, mag, mag_max, unit = cls

    iso_resolved: str | None = None
    iso_time_resolved: str | None = None
    iso_max_resolved: str | None = None

    if anchor is not None and anchor.iso is not None:
        iso_resolved, iso_time_resolved = _add_delta(
            anchor_iso=anchor.iso,
            anchor_time=anchor.iso_time,
            magnitude=mag,
            unit=unit,
            direction=direction,
        )
        if mag_max is not None:
            iso_max_resolved, _ = _add_delta(
                anchor_iso=anchor.iso,
                anchor_time=anchor.iso_time,
                magnitude=mag_max,
                unit=unit,
                direction=direction,
            )

    return _whenanchor(
        iso=iso_resolved,
        iso_partial=None,
        iso_time=iso_time_resolved,
        raw=raw,
        page=page,
        is_relative=True,
        anchor_event_id=anchor_event_id,
        magnitude=mag,
        unit=unit,
        direction=direction,
        iso_max=iso_max_resolved,
    )


def find_relative_expressions(
    source_text: str,
) -> list[tuple[int, int, str]]:
    """Scan ``source_text`` for relative temporal expressions.

    Returns a list of ``(start, end, raw)`` triples in left-to-right
    document order, where ``raw`` is the verbatim source slice (so
    callers can re-display it untouched). Overlapping matches are
    suppressed: the longest match wins, ties broken by left-most
    position.

    The scanner is NFC-tolerant — it operates on the input as-is
    but the patterns are case-insensitive and accept the lowercase
    Vietnamese unit lemmas. Callers should NFC-normalise the
    source themselves if their offsets need to align with another
    NFC consumer.
    """
    spans: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for pat in _SCAN_PATTERNS:
        for m in pat.finditer(source_text):
            span = (m.start(), m.end())
            if span in seen:
                continue
            seen.add(span)
            spans.append((m.start(), m.end(), m.group(0)))

    # Suppress shorter matches contained within a longer span.
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    out: list[tuple[int, int, str]] = []
    cursor_end = -1
    for start, end, raw in spans:
        if start < cursor_end:
            continue
        out.append((start, end, raw))
        cursor_end = end
    out.sort(key=lambda x: x[0])
    return out


__all__ = [
    "find_relative_expressions",
    "is_relative_form",
    "parse_date_to_anchor",
    "parse_relative_to_anchor",
]
