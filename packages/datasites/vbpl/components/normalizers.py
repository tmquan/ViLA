"""Pure text-cleanup helpers for vbpl (extracted from :mod:`parser`).

Every function here is I/O-free and deterministic: it maps a raw
string (a scraped ``soHieu`` / ``tieuDe`` / markdown blob) to its
canonical form. They are split out of :mod:`packages.datasites.vbpl.components.parser`
so the parsing surface stays lean and these normalizers can be unit-
tested and reused (see :mod:`packages.datasites.vbpl.normalizers`,
which wraps them as declarative NeMo-Curator normalizer stages).

``parser`` re-exports the public names below so existing
``from ...parser import clean_title`` call-sites keep working.
"""

from __future__ import annotations

import html
import re
import unicodedata


#: Recognised administrative-defect suffixes that vbpl.vn editors
#: append to ``soHieu`` when they flag a record as corrupt
#: (e.g. ``"1333/TP-KHTC Lỗi"`` where the body is just CSS junk).
#: The leading whitespace before the marker is mandatory in the
#: pattern -- without it we would chop "Lỗi" out of legitimate
#: ``doc_number`` values that happen to end in "Lỗi" (unlikely but
#: future-proofing against a Vietnamese acronym collision).
_DEFECT_SUFFIX_RE = re.compile(r"\s+(?:Lỗi|lỗi|LỖI|ERROR|error)\s*$")

#: Leading "force-text" sentinels (Excel/CSV apostrophe, stray
#: double-quote) that the source CMS pastes into ``soHieu`` to keep
#: spreadsheet imports from coercing the number to a date or float.
#: ``''`` (double-apostrophe) was observed on a handful of TT-BCT
#: rows, so we strip greedily.
_LEADING_QUOTES_RE = re.compile(r"^[\s'\"`\u2018\u2019\u201C\u201D]+")

#: Whitespace tucked around ``/`` or ``-`` separators (e.g.
#: ``"04/2007/TT- NHNN"``, ``"07 /2024/QĐ-UBND"``). We collapse
#: them so the canonical form ``"04/2007/TT-NHNN"`` is what every
#: downstream consumer sees.
_SEPARATOR_SPACE_RE = re.compile(r"\s*([/\-])\s*")

#: Leading labels accidentally pasted into ``soHieu`` from a copy-
#: paste of the document header (e.g. ``"Số: 06/2023/QĐ-UBND"``,
#: ``"No.: 12/2024/TT-BTC"``). Strips the label + colon + ws.
_LEADING_LABEL_RE = re.compile(
    r"^(?:Số|So|S[oô]|No\.?|Number|N°)\s*[:\.]?\s*",
    re.IGNORECASE,
)

#: Trailing punctuation that leaked from the surrounding sentence
#: (period, semicolon, comma). Single ``-`` is kept because some
#: legacy IDs legitimately end with one (e.g. ``"27-NQ/HĐND-"`` is
#: not observed but we stay conservative on dashes/slashes).
_TRAILING_PUNCT_RE = re.compile(r"[.,;]+\s*$")


#: Doc-number token shape. The canonical doc_number after normalisation
#: is ``<digits>[<letter>][/-]<word-segment>+``. ``\w`` is Unicode-
#: aware in Python 3 and includes Vietnamese letters (``Đ``, ``Đ``
#: U+0110, accented forms) so ``142/2009/QĐ-TTg`` validates cleanly.
_DOCNUM_TOKEN_RE = re.compile(r"^\d+[A-Za-z]?[/\-][\w/\-]+$", re.UNICODE)

#: Doc-number head matcher used by :func:`_strip_trailing_noise`.
#: Greedy on the trailing ``[\w/\-]`` class so we capture as much of
#: the canonical token as possible before the noise tail starts.
_DOCNUM_HEAD_RE = re.compile(r"^(\d+[A-Za-z]?[/\-][\w/\-]*)", re.UNICODE)

#: Recognise the legitimate "Không số" (= "no number") sentinel
#: vbpl uses for 1957-era cultural resolutions with no assigned
#: identifier. Diacritic-tolerant: matches Vietnamese precomposed
#: ``ố`` (U+1ED1) as well as the ASCII fallbacks. Casefold so
#: ``KHÔNG SỐ`` and ``khong so`` round-trip.
_KHONG_SO_VARIANTS: frozenset[str] = frozenset({
    "không số", "khong so", "khong số", "không so",
})


def _is_khong_so(chunk: str) -> bool:
    """Case-insensitive Vietnamese-diacritic-tolerant ``Không số`` match."""
    return chunk.strip().casefold() in {
        v.casefold() for v in _KHONG_SO_VARIANTS
    }

#: Legal-type words that vbpl editors occasionally paste at the head
#: of ``soHieu`` (e.g. ``"Nghị quyết số: 528/2018/UBTVQH14"``,
#: ``"Thông tư 04/2018/TT-BKHĐT"``, ``"Văn bản hợp nhất / Thông tư
#: 18/VBHN-BTC"``). The pattern matches one or two legal-type words
#: (the second after a ``" / "`` separator for the VBHN case) plus
#: optional ``" liên tịch"`` and optional ``"số[:.]"`` label, ending
#: at the whitespace that precedes the actual number. Case-sensitive
#: because vbpl is consistent about capitalisation in source data.
_SOHIEU_LEGAL_TYPE_PREFIX_RE = re.compile(
    r"""^\s*
        (?:
            Văn\s+bản\s+hợp\s+nhất
            |
            Nghị\s+quyết\s+liên\s+tịch
            |
            Thông\s+tư\s+liên\s+tịch
            |
            Thông\s+tư\s+liên\s+bộ
            |
            Nghị\s+định\s+thư
            |
            Bộ\s+luật
            |
            Pháp\s+lệnh
            |
            Hiến\s+pháp
            |
            Sắc\s+lệnh
            |
            Sắc\s+luật
            |
            Pháp\s+điển
            |
            Hiệp\s+định
            |
            Thỏa\s+thuận
            |
            Công\s+văn
            |
            Thông\s+báo
            |
            Báo\s+cáo
            |
            Quyết\s+định
            |
            Nghị\s+quyết
            |
            Nghị\s+định
            |
            Thông\s+tư
            |
            Chỉ\s+thị
            |
            Văn\s+bản
            |
            Luật
            |
            Lệnh
        )
        (?:\s*/\s*
            (?:
                Thông\s+tư|Nghị\s+định|Quyết\s+định|Thông\s+báo|
                Pháp\s+lệnh|Nghị\s+quyết|Luật|Văn\s+bản
            )
        )?
        (?:\s+liên\s+tịch)?
        \s+
        (?:s\w\s*[:.]?\s*)?
    """,
    re.VERBOSE | re.IGNORECASE | re.UNICODE,
)


def normalise_doc_number(raw: str | None) -> str | None:
    """Clean up the cosmetic artifacts present in vbpl ``soHieu`` values.

    The vbpl.vn CMS exposes the document number ("so hieu") with a
    handful of recurring data-entry defects:

    * **Leading apostrophes** (e.g. ``"'22/2025/QĐ-UBND"``) -- a
      side-effect of Excel/CSV "force-text" markers that leaked
      from the editing workflow into the published payload.
    * **Internal whitespace around separators** (e.g.
      ``"07 /2024/QĐ-UBND"`` or ``"04/2007/TT- NHNN"``) -- the
      source itself ships these.
    * **Administrative defect annotations** (e.g.
      ``"1333/TP-KHTC Lỗi"``) -- a trailing flag that vbpl
      editors append when the underlying page is broken
      (typically the body is just CSS/HTML chrome with no
      content). The flag is metadata, not part of the legal
      identifier, so we strip it from ``doc_number``.

    Returns ``None`` for empty / whitespace-only input, the
    sentinel string ``"Không số"`` (Vietnamese for "no number")
    untouched (legitimately nameless documents like 1957-era
    Quốc hội cultural resolutions use it), or the cleaned form.

    >>> normalise_doc_number("'22/2025/QĐ-UBND")
    '22/2025/QĐ-UBND'
    >>> normalise_doc_number("04/2007/TT- NHNN")
    '04/2007/TT-NHNN'
    >>> normalise_doc_number("1333/TP-KHTC Lỗi")
    '1333/TP-KHTC'
    >>> normalise_doc_number("Số: 06 /2023/QĐ-UBND")
    '06/2023/QĐ-UBND'
    >>> normalise_doc_number("03/2020/QĐ-UBND.")
    '03/2020/QĐ-UBND'
    >>> normalise_doc_number("Không số")
    'Không số'
    >>> normalise_doc_number("  ") is None
    True
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Order matters: strip leading quotes first so the label regex
    # can see "Số:" without a preceding apostrophe; strip the
    # defect suffix before the trailing-punct pass so we don't
    # leave a stray "." behind from "X. Lỗi" cases.
    s = _LEADING_QUOTES_RE.sub("", s)
    s = _LEADING_LABEL_RE.sub("", s)
    s = _DEFECT_SUFFIX_RE.sub("", s)
    s = _TRAILING_PUNCT_RE.sub("", s)
    # Only collapse whitespace around separators if the value
    # already contains one; this avoids touching legitimate
    # legacy ids like "Không số" or "191" (an SL-era pre-decree id).
    if any(sep in s for sep in ("/", "-")):
        s = _SEPARATOR_SPACE_RE.sub(r"\1", s)
    # Final whitespace squeeze (defensive -- some rows have a
    # tab between the prefix and "TT-LB" etc) plus trailing-quote
    # strip (a small set of titles leak a closing quote into doc_number
    # when the source title was a quoted phrase).
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[\"'`\u2018\u2019\u201C\u201D]+$", "", s).strip()
    return s or None


def _strip_doc_number_legal_prefix(s: str) -> str:
    """Iteratively peel a leading ``<LegalType>[/<LegalType>][số:] `` prefix.

    Convergent: re-applies the regex until the head no longer
    matches so chained prefixes (the rare ``"Văn bản hợp nhất /
    Thông tư số 18/VBHN-BTC"`` shape) collapse to the bare number
    even when the inner regex only catches one layer per pass.
    """
    prev = None
    while prev != s:
        prev = s
        m = _SOHIEU_LEGAL_TYPE_PREFIX_RE.match(s)
        if m is None:
            break
        candidate = s[m.end():].strip()
        if not candidate:
            break
        s = candidate
    return s


def _strip_doc_number_trailing_noise(s: str) -> str:
    """Drop everything after the FIRST whitespace inside a doc_number chunk.

    The vbpl CMS pastes a wide variety of editorial trailers onto
    individual ``soHieu`` cells once the doc-number itself is
    complete: ``" ngày 18/5/2007"``, ``" (1)"``, ``" 2022"``,
    ``" VĂN BẢN TRÙNG"``, ``" & XH"``, etc. None of them belong to
    the canonical identifier. The strategy is conservative: anchor
    on the doc-number-shaped head at the start of the string and
    drop everything after the first whitespace, **unless** what
    follows is itself another doc-number token (in which case the
    multi-value splitter in :func:`normalise_doc_number_list` should
    have caught it first; we leave the string untouched here to
    avoid double-handling).

    Returns the input unchanged when the head doesn't match the
    doc-number shape (the downstream validator will reject the
    chunk in that case).
    """
    if not s:
        return s
    m = _DOCNUM_HEAD_RE.match(s)
    if m is None:
        return s
    head = m.group(1)
    rest = s[m.end():].lstrip()
    if not rest:
        return head
    # If the rest LOOKS like another doc-num token, leave the
    # whole string alone -- multi-value splitting is the splitter's
    # job, not the trailing-noise stripper's.
    if _DOCNUM_HEAD_RE.match(rest):
        return s
    return head


#: Multi-value separator regex. Matches ` và ` / ` VÀ ` / ` Và ` /
#: ` và` (any whitespace around) plus ASCII ``,`` and ``;``. We split
#: on these only when *all* resulting chunks survive the doc-number
#: predicate -- agency suffixes that happen to contain commas (e.g.
#: ``"TTLT-BCA-BTC, BTC"``) thus stay whole because the trailing
#: ``BTC`` chunk doesn't look like a doc-number.
_SOHIEU_SPLIT_RE = re.compile(
    r"\s+(?:và|VÀ|Và|v[aà])\s+|\s*[,;]\s*",
    re.UNICODE,
)


def _looks_like_docnum_after_prefix_strip(chunk: str) -> bool:
    """Conservative predicate: does ``chunk`` plausibly contain a doc-num?

    Used by the multi-value splitter to decide whether to commit to a
    split. We strip any leading legal-type prefix and then check that
    the remainder has at least one digit and at least one ``/`` or
    ``-``. This rules out chunks like ``"BTC"`` or ``"UBTƯMTTQVN"``
    that happen to be agency suffixes wrapped in commas.
    """
    if not chunk:
        return False
    c = _strip_doc_number_legal_prefix(chunk.strip())
    if not c:
        return False
    if _is_khong_so(c):
        return True
    return bool(re.search(r"\d", c) and re.search(r"[/\-]", c))


def normalise_doc_number_list(raw: str | None) -> list[str]:
    """Split + clean a ``soHieu`` cell into a list of canonical doc-nums.

    The vbpl ``soHieu`` field is *usually* a single document number
    (``"43/2026/NĐ-CP"``) but a small minority of rows pack several
    identifiers into one cell separated by Vietnamese ``" và "``
    ("and") or ASCII commas (``"60/CP, 61/CP"``). On top of that the
    cell sometimes carries a leading legal-type word and trailing
    editorial annotations -- both stripped here so the canonical
    short form is what ships in the parquet.

    Pipeline:

    1. Run :func:`normalise_doc_number` over the whole string to apply
       the existing baseline cleanup (quotes, ``Số:`` label, defect
       suffix, trailing punctuation, separator whitespace).
    2. Strip any leading punctuation / comma cruft (a small number
       of rows ship ``",31/2019/QĐ-UBND"`` etc.).
    3. Tentatively split on ``" và "`` / ``,`` / ``;`` / ``" Và "``.
       The split is **only committed** when every resulting chunk
       individually looks like a doc-num (contains both a digit and
       a ``/`` or ``-`` after a possible legal-type prefix). This
       prevents agency suffixes such as ``"TTLT-BCA-BTC, BTC"`` from
       getting split into garbage.
    4. For each chunk: strip the leading legal-type prefix (item 1),
       strip the trailing editorial noise (item 2), strip wrapping
       whitespace / quotes / commas.
    5. Validate against ``^\\d+[A-Za-z]?[/-][\\w/-]+$``; tokens that
       don't match are dropped (defensive -- they become ``null``
       at the parquet level, not garbage data).
    6. Keep the legitimate ``"Không số"`` sentinel (canonicalised
       to that exact spelling) when it appears.
    7. Deduplicate while preserving source order.

    Returns ``[]`` for empty / whitespace / invalid input so the
    projection can translate it to ``null`` for the parquet column.

    >>> normalise_doc_number_list('Nghị quyết số: 528/2018/UBTVQH14')
    ['528/2018/UBTVQH14']
    >>> normalise_doc_number_list('Thông tư 04/2018/TT-BKHĐT')
    ['04/2018/TT-BKHĐT']
    >>> normalise_doc_number_list('Văn bản hợp nhất / Thông tư 18/VBHN-BTC')
    ['18/VBHN-BTC']
    >>> normalise_doc_number_list('109/2005/QĐ-BCA (A11)')
    ['109/2005/QĐ-BCA']
    >>> normalise_doc_number_list('22/2023/NĐ-CP (1)')
    ['22/2023/NĐ-CP']
    >>> normalise_doc_number_list('05/2009/NĐ-CP VĂN BẢN TRÙNG')
    ['05/2009/NĐ-CP']
    >>> normalise_doc_number_list('49/2007/TTLT-BTC-BGD ngày 18/5/2007')
    ['49/2007/TTLT-BTC-BGD']
    >>> normalise_doc_number_list('01/VBHN-BCT 2022')
    ['01/VBHN-BCT']
    >>> normalise_doc_number_list('142/2009/QĐ-TTg và 49/2012/QĐ-TTg')
    ['142/2009/QĐ-TTg', '49/2012/QĐ-TTg']
    >>> normalise_doc_number_list('60/CP, 61/CP')
    ['60/CP', '61/CP']
    >>> normalise_doc_number_list('03/2004/TTLT-BCA-BTC-BNV-BLĐTB & XH')
    ['03/2004/TTLT-BCA-BTC-BNV-BLĐTB']
    >>> normalise_doc_number_list(',31/2019/QĐ-UBND')
    ['31/2019/QĐ-UBND']
    >>> normalise_doc_number_list('Không số')
    ['Không số']
    >>> normalise_doc_number_list(None)
    []
    >>> normalise_doc_number_list('   ')
    []
    """
    if raw is None:
        return []
    base = normalise_doc_number(raw)
    if base is None:
        return []
    # Strip leading/trailing comma/semicolon/dot cruft that
    # ``normalise_doc_number`` doesn't touch (e.g. ``",31/2019/QĐ-UBND"``).
    base = re.sub(r"^[\s,;.]+|[\s,;.]+$", "", base).strip()
    if not base:
        return []

    # Tentatively split and verify all chunks look like doc-nums
    # (digit + separator) before committing to the split. Without
    # the verification, agency-suffix commas like
    # ``"TTLT-BCA-BTC, BTC"`` would shred the only valid id.
    raw_chunks = _SOHIEU_SPLIT_RE.split(base)
    if len(raw_chunks) > 1 and all(
        _looks_like_docnum_after_prefix_strip(c) for c in raw_chunks
    ):
        chunks = raw_chunks
    else:
        chunks = [base]

    out: list[str] = []
    for chunk in chunks:
        c = chunk.strip().strip("\"'`\u2018\u2019\u201C\u201D")
        if not c:
            continue
        c = _strip_doc_number_legal_prefix(c)
        c = _strip_doc_number_trailing_noise(c)
        c = c.strip().strip("\"'`\u2018\u2019\u201C\u201D")
        if not c:
            continue
        if _is_khong_so(c):
            cleaned = "Không số"
        elif _DOCNUM_TOKEN_RE.match(c):
            cleaned = c
        else:
            continue
        if cleaned not in out:
            out.append(cleaned)
    return out


#: Unicode codepoints that are formatting cruft rather than glyphs:
#: BOM, zero-width space / non-joiner / joiner, left-to-right mark,
#: word-joiner. Strip on sight; they never carry meaning in legal text.
_FORMAT_CHARS_RE = re.compile(
    r"[\ufeff\u200b\u200c\u200d\u200e\u200f\u2060]"
)

#: Whitespace classes the source CMS leaks: literal tab/newline/CR,
#: non-breaking space (``\u00a0``), thin space (``\u2009``), figure
#: space (``\u2007``) and friends. Normalised to a single ASCII
#: space so downstream tokenisers don't trip on width variants.
_UNICODE_WS_RE = re.compile(r"[\s\u00a0\u2000-\u200a\u202f\u205f\u3000]+")

#: Smart double quotes ("curly quotes") that creep in via Word-to-
#: HTML paste -- visually identical to ``"`` but encoded differently
#: (U+201C, U+201D, U+2033). Mapped to ASCII ``"`` so the same
#: stripping logic catches them, or removed entirely depending on
#: the column's normalisation policy.
_SMART_DOUBLE_QUOTES_RE = re.compile(r"[\u201C\u201D\u201F\u2033]")

#: Doubled "số số" / "Số Số" / mixed-case bigrams that appear when
#: the source title field accidentally appends an upstream
#: ``Số: NNN`` prefix to a template that already starts with the
#: word "số" (e.g. ``"Nghị quyết số Số: 33/2020..."`` -> ``"Nghị
#: quyết số 33/2020..."``). The capture group anchors a *required*
#: trailing whitespace+colon-or-period so the replacement preserves
#: one space before the number that follows (``số 33/...`` not
#: ``số33/...``).
_DOUBLED_SO_RE = re.compile(
    r"\bsố\s+(?:số|Số|SỐ)\b\s*[:.]?\s*",
    re.IGNORECASE,
)

#: Trailing sentence punctuation that the vbpl CMS auto-appends to
#: titles and to ``legal_area`` labels copy-pasted from inline lists
#: (e.g. ``"Viễn thông và Internet;"``). The full stop is purely
#: presentational; the canonical entry doesn't include it.
_TRAILING_SENTENCE_PUNCT_RE = re.compile(r"[.,;]+\s*$")

#: Title quote-character policy. Two separate classes because
#: doubles and singles need different handling:
#:
#: * **Doubles** (``"``, ``"``, ``"``, ``‟``, ``″``, ``❝``, ``❞``)
#:   — stripped unconditionally anywhere in the title. They never
#:   appear word-internally in Vietnamese; every occurrence is a
#:   decorative quotation mark the CMS injected.
#: * **Singles** (``'``, ``'``, ``'``, ``❛``, ``❜``) — stripped
#:   **only** at the title boundary OR when they show up in a run
#:   of two or more (the corpus uses ``''``, ``''``, ``''`` etc.
#:   as a poor-man's double quote). Word-internal single quotes
#:   are deliberately preserved so legitimate Vietnamese / loan-
#:   word names survive intact: ``Đắk R'lấp``, ``M'nông``,
#:   ``M'Drắk``, ``H'Mông``, ``D'Ran``, ``Ea T'ling``,
#:   ``Đạ M'ri``, ``Côte d'Ivoire``, … (~770 occurrences in the
#:   title column).
_DOUBLE_QUOTE_CHARS = "\"\u201C\u201D\u201F\u2033\u275D\u275E"
_SINGLE_QUOTE_CHARS = "'\u2018\u2019\u275B\u275C"
_ALL_QUOTE_CHARS = _DOUBLE_QUOTE_CHARS + _SINGLE_QUOTE_CHARS

_DOUBLE_QUOTES_RE = re.compile(rf"[{_DOUBLE_QUOTE_CHARS}]")
_LEADING_SINGLE_QUOTE_RE = re.compile(rf"^\s*[{_SINGLE_QUOTE_CHARS}]+\s*")
_TRAILING_SINGLE_QUOTE_RE = re.compile(rf"\s*[{_SINGLE_QUOTE_CHARS}]+\s*$")
_QUOTE_RUN_RE = re.compile(rf"[{re.escape(_ALL_QUOTE_CHARS)}]{{2,}}")


def _strip_decorative_quotes(s: str) -> str:
    """Strip every decorative quote position in one canonical pass.

    Applied twice in the title chain: once by :func:`normalise_title`
    on the raw title (handles quotes in the source string) and once
    as the final step of :func:`clean_title` (handles quotes that
    were *exposed* by the intermediate legal-type-prefix /
    cross-reference strippers — e.g. ``Quyết định '145/2002/QĐ-UB
    Về việc...`` -> the prefix strip leaves a stray leading ``'``
    that the first quote pass couldn't see).
    """
    s = _QUOTE_RUN_RE.sub(" ", s)
    s = _LEADING_SINGLE_QUOTE_RE.sub("", s)
    s = _TRAILING_SINGLE_QUOTE_RE.sub("", s)
    s = _DOUBLE_QUOTES_RE.sub("", s)
    return _UNICODE_WS_RE.sub(" ", s).strip()


def normalise_text(raw: str | None) -> str | None:
    """Baseline text cleanup applied to every textual column.

    Handles the universal CMS-export defects that show up in *every*
    string field on vbpl.vn regardless of semantics:

    * Decode HTML entities (``&amp;`` -> ``&``, ``&#x...;`` etc.).
    * Strip BOM, zero-width spaces, and bidi formatting marks.
    * Collapse any run of unicode whitespace (incl. NBSP, thin
      space, ideographic space, tab, newline, CR) to one ASCII
      space.
    * Apply NFC normalisation so combining marks compose to the
      precomposed form Vietnamese readers actually type.
    * Trim leading / trailing whitespace.

    Returns ``None`` for empty / whitespace-only input. The output
    is always a single-line string with no leading or trailing
    whitespace and at most single internal ASCII spaces.

    >>> normalise_text("  Quyết\\u00a0định  số\\t01  ")
    'Quyết định số 01'
    >>> normalise_text("Sở Giáo dục &amp; Đào tạo")
    'Sở Giáo dục & Đào tạo'
    >>> normalise_text("\\ufeffhello\\u200bworld")
    'helloworld'
    >>> normalise_text("") is None
    True
    """
    if raw is None:
        return None
    s = str(raw)
    if not s.strip():
        return None
    s = html.unescape(s)
    s = _FORMAT_CHARS_RE.sub("", s)
    s = unicodedata.normalize("NFC", s)
    s = _UNICODE_WS_RE.sub(" ", s).strip()
    return s or None


def normalise_title(raw: str | None) -> str | None:
    """Strip the title-specific defects on top of :func:`normalise_text`.

    Three extra passes beyond the baseline:

    * **Doubled "số số" prefix** -- 200+ rows have the bigram from
      the CMS pasting the ``Số: NN`` header on top of a template
      string that already starts with "số".
    * **Trailing sentence punctuation** -- 7K+ titles end with a
      stray ``.`` or ``,`` from the source's sentence-style
      formatting. The legal title itself is a noun phrase; the
      terminal punctuation is purely presentational.
    * **Decorative quote stripping** -- the source uses
      ``"..."`` / ``'...'`` / ``''...''`` (sometimes mixing
      straight, smart, and ASCII-apostrophe variants) to wrap
      inline phrases. We strip leading + trailing quote characters
      AND any run of 2 or more quote characters anywhere in the
      title, but we deliberately preserve **word-internal single
      quotes** so legitimate Vietnamese / loan-word names
      (``Đắk R'lấp``, ``M'nông``, ``D'Ran``, ``Côte d'Ivoire``,
      …) survive intact.

    >>> normalise_title('Nghị quyết số số 33/2020/NQ-HĐND .')
    'Nghị quyết số 33/2020/NQ-HĐND'
    >>> normalise_title('Quyết định "Bà mẹ Việt Nam"')
    'Quyết định Bà mẹ Việt Nam'
    >>> normalise_title("'Về việc tổ chức thực hiện của Chính phủ")
    'Về việc tổ chức thực hiện của Chính phủ'
    >>> normalise_title("Cơ chế một cửa'' đối với UBND huyện Tiên Du")
    'Cơ chế một cửa đối với UBND huyện Tiên Du'
    >>> normalise_title("Về đề án \u2018\u2018Tăng cường công tác\u2019\u2019 trên địa bàn")
    'Về đề án Tăng cường công tác trên địa bàn'
    >>> normalise_title("Đặt tên đường thị trấn Ea T'ling, huyện Cư Jut")
    "Đặt tên đường thị trấn Ea T'ling, huyện Cư Jut"
    >>> normalise_title('  ') is None
    True
    """
    s = normalise_text(raw)
    if s is None:
        return None
    s = _DOUBLED_SO_RE.sub("số ", s)
    s = _strip_decorative_quotes(s)
    s = _TRAILING_SENTENCE_PUNCT_RE.sub("", s)
    # Final pass: collapsing the quote stripping can leave doubled
    # spaces (e.g. ``a "x" b`` -> ``a  b``). One more squeeze.
    s = _UNICODE_WS_RE.sub(" ", s).strip()
    return s or None


#: Vietnamese legal-type-name aliases that appear at the head of
#: titles but don't exactly match the canonical ``legal_type``
#: field. ``"Bản dịch văn bản"`` (Translation) is the prime
#: offender -- the title is the original decree's reference so it
#: starts with ``"Nghị định …"`` / ``"Thông tư …"`` / etc., not
#: with ``"Bản dịch văn bản …"``. We try every alias when the
#: per-record ``legal_type`` doesn't match.
_TITLE_PREFIX_ALIASES: tuple[str, ...] = (
    "Nghị định thư",
    "Thông tư liên tịch",
    "Thông tư liên bộ",
    "Nghị quyết liên tịch",
    "Văn bản hợp nhất",
    "Văn bản hành chính liên quan",
    "Văn bản liên quan",
    "Văn bản khác",
    "Bản ghi nhớ",
    "Bản dịch văn bản",
    "Quyết định",
    "Nghị quyết",
    "Nghị định",
    "Thông tư",
    "Chỉ thị",
    "Sắc lệnh",
    "Sắc luật",
    "Pháp lệnh",
    "Hiến pháp",
    "Bộ luật",
    "Luật",
    "Lệnh",
    "Công văn",
    "Thông báo",
    "Hiệp định",
    "Thỏa thuận",
    "Chương trình",
)

#: Minimum length of the stripped remainder for the strip to be
#: applied. Vietnamese legal subjects can be very short
#: (``"Thú y"``, ``"Dân số"``, ``"Cảnh vệ"`` are all real titles),
#: so we only fence against the truly degenerate case where the
#: title is essentially just ``"<legal_type> số <number>"`` with
#: no subject at all (Bản dịch translations, certain pre-1990
#: pháp lệnh entries). ``None`` returns from the inner stripper
#: already cover the empty-remainder case so this floor only
#: matters when a single residual word slips through.
_MIN_STRIPPED_TITLE_LEN = 3


def _coerce_doc_number_list(
    doc_number: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    """Accept old (string) or new (list) ``doc_number`` shape; return a list.

    The legacy callers passed a single string ``"143/QĐ-KHTC"`` to
    :func:`strip_redundant_title_prefix`; the new pipeline ships
    ``["143/QĐ-KHTC"]`` (or ``["A/X", "B/Y"]`` for the multi-value
    rows). Both call shapes round-trip through this helper without
    the call sites needing two different signatures.
    """
    if doc_number is None:
        return []
    if isinstance(doc_number, str):
        return [doc_number] if doc_number.strip() else []
    return [x for x in doc_number if isinstance(x, str) and x.strip()]


def strip_redundant_title_prefix(
    title: str | None,
    legal_type: str | None,
    doc_number: str | list[str] | tuple[str, ...] | None,
) -> str | None:
    """Strip the redundant ``"{legal_type} số {doc_number}"`` head from a title.

    Vietnamese legal-document titles on vbpl.vn follow a consistent
    template::

        <Legal type> số <Number> <Action verb> <Object>

    e.g. ``"Quyết định số 143/QĐ-KHTC Ban hành Quy chế quản lý ngân
    sách ngành Tư pháp"``. The leading ``"<Legal type> số <Number>"``
    block is fully redundant because both pieces ship in their own
    columns (``legal_type`` + ``doc_number``); keeping it in the title
    pads every embedding token by ~10-20 boilerplate tokens that
    push the actual subject matter toward the truncation tail.

    Strategy:

    1. **Anchor on ``doc_number``** when the cleaned value appears
       literally in the title (96% of rows). Match
       ``^{legal_type}\\s+(số\\s+)?{doc_number}\\s+`` and strip.
       When ``doc_number`` is a list, the first element is used as
       the anchor (it's the canonical / primary doc-number).
    2. **Token-based fallback** for the 4% where the cleaned
       doc_number doesn't match the title verbatim (raw title might
       still carry whitespace around ``/`` or ``-`` that the
       doc_number column has had collapsed). Match the legal-type
       prefix + ``"số"`` + a permissive doc_number-shaped token.
    3. **Alias fallback** for ``"Bản dịch văn bản"`` and friends
       whose title starts with a *different* legal-type name (the
       original decree's name, not the translation's).

    .. note::

       Earlier versions of this function also peeled a leading
       ``"Lỗi "`` token, on the assumption that the vbpl CMS used
       it as an editorial marker for broken records. That heuristic
       was retired in May 2026 -- corpus audit showed every
       ``Lỗi`` / ``lỗi`` / ``loi`` prefix in the source titles is
       the Vietnamese **noun meaning "fault / error"** used as a
       legitimate subject ("Lỗi Ban hành Quy chế hoạt động của hội
       đồng tư vấn mua sắm tài sản" — "Faults in issuing the
       operational regulations of the asset procurement advisory
       council"), not a CMS marker. Stripping it silently
       corrupted those titles. ``clean_title`` and the HF projection
       therefore preserve ``Lỗi`` wherever it appears.

    Returns ``None`` for empty input; returns the original title
    unchanged when no prefix matches *or* when stripping would
    leave a remainder shorter than ``_MIN_STRIPPED_TITLE_LEN``
    chars (defensive against turning ``"Nghị định 119/2005/NĐ-CP"``
    into the empty string).

    >>> strip_redundant_title_prefix(
    ...     'Quyết định số 143/QĐ-KHTC Ban hành Quy chế', 'Quyết định', '143/QĐ-KHTC')
    'Ban hành Quy chế'
    >>> strip_redundant_title_prefix(
    ...     'Nghị quyết số Không số Về công tác văn hoá xã hội',
    ...     'Nghị quyết', 'Không số')
    'Về công tác văn hoá xã hội'
    >>> strip_redundant_title_prefix(
    ...     'Quyết định số 07 /2024/QĐ-UBND Bãi bỏ các quyết định',
    ...     'Quyết định', '07/2024/QĐ-UBND')
    'Bãi bỏ các quyết định'
    >>> strip_redundant_title_prefix(
    ...     'Nghị định 119/2005/NĐ-CP', 'Bản dịch văn bản', '119/2005/NĐ-CP')
    'Nghị định 119/2005/NĐ-CP'
    >>> strip_redundant_title_prefix(
    ...     'Decree 05/1998/ND-CP', 'Bản dịch văn bản', '05/1998/ND-CP')
    'Decree 05/1998/ND-CP'
    >>> strip_redundant_title_prefix(None, 'Quyết định', '1/X') is None
    True
    >>> strip_redundant_title_prefix(
    ...     'Thông tư hướng dẫn sửa đổi, bổ sung chế độ thu',
    ...     'Thông tư liên tịch', '24/LB-TT')
    'Thông tư hướng dẫn sửa đổi, bổ sung chế độ thu'
    >>> strip_redundant_title_prefix(
    ...     'Thông tư quy định sửa đổi, bổ sung chế độ thu',
    ...     'Thông tư liên tịch', '80/TT-LB')
    'Thông tư quy định sửa đổi, bổ sung chế độ thu'
    >>> strip_redundant_title_prefix(
    ...     'Lỗi Ban hành Quy chế hoạt động', 'Quyết định', ['1333/TP-KHTC'])
    'Lỗi Ban hành Quy chế hoạt động'
    >>> strip_redundant_title_prefix(
    ...     'Lỗi Tam thoi', 'Thông tư', ['09/TT-LB'])
    'Lỗi Tam thoi'
    >>> strip_redundant_title_prefix(
    ...     'Về việc đính chính lỗi văn bản tại Quyết định ...',
    ...     'Quyết định', ['671/QĐ-BTTTT'])
    'Về việc đính chính lỗi văn bản tại Quyết định ...'
    >>> strip_redundant_title_prefix(
    ...     'Quy định công khai xin lỗi trong giải quyết thủ tục hành chính',
    ...     'Quyết định', ['14/2016/QĐ-UBND'])
    'Quy định công khai xin lỗi trong giải quyết thủ tục hành chính'
    """
    if title is None:
        return None
    s = str(title).strip()
    if not s:
        return None

    doc_number_list = _coerce_doc_number_list(doc_number)
    anchor = doc_number_list[0] if doc_number_list else None

    candidates: list[str] = []
    if legal_type:
        candidates.append(legal_type)
    # Add alias prefixes (e.g. for Bản dịch translations whose
    # title actually starts with "Nghị định" or "Thông tư").
    for alias in _TITLE_PREFIX_ALIASES:
        if alias not in candidates:
            candidates.append(alias)

    for prefix in candidates:
        stripped = _try_strip_prefix(s, prefix, anchor)
        if stripped is not None and len(stripped) >= _MIN_STRIPPED_TITLE_LEN:
            s = stripped
            break

    return s or None


#: Doc-type alternation reused by :func:`strip_doctype_docnum_crossrefs`.
#: Case-sensitive (vbpl is consistent about capitalisation in titles
#: -- legal-type proper nouns are always capitalised, while the
#: closely-related Vietnamese verb forms like ``"quy định"`` or
#: ``"quyết định việc"`` use lowercase first letters). Listed
#: long-to-short so the alternation greedily picks the multi-word
#: variant when one exists (``"Văn bản hợp nhất"`` beats ``"Văn
#: bản"``).
_CROSSREF_DOCTYPE_ALT = (
    r"Văn\s+bản\s+hợp\s+nhất|"
    r"Nghị\s+quyết\s+liên\s+tịch|"
    r"Thông\s+tư\s+liên\s+tịch|"
    r"Thông\s+tư\s+liên\s+bộ|"
    r"Nghị\s+định\s+thư|"
    r"Bộ\s+luật|Pháp\s+lệnh|Hiến\s+pháp|"
    r"Sắc\s+lệnh|Pháp\s+điển|"
    r"Công\s+văn|Thông\s+báo|Báo\s+cáo|"
    r"Quyết\s+định|Nghị\s+quyết|Nghị\s+định|"
    r"Thông\s+tư|Chỉ\s+thị|"
    r"Văn\s+bản|Luật|Lệnh"
)

#: Date trailer that vbpl titles paste right after a cross-reference
#: doc-number. Two shapes occur in this corpus:
#:
#: * Short:  ``ngày 18/5/2007`` / ``ngày 7-12-2018``
#: * Long:   ``ngày 27 tháng 4 năm 2005`` (Vietnamese long form)
#:
#: Both belong to the cross-reference (they pin the cited document
#: to a specific publication) and should be stripped together. The
#: leading whitespace and the entire trailer are part of the same
#: capture group so the regex consumes them as a unit.
_CROSSREF_DATE_TRAILER = (
    r"(?:\s+ngày\s+\d{1,2}"
    r"(?:\s*[/\-]\s*\d{1,2}\s*[/\-]\s*\d{2,4}"
    r"|\s+tháng\s+\d{1,2}\s+năm\s+\d{2,4}"
    r"))?"
)

#: Full cross-reference matcher: ``<DocType> [liên tịch] [số:] <DocNum>
#: [ngày <date>]``. Used by :func:`strip_doctype_docnum_crossrefs` to
#: nuke citations of OTHER documents from this document's title.
_DOCTYPE_DOCNUM_CROSSREF_RE = re.compile(
    rf"(?:{_CROSSREF_DOCTYPE_ALT})"
    r"(?:\s+liên\s+tịch)?"
    r"\s+(?:số\s*[:.]?\s*)?"
    r"\d+[A-Za-z]?[/\-][\w/\-]+"
    + _CROSSREF_DATE_TRAILER,
    re.UNICODE,
)

#: Bare doc_number strip: a doc-num-shaped token preceded by one of the
#: Vietnamese reference connectives ``số`` / ``Số`` / ``theo`` /
#: ``Theo`` / ``tại`` / ``Tại`` / ``của`` / ``Của``. The connective
#: itself stays in place (it's usually carrying the surrounding
#: prose); only the doc-num is removed. Without the connective
#: predicate we'd false-positive on standalone numbers that happen
#: to share the doc-num shape (years, telephone numbers, ratios).
_BARE_DOCNUM_PRECEDED_RE = re.compile(
    r"(số|Số|theo|Theo|tại|Tại|của|Của)\s+"
    r"\d+[A-Za-z]?[/\-][\w/\-]+"
    + _CROSSREF_DATE_TRAILER,
    re.UNICODE,
)


def strip_doctype_docnum_crossrefs(title: str | None) -> str | None:
    """Strip every ``<DocType> [số] <DocNum>`` cross-reference from a title.

    A surprising number of vbpl titles cite other documents inline --
    ``"Sửa đổi, bổ sung một số điều của Nghị định số 57/2005/NĐ-CP
    ngày 27 tháng 4 năm 2005 của Chính phủ về việc xử phạt vi phạm
    hành chính trong lĩnh vực giống cây trồng"``. The legal-type +
    doc-number block is metadata that already lives in dedicated
    columns of the referenced document; carrying it inline in this
    document's title pads embeddings with boilerplate and confuses
    pure-substring searches. We strip the citation block (including
    its trailing ``ngày <date>`` tail) wherever it appears.

    Three passes, applied iteratively to convergence:

    1. **Full citation strip** -- ``<DocType> [liên tịch] [số:]
       <DocNum> [ngày <date>]`` anywhere in the title, including
       the Vietnamese long-form date trailer
       (``ngày 27 tháng 4 năm 2005``).
    2. **Bare doc-num strip** -- a doc-num token preceded by one of
       the Vietnamese reference connectives (``số``, ``theo``,
       ``tại``, ``của``). The connective itself stays; only the
       doc-num is nuked. This handles citations where the legal
       type was elided (``"theo 57/2005/NĐ-CP"``).
    3. **Pathological-title NULL** -- when the entire input is
       *just* a doc-num-shaped token (e.g. ``"1938/QĐ-UBND"`` as
       the whole title), return ``None``. The other columns of
       the parquet still carry the bibliographic handle.

    After stripping, whitespace is collapsed and stray punctuation
    (doubled commas, lonely periods at end, empty parens) is cleaned
    so the result reads as natural Vietnamese prose. If the surviving
    string is shorter than :data:`_MIN_STRIPPED_TITLE_LEN` we return
    ``None`` so the parquet ships ``title=null`` for those rows
    instead of a meaningless residue.

    >>> strip_doctype_docnum_crossrefs('Quyết định 2304/QĐ-UBND') is None
    True
    >>> strip_doctype_docnum_crossrefs('1938/QĐ-UBND') is None
    True
    >>> strip_doctype_docnum_crossrefs(
    ...     'Sửa đổi, bổ sung một số điều của Nghị định số '
    ...     '57/2005/NĐ-CP ngày 27 tháng 4 năm 2005 của Chính phủ '
    ...     'về việc xử phạt vi phạm hành chính trong lĩnh vực '
    ...     'giống cây trồng')
    'Sửa đổi, bổ sung một số điều của của Chính phủ về việc xử phạt vi phạm hành chính trong lĩnh vực giống cây trồng'
    >>> strip_doctype_docnum_crossrefs(
    ...     'Bãi bỏ Nghị quyết số 84/2018/NQ-HĐND ngày 07/12/2018 '
    ...     'của HĐND tỉnh về việc thông qua giá sản phẩm, dịch vụ '
    ...     'công ích thủy lợi trên địa bàn tỉnh')
    'Bãi bỏ của HĐND tỉnh về việc thông qua giá sản phẩm, dịch vụ công ích thủy lợi trên địa bàn tỉnh'
    >>> strip_doctype_docnum_crossrefs(
    ...     'Quy định công khai xin lỗi trong giải quyết thủ tục hành chính')
    'Quy định công khai xin lỗi trong giải quyết thủ tục hành chính'
    >>> strip_doctype_docnum_crossrefs(None) is None
    True
    >>> strip_doctype_docnum_crossrefs('   ') is None
    True
    """
    if title is None:
        return None
    s = str(title).strip()
    if not s:
        return None

    # Pathological case: the whole title is just a doc-num token.
    if _DOCNUM_TOKEN_RE.match(s):
        return None

    prev: str | None = None
    while prev != s:
        prev = s
        s = _DOCTYPE_DOCNUM_CROSSREF_RE.sub(" ", s)
        # Keep the connective word, drop the doc-num+date tail.
        s = _BARE_DOCNUM_PRECEDED_RE.sub(r"\1", s)
        # Whitespace + punctuation cleanup: collapse repeated spaces,
        # remove space before sentence punctuation, drop empty
        # ``()`` artifacts left by stripped parenthetical
        # citations, collapse doubled commas/semicolons, and trim
        # any surrounding cruft.
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s+([,.;:])", r"\1", s)
        s = re.sub(r"\(\s*\)", "", s)
        s = re.sub(r"([,;])\s*\1+", r"\1", s)
        s = re.sub(r"^[\s,;.:]+|[\s,;.:]+$", "", s).strip()

    if not s or len(s) < _MIN_STRIPPED_TITLE_LEN:
        return None
    return s


def clean_title(
    raw: str | None,
    legal_type: str | None,
    doc_number: str | list[str] | tuple[str, ...] | None,
) -> str | None:
    """Run the full title cleanup chain (single entry point).

    Order:

    1. :func:`normalise_title` -- baseline CMS-defect cleanup
       (HTML entities, NFC, ``"số số"`` doubling, decorative
       quote stripping at boundary + paired runs, trailing
       sentence punctuation).
    2. :func:`strip_redundant_title_prefix` -- peel the doc's own
       ``"<legal_type> số <doc_number>"`` head.
    3. :func:`strip_doctype_docnum_crossrefs` -- nuke every
       ``<DocType> <DocNum>`` cross-reference left behind from
       step 2 (these cite *other* documents, not this one).
    4. **Final boundary quote sweep** -- steps 2 + 3 can expose a
       previously *embedded* quote at the new title boundary
       (``Quyết định '145/2002/QĐ-UB Về việc...`` after the
       prefix peel becomes ``'Về việc...``); the final
       :func:`_strip_decorative_quotes` re-runs the canonical
       quote policy on the now-stable boundaries.

    Returns ``None`` when any step produces an empty / too-short
    result; the parquet then ships ``title=null`` for the row, with
    the bibliographic columns still carrying the citation handle.

    >>> clean_title('Quyết định số 143/QĐ-KHTC Ban hành Quy chế',
    ...             'Quyết định', ['143/QĐ-KHTC'])
    'Ban hành Quy chế'
    >>> clean_title('Lỗi Ban hành Quy chế hoạt động', 'Quyết định', ['1/X'])
    'Lỗi Ban hành Quy chế hoạt động'
    >>> clean_title("'Về việc tổ chức thực hiện của Chính phủ", 'Quyết định', ['1/X'])
    'Về việc tổ chức thực hiện của Chính phủ'
    >>> clean_title(
    ...     "Quyết định số 03/2017/QĐ-UBND' Ban hành Quy định quản lý",
    ...     'Quyết định', ['03/2017/QĐ-UBND'])
    'Ban hành Quy định quản lý'
    >>> clean_title('Quyết định 2304/QĐ-UBND', 'Quyết định', ['2304/QĐ-UBND']) is None
    True
    >>> clean_title(None, None, None) is None
    True
    """
    t = normalise_title(raw)
    if t is None:
        return None
    t = strip_redundant_title_prefix(t, legal_type, doc_number)
    if t is None:
        return None
    t = strip_doctype_docnum_crossrefs(t)
    if t is None:
        return None
    # Final pass: the legal-type-prefix + cross-reference strippers
    # can *expose* a leading or trailing decorative quote that was
    # embedded in the middle of the source title and therefore
    # invisible to ``normalise_title`` (which ran first). Apply the
    # canonical quote-strip once more on the now-stable boundaries.
    t = _strip_decorative_quotes(t)
    if not t or len(t) < _MIN_STRIPPED_TITLE_LEN:
        return None
    return t


def _try_strip_prefix(
    title: str, prefix: str, doc_number: str | None,
) -> str | None:
    """Return the title with the ``"{prefix} số <number>"`` head removed.

    Returns ``None`` when the title doesn't start with the
    expected ``"{prefix} số"`` shape. Used as the inner loop of
    :func:`strip_redundant_title_prefix`.
    """
    lt = re.escape(prefix)

    # ``s\w`` matches all observed Vietnamese spellings of the
    # word "số" (with or without diacritics) -- the precomposed
    # ``ố`` U+1ED1 character is a single ``\w`` codepoint and
    # therefore not addressable via a small ``[oô]`` class.
    so_word = r"s\w\.?\s*[:.]?"

    # Path 1: anchor on the literal doc_number value (fast, exact).
    if doc_number:
        anchor = re.escape(doc_number)
        pat = re.compile(
            rf"^\s*{lt}\s+(?:{so_word}\s*)?{anchor}\s+",
            re.IGNORECASE,
        )
        m = pat.match(title)
        if m:
            return title[m.end():].strip() or None

    # Path 2: token-based stripper. Accepts:
    #   * a "Không số" sentinel, OR
    #   * a doc_number-shaped token: alnum/&./() that **must** contain
    #     at least one digit or ``/-`` separator so it can't false-
    #     positive on a plain Vietnamese word like ``"hướng"`` or
    #     ``"quy"`` -- e.g. ``"Thông tư hướng dẫn ..."`` must NOT
    #     get stripped to ``"dẫn ..."``.
    # The token character class includes Vietnamese diacritic-bearing
    # letters via ``\w`` (unicode-aware in Python regex). The leading
    # ``số`` anchor is mandatory in Path 2 -- without it the head of
    # the title is just a noun phrase and there's no real ``doc_number``
    # to peel.
    pat = re.compile(
        rf"""
        ^\s*{lt}\s+                              # "<legal-type> "
        (?:
            (?:{so_word}\s*)                     # required "số" / "Số:" / "Số."
            (?:
                Kh\w+\s+s\w+                     # "Không số" sentinel (diacritic-tolerant)
                |
                [\w\.()&]*[\d/\-][\w\.()&/\-]*   # doc_number-shaped token (must have digit or /-)
                (?:\s*[/\-]\s*[\w\.()&]+)*       # ... extra /- runs
            )
            |
            [\w\.()&]*[\d/\-][\w\.()&/\-]*       # bare doc_number-shaped token (must have digit or /-)
            (?:\s*[/\-]\s*[\w\.()&]+)*           # ... extra /- runs
        )
        \s+                                      # whitespace before subject
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    m = pat.match(title)
    if m:
        return title[m.end():].strip() or None

    return None


def normalise_label(raw: str | None) -> str | None:
    """Normalise a short label (``legal_area``, ``issuing_authority``).

    Same baseline as :func:`normalise_text` plus stripping the
    trailing sentence punctuation that vbpl pastes into the
    inline-list source values (e.g. ``"Viễn thông và Internet;"``).

    >>> normalise_label('Viễn thông và Internet;')
    'Viễn thông và Internet'
    >>> normalise_label('Sở Giáo dục &amp; Đào tạo')
    'Sở Giáo dục & Đào tạo'
    """
    s = normalise_text(raw)
    if s is None:
        return None
    s = _TRAILING_SENTENCE_PUNCT_RE.sub("", s)
    return s.strip() or None


#: Canonical VBPL document-type codes that occasionally leak into the
#: ``agencyName`` field on the gateway side (e.g. Thanh Hóa ships
#: ``"CT UBND Tỉnh Thanh Hóa"`` for 158 docs across multiple doc
#: types -- the ``CT`` is an editorial slip, not part of the agency
#: name). We only strip a code when the *remainder* begins with a
#: real Vietnamese government-body keyword so that legitimate
#: short-form agency names (``UBND``, ``HĐND``, ``Bộ ...``) stay
#: intact and we don't false-positive on natural prose.
_AGENCY_LEAKED_CODE_RE = re.compile(
    r"^(?:CT|QĐ|NQ|NĐ|TT|TTLT|TTLB|CTr|CV|PL|BD|BGN|BL|HP|HĐ|KXĐ|L|"
    r"L-CTN|NĐT|NQLT|SL|SLT|TB|ThT|VBHC|VBHN|VBK|VBLQ)\s+"
    r"(?=(?:UBND|HĐND|Ủy\s*ban|Bộ\s|Sở\s|Cục\s|Tổng\s*cục|"
    r"Văn\s*phòng|Tòa\s*án|Toà\s*án|Viện\s|Hội\s|Ban\s|Liên\s|"
    r"Chính\s*phủ|Chủ\s*tịch|Phòng\s|Trung\s*tâm|Quốc\s*hội|"
    r"Thủ\s*tướng|Ủy\s*viên|Đảng\s|Mặt\s*trận|Ngân\s*hàng))",
)


def normalise_issuing_authority(raw: str | None) -> str | None:
    """Normalise the issuing-agency name.

    Applies :func:`normalise_label` (text cleanup + trailing
    punctuation strip) then peels any leaked VBPL doc-type code
    prefix (e.g. ``"CT UBND ..."`` -> ``"UBND ..."``). Idempotent.

    >>> normalise_issuing_authority('CT UBND Tỉnh Thanh Hóa')
    'UBND Tỉnh Thanh Hóa'
    >>> normalise_issuing_authority('UBND Tỉnh Thanh Hóa')
    'UBND Tỉnh Thanh Hóa'
    >>> normalise_issuing_authority('QĐ Bộ Tài chính')
    'Bộ Tài chính'
    >>> normalise_issuing_authority('Bộ Công an')
    'Bộ Công an'
    >>> normalise_issuing_authority('CT')
    'CT'
    >>> normalise_issuing_authority('  CT  UBND Tỉnh Thanh Hóa  ')
    'UBND Tỉnh Thanh Hóa'
    >>> normalise_issuing_authority(None) is None
    True
    >>> normalise_issuing_authority('Hội đồng nhân dân tỉnh CT')
    'Hội đồng nhân dân tỉnh CT'
    """
    s = normalise_label(raw)
    if s is None:
        return None
    s = _AGENCY_LEAKED_CODE_RE.sub("", s)
    return s.strip() or None


#: Boilerplate preamble injected by the vbpl.vn gateway when the
#: body ships inline as HTML. The portal wraps every body in a
#: tiny CSS shim so the browser preview renders nicely; that shim
#: is pure scaffolding and has no place in the embedding text.
#: The three pieces (``Document Content`` label, ``body { … }``
#: rule, ``p { … }`` rule) are all optional and appear in roughly
#: 45-55 % of docs depending on the source's vintage.
#:
#: The label terminator is ``\s*`` (any whitespace), not ``\n+``,
#: because the May-2026 corpus has many bodies where the gateway
#: dropped the CSS shim but kept the label inline, so the document
#: opens like ``Document Content SẮC LỆNH ...`` with only a single
#: space between the label and the real body. ``\b`` is required so
#: ``Document\s*Content`` doesn't gobble a legitimate word starting
#: with ``Content`` (eg. an English-language doc).
_MD_LEADING_WRAPPER_RE = re.compile(
    r"\A\s*"
    r"(?:Document\s*Content\b\s*)?"
    r"(?:\s*body\s*\{[^}]*\}\s*\n*)?"
    r"(?:\s*p\s*\{[^}]*\}\s*\n*)?",
    re.IGNORECASE,
)

#: Non-anchored sweep for the ``Document Content`` gateway label.
#: The leading-anchor regex above catches the common case (label at
#: ``\A``, ~74 K rows). For PDF/DOCX-sourced docs the parser
#: sometimes splices the bibliographic header (``BỘ TÀI CHÍNH ...
#: Số: 65/2020/TT-BTC Ngày 9 tháng 7 năm 2020``) in front of the
#: gateway label, so it ends up mid-stream (~13 rows). The literal
#: phrase ``Document Content`` is English boilerplate that does not
#: occur in legitimate Vietnamese legal text, so a non-anchored
#: sweep is safe.
#:
#: ``\s+`` (not ``\s*``) between ``Document`` and ``Content`` is
#: required so we never match the camel-case JavaScript i18n key
#: ``documentContent`` that leaks in via ``shell_html`` bodies --
#: those rows are NULL'd in the published parquet anyway, but the
#: extract tier should still leave them untouched for diagnostic
#: replay. ``\b`` on both sides guards against word-internal
#: collisions (no realistic Vietnamese text combines these two
#: English nouns with a single space, but the boundary is cheap
#: insurance).
_MD_DOC_CONTENT_LABEL_RE = re.compile(
    r"\bDocument\s+Content\b\s*",
    re.IGNORECASE,
)

#: HTML comments survive ``markdownify``'s default conversion --
#: they carry the entire Microsoft-Word-derived ``@font-face`` /
#: ``p.MsoNormal { … }`` stylesheet that bloats ~50 % of the
#: corpus. Everything between the markers is junk for our purpose;
#: the trailing ``\Z`` alternative covers the (rare) case where
#: the upstream HTML is malformed and never closes the comment.
_MD_HTML_COMMENT_RE = re.compile(r"<!--[\s\S]*?(?:-->|\Z)")

#: CSS-property tells. Matching ANY of these inside a block's body
#: marks the block as CSS junk worth deleting. The list is kept
#: tight to avoid false positives on natural Vietnamese text
#: (which never contains ``font-family``, ``mso-…``, panose
#: vectors or ``\d+pt`` declarations).
_MD_CSS_PROPERTY_TELL_RE = re.compile(
    r"font-(?:family|size|weight|style)\b"
    r"|margin(?:-(?:top|right|bottom|left))?\s*:"
    r"|padding(?:-(?:top|right|bottom|left))?\s*:"
    r"|line-height\s*:"
    r"|text-(?:align|indent|transform|decoration|justify|rendering)\s*:"
    r"|page-break(?:-[a-z]+)?\s*:"
    r"|border(?:-collapse|-style|-width|-color)?\s*:"
    r"|background(?:-color|-image)?\s*:"
    r"|color\s*:\s*(?:#|rgb|rgba)"
    r"|vertical-align\s*:"
    r"|display\s*:\s*(?:inline|block|flex|grid|none|table)"
    r"|panose-\d"
    r"|mso-[a-z\-]+"
    r"|-(?:webkit|moz|ms|o)-[a-z\-]+"
    r"|@font-face|@page|@keyframes|@media\b"
    r"|\d+(?:\.\d+)?(?:pt|px|cm|in|em|rem|%)\b",
    re.IGNORECASE,
)

#: CSS-selector tells. A block whose selector looks like a known
#: Word / Ant Design / framework selector is junk even if its
#: properties are empty (``div.Section1 {}``).
_MD_CSS_SELECTOR_TELL_RE = re.compile(
    r"(?:^|[\s,>+~])"
    r"(?:"
    r"p\.Mso\w*|li\.Mso\w*|div\.Mso\w*"
    r"|span\.Mso\w*|span\.Gram\w*|span\.Spelling\w*"
    r"|div\.Section\d+|div\.WordSection\d+"
    r"|\.MsoChpDefault|\.MsoTableGrid|\.MsoFootnoteText"
    r"|@font-face|@page|@media\b"
    r"|@(?:-(?:webkit|moz|ms|o)-)?keyframes\b"
    r"|\.anticon[\w\-]*|\.ant-[\w\-]+"
    r"|\.css-[\w]+|:where\("
    r"|input:[\w\-]+\([^)]*\)"   # Ant Design ``input:where(…)`` chains
    r")",
    re.IGNORECASE,
)


#: Structural "looks like CSS" sniffer. Matches a brace block whose
#: body is one or more ``prop:value;`` declarations. This catches
#: rare CSS properties that aren't in our explicit tells list
#: (``transform``, ``opacity``, ``box-shadow``, ``flex-…``,
#: ``cursor``, ``outline``, …) -- common in Ant Design's
#: ``@keyframes`` blocks. Anchored on the brace shape so Vietnamese
#: text like ``Điều 5 {bao gồm cả khoản 3}`` (no ``key:value;``
#: structure) is never misread as CSS.
_MD_CSS_PROPS_SHAPE_RE = re.compile(
    r"\{\s*(?:[\w\-]+\s*:\s*[^;{}]+;\s*){1,}\}"
)


def _is_css_chunk(text: str) -> bool:
    """Return ``True`` when ``text`` looks like a CSS rule block."""
    return bool(
        _MD_CSS_PROPERTY_TELL_RE.search(text)
        or _MD_CSS_SELECTOR_TELL_RE.search(text)
        or _MD_CSS_PROPS_SHAPE_RE.search(text)
    )


#: Single CSS rule: ``selector { props }``. The selector class is
#: ASCII-only (CSS selectors never legitimately contain Vietnamese
#: chars), bounded to 8000 chars (Ant Design's chained
#: ``:where(.css-…).ant-typography div +h1, …`` selectors run
#: 2-7 KB in this corpus), and excludes ``|`` so markdown tables
#: can't be misread as selectors. ``[^{}]*`` for properties relies
#: on the fact that flat CSS rules dominate; nested
#: ``@keyframes name { 100% { … } }`` blocks are handled by
#: running the regex iteratively in :func:`strip_markdown_junk`,
#: peeling off one level of braces per pass.
_MD_CSS_BLOCK_RE = re.compile(
    # Selector char class -- ASCII only, plus ``%`` for keyframe
    # percentages (``0% { … }``, ``100% { … }``), plus ``&`` and
    # ``/`` so vendor-prefixed selectors and Sass/SCSS-style
    # parenthesised chains both flow through.
    r"[a-zA-Z0-9_ \t.#:>+~,*@\-\"'\\\[\]\$=^()&%/]{1,8000}?\{[^{}]{0,12000}?\}"
)

#: Orphan CSS selector lines/fragments left over after the inner
#: ``{ … }`` of a comma-chained selector was stripped from a
#: different position. Example: ``.anticon-spin::before,`` on its
#: own line after the parent block was deleted. Anchored on
#: lines that look like CSS selectors (start with ``.``, ``#``,
#: ``@``, ``:`` or known framework prefixes) and contain only
#: ASCII selector chars (no Vietnamese text, no ``{`` / ``}``).
_MD_ORPHAN_SELECTOR_RE = re.compile(
    r"(?m)"
    r"^[ \t]*"
    r"(?:[\.#@:][a-zA-Z][\w.\-:#@\[\]()*=,>+~\"'\\^$ \t]{0,500}"
    r"|input:[\w\-]+\([^)\n]*\)[a-zA-Z0-9_ \t.#:>+~,*@\-\"'\\\[\]\$=^()]{0,500}"
    r")"
    r"[,;:]?[ \t]*$",
)

#: Lonely CSS comment markers (``/\* Font Definitions \*/``) that
#: ``markdownify`` escapes when they live outside an HTML comment.
#: After we strip the surrounding CSS block these escaped markers
#: can persist as orphan lines; sweep them out so the body doesn't
#: keep stray ``\*`` decoration.
_MD_CSS_COMMENT_RE = re.compile(
    r"^/\s*\\\*[\s\S]*?\\\*/\s*$", re.MULTILINE,
)

#: Inline-styled HTML tags that ``markdownify`` fails to strip when
#: the source HTML uses malformed attribute syntax (vbpl.vn pastes
#: documents from Word with attributes like
#: ``<span lang="EN-GBstyle=" font-size:12.0pt;font-family:'times"="">``).
#: We sweep them as a separate, conservative pass so the visible
#: text becomes pure markdown.
_MD_INLINE_STYLE_TAG_RE = re.compile(
    r"</?(?:span|div|p|a|font|table|tr|td|th|tbody|thead|tfoot)\b"
    r"[^>\n]{0,500}"
    r"(?:font-(?:family|size|weight)|style=|class=|lang=)"
    r"[^>\n]{0,500}>",
    re.IGNORECASE,
)


def strip_markdown_junk(md: str | None) -> str | None:
    """Strip the Word/CSS scaffolding that ``markdownify`` leaves behind.

    The vbpl.vn gateway returns bodies wrapped in a small CSS shim
    (so its own preview can render nicely) and, for documents that
    were authored in Microsoft Word, dumps the entire Word
    stylesheet into the body. A third class of docs falls through
    to the rendered Next.js shell, picking up ~200 KB of
    Ant Design CSS along the way. All of it rides through the
    ``markdownify`` pass intact and ends up in the markdown column
    as visible noise like::

        Document Content

        body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
        p { margin: 10px 0; }

        <!--
        /\\* Font Definitions \\*/
        @font-face
        {font-family:Wingdings;
        panose-1:5 0 0 0 0 0 0 0 0 0;}
        …
        -->

    or like::

        .anticon { display: inline-flex; … }
        :where(.css-1rb9ewt).ant-breadcrumb{font-family:'Inter',…}

    This is pure scaffolding -- it pads the embedding text by
    1-200 KB per document with content that has zero semantic
    relevance to Vietnamese legal queries. The cleanup is layered:

    1. Drop the leading ``Document Content`` + ``body{}`` + ``p{}``
       wrapper that appears at the head of ~50 % of bodies.
    2. Drop every ``<!-- … -->`` HTML comment (the Word stylesheet
       dump always lives inside one).
    3. Drop any ``selector { properties }`` block whose properties
       trip a CSS-property heuristic (``font-family``, ``mso-*``,
       ``@font-face``, … or any ``\\d+pt`` declaration) **or**
       whose selector matches a Word / Ant Design framework name
       (``p.MsoNormal``, ``div.Section1``, ``:where(.css-…)``,
       ``.anticon``, …). The latter catches empty rule blocks
       (``div.Section1 {}``) and Ant Design chains.
    4. Sweep orphan ``/\\* … \\*/`` CSS comment lines that survive
       steps 1-3 outside an HTML comment.
    5. Collapse the resulting blank-line runs to at most two so the
       output looks like normal post-``markdownify`` output.

    The selector regex is **ASCII-only** so it never matches a
    legitimate Vietnamese phrase that happens to wrap a parenthetical
    in curly braces (``Điều 5 {bao gồm cả khoản 3}``).

    >>> strip_markdown_junk('Document Content\\n\\nbody { font-family: Arial; }\\np { margin: 10px 0; }\\n\\nHello\\n')
    'Hello'
    >>> strip_markdown_junk('Document Content SẮC LỆNH ...')
    'SẮC LỆNH ...'
    >>> strip_markdown_junk('Document Content\\nReal text')
    'Real text'
    >>> strip_markdown_junk('Số: 65/2020/TT-BTC Ngày 9 tháng 7 năm 2020 Document Content Bộ TÀI CHÍNH')
    'Số: 65/2020/TT-BTC Ngày 9 tháng 7 năm 2020 Bộ TÀI CHÍNH'
    >>> strip_markdown_junk('Header\\n\\n<!-- /* css */ p.MsoNormal { margin: 0; } -->\\n\\nBody\\n')
    'Header\\n\\n\\nBody'
    >>> strip_markdown_junk('p.MsoNormal { margin:0in;\\nfont-size:12pt;}\\nh1 { font-family: Arial; }\\nReal text')
    'Real text'
    >>> strip_markdown_junk('div.Section1 {\\n}\\nspan.GramE {}\\nReal text')
    'Real text'
    >>> strip_markdown_junk('a{color:#fff;}b{font-size:12px;}c{margin:0;}Real text')
    'Real text'
    >>> strip_markdown_junk('Điều 5 {bao gồm cả khoản 3}\\nNội dung')
    'Điều 5 {bao gồm cả khoản 3}\\nNội dung'
    >>> strip_markdown_junk(None) is None
    True
    >>> strip_markdown_junk('') == ''
    True
    """
    if md is None:
        return None
    if not md:
        return md

    s = md
    s = _MD_LEADING_WRAPPER_RE.sub("", s, count=1)
    s = _MD_DOC_CONTENT_LABEL_RE.sub("", s)
    s = _MD_HTML_COMMENT_RE.sub("", s)

    def _maybe_strip(m: re.Match[str]) -> str:
        block = m.group(0)
        return "" if _is_css_chunk(block) else block

    # Iterate to peel one level of nested CSS per pass. ``@keyframes``
    # blocks (``@keyframes name { 100% { … } }``) need exactly two
    # passes: the first strips the inner ``100% { … }`` rule, the
    # second strips the now-empty ``@keyframes name { … }`` outer
    # rule. Cap at 5 iterations as a safety net against pathological
    # inputs; in practice 2-3 passes cover everything.
    for _ in range(5):
        new_s = _MD_CSS_BLOCK_RE.sub(_maybe_strip, s)
        if new_s == s:
            break
        s = new_s
    s = _MD_CSS_COMMENT_RE.sub("", s)
    s = _MD_INLINE_STYLE_TAG_RE.sub("", s)

    # Sweep orphan selector fragments. Comma-chained selectors like
    # ``.anticon-spin::before, .anticon-spin { … }`` get partially
    # stripped (the inner ``{ … }`` and the trailing selector are
    # removed, but the leading ``.anticon-spin::before,`` line is
    # left orphaned). Drop lines that look like pure CSS selectors
    # with no Vietnamese / textual content -- a conservative cleanup
    # since the regex requires the line to start with a CSS-only
    # punctuation character and to contain no body text characters.
    s = _MD_ORPHAN_SELECTOR_RE.sub("", s)

    out_lines: list[str] = []
    blanks = 0
    for line in s.splitlines():
        if not line.strip():
            blanks += 1
            if blanks <= 2:
                out_lines.append("")
            continue
        blanks = 0
        out_lines.append(line.rstrip())
    return "\n".join(out_lines).strip()


__all__ = [
    "clean_title",
    "normalise_doc_number",
    "normalise_doc_number_list",
    "normalise_issuing_authority",
    "normalise_label",
    "normalise_text",
    "normalise_title",
    "strip_doctype_docnum_crossrefs",
    "strip_markdown_junk",
    "strip_redundant_title_prefix",
]
