"""congbobanan-specific normalizers (wiki/DATASITES.md §3.5 + registry pattern).

congbobanan PDFs are digital-text court judgments parsed by ``pypdf``.
Three residual artefact classes survive the universal
``letter_spaced_collapse → vietnamese_text`` chain:

1. **Single-space mid-word splits.** pypdf injects a single space
   inside a word whenever the kerning between two adjacent glyphs
   exceeds its character-spacing threshold. The artefact looks like

       "hu yện", "ch ung", "chứn g", "ng ười", "t hụ", "phá t sóng"

   The universal :class:`LetterSpacedCollapseNormalizer` only fires
   on 2+-space all-letter runs (``T h ô n g  t i n``), so these
   single-space cases sail through. The dedicated
   :class:`JoinWordBreaks` normalizer below rebuilds the original
   word using a Vietnamese-syllable predicate (onset + nucleus +
   coda regex), constrained to splits where at least one side is
   ≤ 2 chars -- the regime where the split is implausible for
   natural Vietnamese text.

2. **Soft-wrap line breaks.** The PDF lays paragraphs out as
   physically wrapped lines (a paragraph that occupies four
   visual lines emits four ``\\n``-terminated rows in pypdf
   output) but the *logical* paragraph is one sentence. Without
   reflow, every clause is split mid-phrase ("đội tuyển Anh với
   đội\\ntuyển Iceland"). :class:`JoinSoftWraps` reflows
   continuation lines into the previous line with a single space,
   guarded by terminal-punctuation, markdown-header, and
   list-marker heuristics so paragraph and structural boundaries
   stay intact.

3. **Per-page bare-digit line.** Every ``## Page N`` block emitted
   by :class:`packages.parser.stage.PdfParseStage` starts with the
   PDF's own page number on its first body line ("## Page 2\\n\\n2
   \\nNam 03..."). That single-digit line is page furniture, not
   document content, and shows up as a stray ``"2"`` paragraph in
   the rendered markdown. :class:`StripPageNoise` removes it when
   it matches the header's number.

All three normalizers are idempotent: applying them twice yields
the same string. They register themselves into the global
:data:`packages.extractor.normalizers.NORMALIZER_REGISTRY` at
import time -- see :mod:`packages.datasites.congbobanan.__init__`
for the eager-import pattern that ensures the registry is
populated for both the driver and remote Ray workers.

Chain order recommended by ``configs/default.yaml``::

    parser:
      normalizers:
        - letter_spaced_collapse            # universal (2+-space runs)
        - congbobanan_join_word_breaks      # site   (1-space mid-word)
        - vietnamese_text                   # universal (ftfy + NFC + tone)
        - congbobanan_join_soft_wraps       # site   (PDF line-wrap reflow)
        - congbobanan_strip_page_noise      # site   (per-page bare-digit)

``congbobanan_join_soft_wraps`` runs AFTER ``vietnamese_text`` so
its terminal-punctuation tests see canonicalised Unicode (NFC tone
marks, ftfy-repaired mojibake) rather than raw pypdf output. It
runs BEFORE ``congbobanan_strip_page_noise`` so the bare-digit
body line that follows each page header is still on its own line
and intact when the page-noise stripper's regex hunts for it.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from packages.extractor.normalizers import register_normalizer

# --------------------------------------------------------------------- syllable predicate

# Vietnamese nucleus inventory: 12 base vowels each with 5 tone marks
# plus an ALL-CAPS variant. Building the class explicitly avoids
# any locale-sensitive surprises from ``re.IGNORECASE``.
_VN_VOWELS_LOWER = (
    "aăâeêioôơuưy"
    "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩị"
    "òóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ"
)
_VN_VOWELS = _VN_VOWELS_LOWER + _VN_VOWELS_LOWER.upper()
_VN_VOWEL_CLASS = f"[{_VN_VOWELS}]"

# Onsets sorted longest-first so the regex engine tries multi-letter
# clusters before single-letter ones (``ngh`` before ``ng`` before
# ``n``). Codas use the canonical 8-element Vietnamese set.
_VN_ONSETS = (
    "ngh", "ng", "nh", "ch", "gh", "gi", "kh", "ph", "qu", "th", "tr",
    "b", "c", "d", "đ", "g", "h", "k", "l", "m", "n", "p", "r", "s",
    "t", "v", "x",
)
_VN_CODAS = ("ng", "ch", "nh", "c", "m", "n", "p", "t")

_ONSET_PATTERN = "(?:" + "|".join(_VN_ONSETS) + ")?"
_CODA_PATTERN = "(?:" + "|".join(_VN_CODAS) + ")?"

# A valid Vietnamese syllable: optional onset + 1-3 vowel chars +
# optional coda. ``re.IGNORECASE`` lets ALL-CAPS letterheads
# ("TUYÊN") match the same regex as lowercase body text.
_VN_SYLLABLE_RE = re.compile(
    rf"^{_ONSET_PATTERN}{_VN_VOWEL_CLASS}{{1,3}}{_CODA_PATTERN}$",
    re.IGNORECASE,
)

# Vietnamese syllables are at most 7 NFC characters in this corpus
# (``nghiêng``, ``trường``); a joined run exceeding this is almost
# always two real words gluing accidentally.
_MAX_SYLLABLE_LEN = 7
# A token of >2 chars is plausibly a standalone Vietnamese word, so
# the heuristic refuses to glue it onto a neighbour even when the
# joined form parses as a syllable. This avoids the "khi anh" ->
# "khianh" false-positive class while still catching every observed
# pypdf split in the congbobanan corpus (left or right side is
# always ≤ 2 chars in practice).
_MAX_SHORT_SIDE_LEN = 2
# Minimum joined-form length. Without this guard the joiner glues
# pypdf's lossy-glyph artefacts together: when the PDF's embedded
# font has no Unicode mapping for a tone-marked vowel pypdf drops
# the codepoint and emits a bare space, so "đấu" -> "đ u". Joining
# "đ" + "u" -> "đu" passes the syllable regex (đ + vowel) but
# DESTROYS evidence of the dropped tone mark and produces a
# different real Vietnamese word. Requiring joined-len ≥ 3 keeps
# every legitimate split case ("thụ", "phát", "chung", "chứng",
# "người", "phường") while rejecting the 2-char lossy-glyph
# artefact class.
_MIN_JOINED_LEN = 3


_HAS_VN_VOWEL_RE = re.compile(_VN_VOWEL_CLASS)


def _is_valid_syllable(token: str) -> bool:
    """Return True if ``token`` matches the Vietnamese syllable shape."""
    return bool(token) and bool(_VN_SYLLABLE_RE.match(token))


def _has_vietnamese_vowel(token: str) -> bool:
    """Return True if ``token`` contains at least one Vietnamese vowel char."""
    return bool(_HAS_VN_VOWEL_RE.search(token))


def _should_join(left: str, right: str) -> bool:
    """Decide whether ``left`` + ``right`` is a pypdf mid-word split.

    All five conditions must hold:

    1. Neither side is empty.
    2. ``min(len(left), len(right)) ≤ 2`` -- the regime where a
       standalone token is implausible for natural Vietnamese
       legal text.
    3. Neither side is pure ASCII digits. ``"1 2"`` must never
       become ``"12"``; the rare year-split case (``"201 8"``) is
       not worth the false-positive risk on list numbering.
    4. **At least one side has no Vietnamese vowel.** A
       consonant-only fragment (``"ch"``, ``"ng"``, ``"t"``,
       ``"g"``) cannot stand alone as a real Vietnamese word, so
       its presence is a strong signal that a glyph break inside
       one syllable produced the split. Conversely, two short
       sides that BOTH carry vowels (``"Tòa"`` + ``"án"``,
       ``"có"`` + ``"a"``) are presumed to be two real
       neighbouring words and stay separated -- this rule keeps
       the joiner from corrupting common Vietnamese 2-char words
       that happen to glue into a phonotactically valid syllable
       shape.
    5. The joined token's length sits in
       ``[_MIN_JOINED_LEN, _MAX_SYLLABLE_LEN]``. The lower bound
       (3 chars) is the lossy-pypdf guard described above; the
       upper bound caps the predicate at the longest real
       Vietnamese syllable length. The regex match is the final
       phonotactic-shape filter.

    Trade-off: rule 4 misses the rare "vowel-only mid-word"
    splits where pypdf cuts a single syllable between two
    vowel runs (e.g. ``"hu yện"`` -> would-be ``"huyện"``).
    These are uncommon in the congbobanan corpus and the
    downstream extractor is robust to them; the alternative
    (dropping rule 4) corrupts ``"Tòa án"`` into ``"Tòaán"``
    everywhere, which would break case-citation NER. The strict
    rule is the safer trade.
    """
    if not left or not right:
        return False
    if min(len(left), len(right)) > _MAX_SHORT_SIDE_LEN:
        return False
    if left.isdigit() or right.isdigit():
        return False
    if _has_vietnamese_vowel(left) and _has_vietnamese_vowel(right):
        return False
    joined = left + right
    if not (_MIN_JOINED_LEN <= len(joined) <= _MAX_SYLLABLE_LEN):
        return False
    return _is_valid_syllable(joined)


# --------------------------------------------------------------------- mid-word joiner

# Single-space runs only. The chain runs ``letter_spaced_collapse``
# first, which leaves 2+-space boundaries between rebuilt words
# untouched; we must NOT join across those.
_SINGLE_SPACE_SPLIT_RE = re.compile(r"( +)")


def _join_line(line: str) -> str:
    """Walk a single line, fusing pypdf mid-word splits in place."""
    if not line.strip():
        return line
    chunks = _SINGLE_SPACE_SPLIT_RE.split(line)
    # chunks alternates: [tok, sep, tok, sep, ..., tok]. Tokens live
    # at even indices, separators at odd. A line with no spaces
    # produces a single-element list -- nothing to fuse.
    if len(chunks) < 3:
        return line

    out_tokens: list[str] = [chunks[0]]
    out_seps: list[str] = []
    i = 1
    while i < len(chunks) - 1:
        sep = chunks[i]
        next_tok = chunks[i + 1]
        if sep == " " and _should_join(out_tokens[-1], next_tok):
            # Cascading merge: the rebuilt token becomes the new
            # ``left`` candidate for the next iteration, so
            # "ng ư ời" -> "ngư ời" -> "người" in one pass.
            out_tokens[-1] = out_tokens[-1] + next_tok
        else:
            out_seps.append(sep)
            out_tokens.append(next_tok)
        i += 2

    rebuilt: list[str] = [out_tokens[0]]
    for sep, tok in zip(out_seps, out_tokens[1:], strict=True):
        rebuilt.append(sep)
        rebuilt.append(tok)
    return "".join(rebuilt)


def _join_word_breaks(text: str) -> str:
    """Fuse pypdf single-space mid-word splits across all lines of ``text``."""
    if not text or not isinstance(text, str):
        return text
    # ``splitlines(keepends=True)`` preserves \r\n / \n / \r terminators
    # so the rebuilt text is byte-for-byte identical on lines that
    # don't trigger any fusion.
    lines = text.splitlines(keepends=True)
    if not lines:
        return text
    out: list[str] = []
    for raw in lines:
        if raw.endswith("\r\n"):
            body, term = raw[:-2], "\r\n"
        elif raw.endswith("\n") or raw.endswith("\r"):
            body, term = raw[:-1], raw[-1]
        else:
            body, term = raw, ""
        out.append(_join_line(body) + term)
    return "".join(out)


@register_normalizer("congbobanan_join_word_breaks")
class JoinWordBreaks:
    """Rebuild pypdf mid-word single-space splits on the ``markdown`` column.

    Targets the residual artefact class the universal
    :class:`LetterSpacedCollapseNormalizer` leaves behind: cases
    where pypdf injects a *single* space between glyphs inside one
    word, never reaching the 2+-space threshold the letter-spacing
    collapser keys on.

    Place BEFORE :class:`packages.extractor.normalizers.VietnameseTextNormalizer`
    in the chain so the regex still sees the original NFC tone marks
    on the pre-normalisation glyphs.

    Idempotent: a re-run is a no-op because the previously fused
    token alone is already a valid Vietnamese syllable, and the
    ``min(len) ≤ 2`` guard refuses to glue it onto its neighbours.
    """

    name: str = "congbobanan_join_word_breaks"
    columns: tuple[str, ...] = ("markdown",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "markdown" not in df.columns:
            return df
        df["markdown"] = df["markdown"].map(
            lambda v: _join_word_breaks(v) if isinstance(v, str) and v else v,
        )
        return df


# --------------------------------------------------------------------- soft-wrap reflow

# Terminal punctuation that ENDS a logical line. When the previous
# line ends with one of these, the next line is treated as a new
# paragraph / clause / list item and is NOT merged. Includes both
# ASCII and the Vietnamese-typographic close-quote (``”``) commonly
# emitted by ftfy after fix_latin_ligatures.
_TERMINAL_PUNCT = frozenset(".:?!;…”»)]}>")

# Open punctuation. When the previous line ENDS with one of these,
# the next line joins WITHOUT an inserted space ("phạm tội (\ntheo
# Điều 248)" -> "phạm tội (theo Điều 248)"). Symmetric with the
# terminal set above on bracket pairs.
_OPEN_PUNCT = frozenset("([{<\"'“«")

# A line that starts with one of these patterns is a structural
# element (list marker, table row, blockquote, footnote ref) and
# never folds into the previous line. The regex is anchored to the
# first non-whitespace position so indented lists still match.
#
#   - foo, * foo, • foo, ‣ foo            -- bullet lists
#   1. foo, 12) foo                       -- ordered lists
#   a) foo, iv. foo                       -- alpha / Roman lists
#   | col | col                           -- markdown table row
#   > quoted                              -- blockquote
#   [^1]: footnote                        -- footnote definition
_STRUCTURAL_START_RE = re.compile(
    r"^\s*("
    r"[-*•‣]\s"                           # bullet markers
    r"|\d{1,3}[.)]\s"                     # ordered list (1. / 12))
    r"|[a-zA-Z][.)]\s"                    # alpha list (a) / b.)
    r"|[ivxlcdmIVXLCDM]+[.)]\s"           # Roman numeral list
    r"|\|"                                # table row
    r"|>"                                 # blockquote
    r"|\[\^[^\]]+\]:"                     # footnote definition
    r")",
)


def _is_continuation(prev: str, nxt: str) -> bool:
    """Decide whether ``nxt`` continues the same logical line as ``prev``.

    True iff *every* condition holds:

    1. Both lines have non-whitespace content. A blank line is the
       canonical paragraph separator and never folds.
    2. Neither line is a markdown header (``#`` … ``######``).
       Headers are structural and must keep their newline.
    3. ``nxt`` doesn't start with a structural marker (list bullet,
       ordered-list digit, table row, blockquote, footnote ref).
    4. ``prev`` doesn't end with terminal punctuation (``.:?!;…``)
       or a close bracket. Those mark sentence / clause / scope
       boundaries -- folding would mash distinct sentences together.

    The check is purely string-shape; it has no notion of "the
    paragraph this line belongs to" beyond looking at the
    immediately-adjacent neighbours, which is enough for the
    pypdf-style "every paragraph wraps every ~80 chars" output.
    """
    prev_stripped = prev.strip()
    nxt_stripped = nxt.strip()
    if not prev_stripped or not nxt_stripped:
        return False
    if prev_stripped.startswith("#"):
        return False
    if nxt_stripped.startswith("#"):
        return False
    if _STRUCTURAL_START_RE.match(nxt):
        return False
    if prev_stripped[-1] in _TERMINAL_PUNCT:
        return False
    return True


def _join_soft_wraps(text: str) -> str:
    """Fold continuation lines back into their logical paragraph.

    Walks the document line-by-line, maintaining an ``out`` buffer.
    For each incoming line, if the previous emitted line is a
    soft-wrap continuation point (see :func:`_is_continuation`),
    the new line is appended to the previous one separated by a
    single space (or no space if the previous line ends with an
    open-bracket / open-quote). Otherwise the incoming line is
    pushed as a new entry, preserving the structural break.

    Idempotent: a cleaned document has no soft-wrap boundaries
    left to fold, so a second application emits the same string.
    """
    if not text or not isinstance(text, str):
        return text
    # Preserve trailing newlines on the document as a whole.
    trailing = ""
    body = text
    while body.endswith("\n"):
        trailing += "\n"
        body = body[:-1]
    if not body:
        return text

    lines = body.split("\n")
    out: list[str] = []
    for line in lines:
        if out and _is_continuation(out[-1], line):
            prev = out.pop()
            prev_stripped_right = prev.rstrip()
            if prev_stripped_right and prev_stripped_right[-1] in _OPEN_PUNCT:
                # "phạm tội (\ntheo" -> "phạm tội (theo" -- the open
                # bracket already implies a logical attachment to
                # the next token, so no separator space.
                out.append(prev_stripped_right + line.lstrip())
            else:
                out.append(prev_stripped_right + " " + line.lstrip())
        else:
            out.append(line)
    return "\n".join(out) + trailing


@register_normalizer("congbobanan_join_soft_wraps")
class JoinSoftWraps:
    """Reflow PDF soft-wrap line breaks into continuous paragraphs.

    pypdf preserves the PDF's *visual* line wraps as hard newlines
    in the extracted markdown, so a paragraph that occupies four
    visual lines arrives as four ``\\n``-terminated rows even though
    the logical content is one sentence. This normalizer joins
    consecutive lines back into the paragraph they belong to,
    using terminal punctuation, markdown headers, and list
    markers as paragraph-break signals.

    Place AFTER :class:`packages.extractor.normalizers.VietnameseTextNormalizer`
    so the join logic sees canonical NFC text and ftfy-cleaned
    quote glyphs (e.g. ``”`` recognised as a terminal close-quote).
    Place BEFORE :class:`StripPageNoise` so the page-noise stripper
    still finds its bare-digit body line intact on its own row.

    Idempotent: a previously-reflowed document presents no
    soft-wrap boundaries that pass :func:`_is_continuation`, so a
    re-run is a no-op.
    """

    name: str = "congbobanan_join_soft_wraps"
    columns: tuple[str, ...] = ("markdown",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "markdown" not in df.columns:
            return df
        df["markdown"] = df["markdown"].map(
            lambda v: _join_soft_wraps(v) if isinstance(v, str) and v else v,
        )
        return df


# --------------------------------------------------------------------- page-noise stripper

# Strip the bare-digit body line that follows each ``## Page N``
# header. Captures the header (group 1) + the original whitespace
# run between header and digit (group 3); the digit run is back-
# referenced via \\2 so we only strip when it matches the header's
# own number. Any trailing whitespace after the digit line is also
# consumed and normalised to a single blank line so the body stays
# tight against the header.
_PAGE_NOISE_RE = re.compile(
    r"^(## Page (\d+))(\n+)\2[ \t]*\n+",
    re.MULTILINE,
)


def _strip_page_noise(text: str) -> str:
    """Drop the bare-digit body line under each ``## Page N`` header."""
    if not text or not isinstance(text, str):
        return text
    return _PAGE_NOISE_RE.sub(r"\1\n\n", text)


@register_normalizer("congbobanan_strip_page_noise")
class StripPageNoise:
    """Remove the per-page leading bare-digit line on the ``markdown`` column.

    pypdf preserves the printed page number from the PDF's own
    header glyph and emits it as the first body line under the
    parser's ``## Page N`` marker. The bare ``"2"`` paragraph is
    page furniture, not content; this normalizer removes it iff
    the digit run matches the header's page number, leaving body
    numerals (statute clause numbers, dates, money amounts) intact.

    Place LAST in the chain, after every whitespace-touching
    normalizer (``letter_spaced_collapse`` / ``vietnamese_text``)
    has already canonicalised the inter-line spacing. The regex
    keys on ``## Page N\\n+<digit>\\n+`` so any non-canonical
    spacing variant the parser ever emits is normalised down to a
    single blank-line separator after the strip.

    Idempotent: a cleaned document has no digit line after the
    header, so the regex finds nothing and the normalizer is a
    no-op on the second pass.
    """

    name: str = "congbobanan_strip_page_noise"
    columns: tuple[str, ...] = ("markdown",)

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        if "markdown" not in df.columns:
            return df
        df["markdown"] = df["markdown"].map(
            lambda v: _strip_page_noise(v) if isinstance(v, str) and v else v,
        )
        return df


__all__ = [
    "JoinSoftWraps",
    "JoinWordBreaks",
    "StripPageNoise",
]
