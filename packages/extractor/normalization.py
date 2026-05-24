"""Vietnamese-aware text normalization for the extraction stage.

Vietnamese-language PDFs handed back by the parser stage carry three
classes of artefacts that downstream regex / segment / NER extractors
must deal with otherwise:

1. Decomposed Unicode + mojibake. Tone marks may arrive as combining
   characters (NFD) instead of pre-composed (NFC) glyphs; bytes may
   round-trip through Latin-1 as mojibake. NeMo Curator's
   :class:`~nemo_curator.stages.text.modifiers.UnicodeReformatter`
   (ftfy backend) cleans both with one switch.

2. Old (pre-1984) vs new (post-1984) tone-mark orthography. The
   modern Vietnamese spelling rule places the tone mark on the head
   vowel of an open-syllable diphthong, so

       Toà -> Tòa,  hoà -> hòa,  thuỷ -> thủy,  quý -> quý  (no change)

   PDF text extractors faithfully preserve whichever form the source
   used, so a single corpus mixes both. Canonicalising to the modern
   orthography lets every downstream pattern match a single form.

3. PDF-extractor whitespace artefacts: runs of spaces inside words
   ("thà nh phố"), trailing spaces before newlines, and stray
   non-breaking-space (U+00A0) characters. These don't affect token
   identity but inflate regex complexity.

This module exposes:

* :func:`normalize_text` -- pure function used by the structure /
  generic / precedent extractors before they tokenise.
* :class:`VietnameseTextNormalizer` -- a Curator
  :class:`DocumentModifier` so an upstream pipeline (e.g. parse stage)
  can normalise the markdown column in place via the standard
  :class:`~nemo_curator.stages.text.modifiers.Modify` stage.
"""

from __future__ import annotations

import re
import unicodedata

from ftfy import TextFixerConfig
from nemo_curator.stages.text.modifiers.doc_modifier import DocumentModifier
from nemo_curator.stages.text.modifiers.unicode.unicode_reformatter import (
    UnicodeReformatter,
)

# ----------------------------------------------------- tone-mark table


def _build_tone_table() -> dict[str, str]:
    """Old → new orthography map for the three open-syllable diphthongs.

    Vietnamese spelling reform (post-1984):

    * **Old** (pre-1984): tone mark on the TAIL vowel of the digraph.
      E.g. ``Toà`` = T + o + à, ``hoà`` = h + o + à, ``thuỷ`` = th + u + ỷ.
    * **New** (post-1984): tone mark on the HEAD vowel of the digraph.
      E.g. ``Tòa`` = T + ò + a, ``hòa`` = h + ò + a, ``thủy`` = th + ủ + y.

    Five tones (grave / acute / hook / tilde / dot-below) × three
    head/tail pairs (o-a, o-e, u-y) = 15 lowercase digraph mappings.
    Each maps to its modern equivalent. We also register the title-
    cased form (rare but cheap, covers "Oà-" word-initial) and the
    all-caps form (used in legal letterheads, e.g. "TOÀ ÁN ...").
    """
    # All toned variants in fixed tone order: grave, acute, hook,
    # tilde, dot-below.
    toned = {
        "o": "òóỏõọ",
        "u": "ùúủũụ",
        "a": "àáảãạ",
        "e": "èéẻẽẹ",
        "y": "ỳýỷỹỵ",
    }
    diphthongs = [("o", "a"), ("o", "e"), ("u", "y")]
    table: dict[str, str] = {}
    for head, tail in diphthongs:
        for i in range(5):
            # Old orthography: tone on the tail vowel.
            old_lower = head + toned[tail][i]
            # New orthography: tone on the head vowel.
            new_lower = toned[head][i] + tail
            table[old_lower] = new_lower
            # Title case (e.g. "Oà") -- rare in legal docs but cheap.
            table[old_lower[0].upper() + old_lower[1]] = (
                new_lower[0].upper() + new_lower[1]
            )
            # ALL-CAPS (e.g. "OÀ" in "TOÀ ÁN NHÂN DÂN").
            table[old_lower.upper()] = new_lower.upper()
    return table


_TONE_TABLE: dict[str, str] = _build_tone_table()
_TONE_RE = re.compile("|".join(re.escape(k) for k in _TONE_TABLE))

# PDF artefacts: collapse multi-space runs but preserve newlines.
_INTRA_LINE_WS_RE = re.compile(r"[ \t\u00a0\u2007\u202f\u200b]+")
# Trailing horizontal whitespace before a newline.
_TRAILING_WS_RE = re.compile(r"[ \t\u00a0\u2007\u202f]+(?=\n)")
# Leading horizontal whitespace at the start of a line we want to keep
# (lists may have meaningful indentation), so we DON'T strip leading
# spaces aggressively. Only obvious tab→space conversion.
_TAB_RE = re.compile(r"\t")


# Precomposed UnicodeReformatter (ftfy) instance with NFC + mojibake +
# control-char cleanup. Cheap to instantiate but repeatable, so cache
# at module scope.
_UNICODE_REFORMATTER = UnicodeReformatter(
    config=TextFixerConfig(
        unescape_html=False,           # legal markdown is plain text
        remove_terminal_escapes=True,
        fix_encoding=True,
        restore_byte_a0=True,
        replace_lossy_sequences=True,
        decode_inconsistent_utf8=True,
        fix_c1_controls=True,
        fix_latin_ligatures=True,      # ligature 'ﬁ' → 'fi' helps PDFs
        fix_character_width=False,     # don't touch fullwidth on purpose
        uncurl_quotes=False,           # legal docs use “ ” intentionally
        fix_line_breaks=True,
        fix_surrogates=True,
        remove_control_chars=True,
        normalization="NFC",           # the central knob for our use case
        explain=False,
    ),
)


# ----------------------------------------------------- pure function


def normalize_text(text: str) -> str:
    """Return a canonical-form copy of ``text``.

    Idempotent: applying twice yields the same string. Designed to be
    called by every extractor algorithm (generic / precedent /
    structure) right before it inspects the markdown, so each layer
    can use a single canonical regex form.

    Steps:
        1. NFC + mojibake + control-char cleanup via ftfy.
        2. Old → new Vietnamese tone-mark orthography rewrite.
        3. Whitespace cleanup (tabs → space, collapse runs of
           horizontal whitespace, drop trailing whitespace before
           newlines, strip non-breaking-space).
    """
    if not text:
        return text
    # 1. Unicode normalization.
    text = _UNICODE_REFORMATTER.modify_document(text)
    # Defensive NFC -- ftfy already does it when normalization="NFC",
    # but UnicodeReformatter could be reconfigured by future code, and
    # downstream regexes assume NFC. Guard against a regression.
    text = unicodedata.normalize("NFC", text)
    # 2. Tone-mark orthography rewrite.
    text = _TONE_RE.sub(lambda m: _TONE_TABLE[m.group(0)], text)
    # 3. Whitespace cleanup.
    text = _TAB_RE.sub(" ", text)
    text = _INTRA_LINE_WS_RE.sub(" ", text)
    text = _TRAILING_WS_RE.sub("", text)
    return text


# ----------------------------------------------------- Curator stage


class VietnameseTextNormalizer(DocumentModifier):
    """Curator :class:`DocumentModifier` wrapping :func:`normalize_text`.

    Use via the standard ``Modify(VietnameseTextNormalizer(),
    input_fields="markdown")`` pattern in any pipeline that wants
    normalisation as a first-class stage rather than per-extractor
    side effect.
    """

    name = "vietnamese_text_normalizer"

    def modify_document(self, text: str) -> str:  # type: ignore[override]
        return normalize_text(text)


__all__ = [
    "VietnameseTextNormalizer",
    "normalize_text",
]
