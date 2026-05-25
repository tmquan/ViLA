"""Vietnamese-aware text normalization for the extraction stage.

Vietnamese-language PDFs handed back by the parser stage carry four
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

   The four covered open-syllable diphthongs are ``oa``, ``oe``,
   ``uy``, ``ua`` (matches the upstream undertheseanlp rule set --
   see step 4 below). The ``u``-headed digraphs are guarded by a
   ``(?<![qQ])`` lookbehind so qu-initial syllables stay intact
   (``Quỳnh``, ``Quý``, ``quà``, ``quá``: the ``qu`` digraph is a
   single onset, the vowel after it carries the tone natively, no
   reform happened).

3. PDF-extractor whitespace artefacts: runs of spaces inside words
   ("thà nh phố"), trailing spaces before newlines, and stray
   non-breaking-space (U+00A0) characters. These don't affect token
   identity but inflate regex complexity.

4. Word-level orthographic variants from the pre-/post-1984 spelling
   reform (``công ti / công ty``, ``lí / lý``, ``xẩy / xảy``, …).
   Vendored from `undertheseanlp/text_normalization`_'s
   ``rules.json`` (GPL-3.0) and applied after the tone-mark rewrite.
   Each rule canonicalises the older / dialectal spelling on the
   left to the modern legal-document standard on the right. ``\\b``
   word-boundary regex match (Unicode-aware in Python 3) keeps the
   replacement word-scoped: ``lí`` rewrites in ``lí luận`` but not in
   ``lít``.

   .. _undertheseanlp/text_normalization:
      https://github.com/undertheseanlp/text_normalization

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
    """Old → new orthography map for the four open-syllable diphthongs.

    Vietnamese spelling reform (post-1984):

    * **Old** (pre-1984): tone mark on the TAIL vowel of the digraph.
      E.g. ``Toà`` = T + o + à, ``hoà`` = h + o + à, ``thuỷ`` = th + u + ỷ,
      ``muà`` = m + u + à.
    * **New** (post-1984): tone mark on the HEAD vowel of the digraph.
      E.g. ``Tòa`` = T + ò + a, ``hòa`` = h + ò + a, ``thủy`` = th + ủ + y,
      ``mùa`` = m + ù + a.

    Five tones (grave / acute / hook / tilde / dot-below) × four
    head/tail pairs (o-a, o-e, u-y, u-a) = 20 lowercase digraph
    mappings; the ``u-a`` pair is vendored from
    `undertheseanlp/text_normalization`_'s ``rules.json``. Each
    mapping registers three cases: lowercase, title-case (``Oà``
    word-initial), and ALL-CAPS (``TOÀ ÁN NHÂN DÂN`` letterheads).

    .. _undertheseanlp/text_normalization:
       https://github.com/undertheseanlp/text_normalization
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
    diphthongs = [("o", "a"), ("o", "e"), ("u", "y"), ("u", "a")]
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


def _build_tone_re(table: dict[str, str]) -> re.Pattern[str]:
    """Compile the tone-mark alternation with a ``(?<![qQ])`` guard.

    ``qu`` is a single onset in Vietnamese, so the ``u`` in ``Quỳ``,
    ``Quý``, ``quà``, ``quá`` is NOT the head of a diphthong rime;
    rewriting ``uỳ → ùy`` (or ``uà → ùa``) inside ``Quỳ`` corrupts
    the word. A negative-lookbehind on every ``u``-headed key keeps
    qu-initial syllables intact while still rewriting ``thuỳ → thùy``
    and ``muà → mùa``.

    ``o``-headed digraphs have no analogous trap (Vietnamese has no
    ``qo`` / ``ko`` single-onset that would put ``o`` outside the
    rime), so they alternate without a guard.
    """
    o_keys = sorted(
        (k for k in table if k[:1].lower() == "o"),
        key=len, reverse=True,
    )
    u_keys = sorted(
        (k for k in table if k[:1].lower() == "u"),
        key=len, reverse=True,
    )
    parts: list[str] = []
    if o_keys:
        parts.append("(?:" + "|".join(re.escape(k) for k in o_keys) + ")")
    if u_keys:
        parts.append(
            r"(?<![qQ])(?:"
            + "|".join(re.escape(k) for k in u_keys)
            + ")"
        )
    return re.compile("|".join(parts))


_TONE_RE = _build_tone_re(_TONE_TABLE)

# Word-level orthographic variants -- pre-1984 / dialectal spelling
# on the LEFT, modern legal-document canonical form on the RIGHT.
# Vendored from undertheseanlp/text_normalization (rules.json,
# https://github.com/undertheseanlp/text_normalization, GPL-3.0).
#
# The trailing self-mapping entries are NO-OP guards: ``lẩy bẩy``
# (trembling reduplication) must NOT decay to ``lẩy bảy`` even
# though the bare ``bẩy`` rule fires elsewhere. With ``\b``-anchored
# alternation sorted longest-first, the multi-word no-op wins before
# the single-word rule gets a chance to match. ``tham công tiếc
# việc`` is preserved as a unit because ``công ti`` would otherwise
# look like it should rewrite to ``công ty`` (the bare-``ti`` here
# is a fragment of ``tiếc``, not the company-form suffix).
_WORD_VARIANT_BASE: dict[str, str] = {
    # Pre- vs post-1984 single-word variants
    "công ti": "công ty",
    "lí": "lý",
    "xẩy": "xảy",
    "bẩy": "bảy",
    "gẫy": "gãy",
    # No-op guards for reduplications / set phrases
    "lẩy bẩy": "lẩy bẩy",
    "tham công tiếc việc": "tham công tiếc việc",
}


def _build_word_variant_table(base: dict[str, str]) -> dict[str, str]:
    """Expand each rule into lowercase + Capital + ALLCAPS variants.

    Vietnamese legal documents use either all-lowercase body text or
    ALLCAPS letterheads / section titles; Title Case is rare, and
    a single first-word capitalisation (``Lẩy bẩy``) covers the
    sentence-initial case. Title-casing every word (``Lẩy Bẩy``) is
    intentionally not covered to keep the table small.
    """
    out: dict[str, str] = {}
    for old, new in base.items():
        out[old] = new
        # capitalize() uppercases the first cased character and
        # lowercases the rest -- exactly the "sentence-initial"
        # variant we want.
        out[old.capitalize()] = new.capitalize()
        out[old.upper()] = new.upper()
    return out


_WORD_VARIANT_TABLE: dict[str, str] = _build_word_variant_table(
    _WORD_VARIANT_BASE,
)
# ``\b`` in Python 3's ``re`` defaults to Unicode-aware mode, so it
# correctly identifies word boundaries around Vietnamese diacritic
# characters. Longest-first sort ensures the multi-word no-op guards
# (``lẩy bẩy``, ``tham công tiếc việc``) match before any of their
# sub-words can fire.
_WORD_VARIANT_RE = re.compile(
    r"\b(?:"
    + "|".join(
        re.escape(k)
        for k in sorted(_WORD_VARIANT_TABLE, key=len, reverse=True)
    )
    + r")\b"
)

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
        2. Old → new Vietnamese tone-mark orthography rewrite
           (``Toà → Tòa``, ``thuỷ → thủy``, ``muà → mùa``;
           qu-initial syllables exempt).
        3. Word-level orthographic variant rewrite (``công ti →
           công ty``, ``lí → lý``, …) vendored from
           ``undertheseanlp/text_normalization``.
        4. Whitespace cleanup (tabs → space, collapse runs of
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
    # 2. Tone-mark orthography rewrite (character-level).
    text = _TONE_RE.sub(lambda m: _TONE_TABLE[m.group(0)], text)
    # 3. Word-level orthography rewrite (multi-character word units).
    text = _WORD_VARIANT_RE.sub(
        lambda m: _WORD_VARIANT_TABLE[m.group(0)], text,
    )
    # 4. Whitespace cleanup.
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
