"""Hybrid pypdf -> nemotron-parse parser backend.

Tries the free local :class:`PypdfParser` first. Falls back to the
NIM ``nemotron-parse`` endpoint on two distinct failure modes:

1. **Empty / near-empty local output** -- the tell-tale sign of an
   image-only scan that pypdf cannot extract text from. ``min_chars``
   (default 50) gates this; a stray header/footer ("Page 1 of 3")
   doesn't count as real content.

2. **Catastrophically lossy local output** -- pypdf returned plenty
   of text, but the PDF's embedded font lacks ToUnicode entries for
   most glyphs and the output is mostly unreadable (codepoints in
   the wrong order, tone marks scattered, words shredded into 1-2
   character fragments). The detector flags the fraction of
   lowercase 1-2-char ASCII fragments appearing in word context;
   ``max_lossy_score`` (default 0.05) routes catastrophically-lossy
   docs (~6% of the congbobanan corpus) to OCR. Setting
   ``max_lossy_score=1.0`` disables this branch.

   **Known limitation.** The detector catches *catastrophic* font
   corruption -- whole-document garble like ``"QU N LÊ CHÂNẬ"`` and
   ``"T n Cường"`` repeated throughout the body. It is NOT
   surgical: a document where only a few tone marks are dropped
   (e.g. ``"đấu"`` -> ``"đ u"`` in two paragraphs of an otherwise
   clean 2000-word body) scores in the same band as healthy text
   and stays on the local backend. Cleaning up those mild cases
   requires either a glyph-level signal that ``pypdf`` does not
   currently expose, or running the whole corpus through OCR --
   neither is justified at corpus scale.

Output shape is the :class:`ParserAlgorithm` contract (``pages``,
``markdown``, ``confidence``), so downstream stages (``PdfParseStage``
and onward) never know which backend produced a given document. A
``parser_backend`` key is added to the response for operational
traceability but is not required by the schema. ``local_lossy_score``
is always attached so operators can post-hoc audit threshold tuning.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from packages.parser.base import ParserAlgorithm

logger = logging.getLogger(__name__)

# Word tokenizer used by ``lossy_score``. Unicode-aware ``\w`` so
# Vietnamese tone-marked vowels count as word characters, not
# token separators.
_WORD_TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)

# Match a lowercase 1-2-letter ASCII fragment sandwiched between
# non-whitespace neighbours (i.e. embedded in a body of text, not
# at line edges or alone on a line). Anchoring on lowercase is the
# critical false-positive guard: legitimate anonymized initials in
# Vietnamese legal docs ("Đặng Đức H", "bị cáo M") are uppercase
# and don't trip this signal; the glyph-drop relics ("đ u" from
# "đấu", "t ng" from "tổng", "ra" from a 2-char ASCII spew) are
# overwhelmingly lowercase because the source glyphs were
# lowercase Vietnamese vowels.
_LOSSY_FRAGMENT_RE = re.compile(r"(?<=\S)\s+([a-z]{1,2})\s+(?=\S)")


def lossy_score(markdown: str) -> float:
    """Fraction of word-tokens that are lowercase 1-2-char ASCII fragments
    embedded in body text.

    Calibrated against a 500-doc random sample of
    ``data/congbobanan.toaan.gov.vn/md/``:

    =====  =================  =================================
    p-tile lossy_score        regime
    =====  =================  =================================
    p50    0.016              healthy
    p75    0.022              healthy
    p90    0.031              healthy / mildly noisy
    p95    0.088              catastrophic font corruption
    p99    0.227              total garble
    p100   0.303              unreadable
    =====  =================  =================================

    A threshold of 0.05 cleanly separates the long healthy tail
    (~94% of docs) from the catastrophic regime (~6%) where text
    is dominated by short glyph-drop fragments like ``"QU N LÊ"``,
    ``"T H GIA"``, ``"do an"``. The signal anchors on **lowercase**
    short ASCII tokens so it ignores anonymized party initials
    (``"Đặng Đức H"``, ``"bị cáo M"``) which are always uppercase,
    and it requires the fragment to sit between non-whitespace
    neighbours so line headers and structural markers don't
    inflate the score.

    Returns 0.0 on empty input.
    """
    if not markdown:
        return 0.0
    total = len(_WORD_TOKEN_RE.findall(markdown))
    if total == 0:
        return 0.0
    return len(_LOSSY_FRAGMENT_RE.findall(markdown)) / total


class HybridParser(ParserAlgorithm):
    """pypdf first, nemotron-parse fallback on empty / near-empty / lossy output."""

    runtime = "hybrid"

    def __init__(
        self,
        local: ParserAlgorithm,
        nim: ParserAlgorithm,
        *,
        min_chars: int = 50,
        max_lossy_score: float = 0.05,
    ) -> None:
        self.local = local
        self.nim = nim
        self._min_chars = int(min_chars)
        self._max_lossy_score = float(max_lossy_score)
        # Advertise both model IDs so the downstream ``parser_model``
        # column captures which backends are in play for this site.
        self.model_id = f"{local.model_id}+{nim.model_id}"

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        local_result = self.local.parse(
            pdf_bytes, preserve_tables=preserve_tables
        )
        local_md = str(local_result.get("markdown") or "").strip()

        local_len = len(local_md)
        local_score = lossy_score(local_md)
        # Local output is acceptable iff it's long enough AND not
        # lossy. Either failure mode routes to NIM OCR.
        long_enough = local_len >= self._min_chars
        below_lossy = local_score <= self._max_lossy_score
        if long_enough and below_lossy:
            local_result["parser_backend"] = getattr(
                self.local, "runtime", "local"
            )
            local_result["local_lossy_score"] = local_score
            return local_result

        if long_enough and not below_lossy:
            reason = (
                f"lossy_score={local_score:.3f} > "
                f"threshold={self._max_lossy_score:.3f} "
                f"(local produced {local_len} chars but the embedded "
                f"font appears to drop tone-marked glyphs)"
            )
        else:
            reason = (
                f"local produced {local_len} chars "
                f"(threshold={self._min_chars})"
            )

        logger.info(
            "HybridParser: %s; falling back to NIM %s",
            reason, self.nim.model_id,
        )
        try:
            nim_result = self.nim.parse(
                pdf_bytes, preserve_tables=preserve_tables
            )
        except Exception as exc:
            logger.error(
                "HybridParser: NIM fallback failed (%s: %s); "
                "keeping local's %d-char output",
                type(exc).__name__, exc, local_len,
            )
            local_result["parser_backend"] = getattr(
                self.local, "runtime", "local"
            )
            local_result["nim_fallback_error"] = f"{type(exc).__name__}: {exc}"
            local_result["local_lossy_score"] = local_score
            return local_result

        nim_md = str(nim_result.get("markdown") or "").strip()
        if not nim_md:
            logger.warning(
                "HybridParser: both local and NIM returned empty markdown "
                "(this PDF is unreadable; probably corrupted or locked)"
            )
        nim_result["parser_backend"] = getattr(self.nim, "runtime", "nim")
        nim_result["local_lossy_score"] = local_score
        return nim_result


__all__ = ["HybridParser", "lossy_score"]
