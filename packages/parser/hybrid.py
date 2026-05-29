"""Hybrid pypdf -> NIM/VLM fallback parser backend.

Tries the free local :class:`PypdfParser` first. Falls back to a
configurable OCR-capable client (cloud nemotron-parse or self-hosted
Qwen3.6+MTP / Nemotron-Omni vLLM, wired by
:func:`packages.parser.stage.build_parser` from
``cfg.parser.hybrid_fallback_runtime``) on three distinct failure
modes:

1. **Empty / near-empty local output** -- the tell-tale sign of an
   image-only scan that pypdf cannot extract text from. ``min_chars``
   (default 50) gates this; a stray header/footer ("Page 1 of 3")
   doesn't count as real content. Whole-document fallback.

2. **Catastrophically lossy local output** -- pypdf returned plenty
   of text, but the PDF's embedded font lacks ToUnicode entries for
   most glyphs and the output is mostly unreadable (codepoints in
   the wrong order, tone marks scattered, words shredded into 1-2
   character fragments). The detector flags the fraction of
   lowercase 1-2-char ASCII fragments appearing in word context;
   ``max_lossy_score`` (default 0.05) routes catastrophically-lossy
   docs (~6% of the congbobanan corpus) to OCR. Setting
   ``max_lossy_score=1.0`` disables this branch. Whole-document
   fallback.

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

3. **Mixed digital/scanned PDF (Case D, per-page surgical)** --
   the document as a whole is long enough and below the lossy
   threshold (so modes 1 + 2 keep it on local), but at least one
   individual page is empty while at least one other page is
   non-empty: a typical Vietnamese court PDF where the body text is
   digital but a stamped signature page or scanned exhibit slipped
   in image-only. Re-rasterizes only the empty pages from the
   original ``pdf_bytes`` via
   :meth:`Qwen36OmniClient.parse_single_page` (or
   :meth:`NemotronOmniClient.parse_single_page` on rollback) and
   splices the OCR'd pages back into ``local_result["pages"]``,
   leaving the digital pages untouched. Activates iff the fallback
   client exposes a ``parse_single_page`` method (the legacy
   nemotron-parse NIM client does not, so it gracefully degrades to
   modes 1 + 2 only). Opt out with ``surgical_pages=False``.

Output shape is the :class:`ParserAlgorithm` contract (``pages``,
``markdown``, ``confidence``), so downstream stages (``PdfParseStage``
and onward) never know which backend produced a given document. A
``parser_backend`` key is added to the response for operational
traceability but is not required by the schema. ``local_lossy_score``
is always attached so operators can post-hoc audit threshold tuning.
``surgical_pages_recovered`` is attached when mode (3) fires, listing
the zero-based page indices that received OCR.
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
    """pypdf first, OCR fallback on empty / near-empty / lossy / mixed output.

    The OCR fallback client is wired by
    :func:`packages.parser.stage.build_parser` from
    ``cfg.parser.hybrid_fallback_runtime`` (default ``qwen3_6_omni``;
    valid values include ``nim`` for the cloud nemotron-parse NIM,
    ``nemotron_omni`` for the self-hosted Nemotron-3 Nano Omni NIM,
    and ``qwen3_6_omni`` for the self-hosted Qwen3.6-27B-FP8 vLLM).
    The constructor parameter is named ``nim`` for backward
    compatibility -- it is just a generic fallback slot and does not
    require an actual NVIDIA NIM endpoint.
    """

    runtime = "hybrid"

    def __init__(
        self,
        local: ParserAlgorithm,
        nim: ParserAlgorithm,
        *,
        min_chars: int = 50,
        max_lossy_score: float = 0.05,
        surgical_pages: bool = True,
    ) -> None:
        self.local = local
        self.nim = nim
        self._min_chars = int(min_chars)
        self._max_lossy_score = float(max_lossy_score)
        self._surgical_pages = bool(surgical_pages)
        # Advertise both model IDs so the downstream ``parser_model``
        # column captures which backends are in play for this site.
        self.model_id = f"{local.model_id}+{nim.model_id}"

    def parse(
        self,
        pdf_bytes: bytes,
        *,
        preserve_tables: bool = True,
    ) -> dict[str, Any]:
        """Run the local parser first, route to OCR per the failure modes.

        Three failure modes route to OCR (see module docstring for
        the why):

        (a) ``local_md`` length below ``min_chars`` -- whole-document
            fallback. Image-only / scan-only PDFs.
        (b) ``lossy_score(local_md) > max_lossy_score`` -- whole-
            document fallback. Catastrophic font corruption (~6% of
            the congbobanan corpus).
        (c) Mode (a) and (b) both pass at the document level, but at
            least one ``pages[i].markdown`` is empty/whitespace and at
            least one OTHER page has content -- per-page surgical
            fallback (Case D in wiki/PARSING.md § 4). Re-rasterizes
            ONLY the empty pages from ``pdf_bytes`` via
            ``self.nim.parse_single_page(...)``, splices each OCR'd
            page back into ``local_result["pages"]`` at the original
            index, and rebuilds ``local_result["markdown"]`` so it
            re-includes the previously-missing content. Activates iff
            ``surgical_pages=True`` (default) AND the fallback client
            exposes ``parse_single_page`` (the legacy nemotron-parse
            NIM client does not). The list of recovered indices is
            attached as ``surgical_pages_recovered`` for audit;
            single-page fallback failures are logged and the page
            stays empty (the rest of the document still ships).
        """
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
            # Mode (c): document-level routing wants to keep the
            # local result, but check for per-page Case-D mixing
            # (some empty pages alongside non-empty ones).
            if self._surgical_pages and hasattr(
                self.nim, "parse_single_page"
            ):
                self._apply_surgical_fallback(local_result, pdf_bytes)
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

    # ------------------------------------------------------ internals

    def _apply_surgical_fallback(
        self,
        local_result: dict[str, Any],
        pdf_bytes: bytes,
    ) -> None:
        """Splice OCR'd markdown into empty pages of ``local_result``.

        Mutates ``local_result`` in place. No-op when no page is empty
        or when every page is empty (the latter is Case C and was
        already handled by mode (a) above). On per-page failure, the
        page stays empty and the rest of the doc still ships.
        """
        pages = list(local_result.get("pages") or [])
        if not pages:
            return
        empty_idx: list[int] = []
        any_non_empty = False
        for i, p in enumerate(pages):
            md = str(p.get("markdown") or "").strip()
            if md:
                any_non_empty = True
            else:
                empty_idx.append(i)
        # Mixed iff there is at least one empty page AND at least one
        # non-empty page. All-empty docs are Case C and are already
        # handled by the whole-doc routing above.
        if not empty_idx or not any_non_empty:
            return

        logger.info(
            "HybridParser: Case-D surgical fallback on %d/%d empty "
            "page(s) via %s",
            len(empty_idx), len(pages), self.nim.model_id,
        )
        recovered: list[int] = []
        for i in empty_idx:
            try:
                page_result = self.nim.parse_single_page(pdf_bytes, i)
            except Exception as exc:
                logger.warning(
                    "HybridParser: surgical OCR for page %d failed "
                    "(%s: %s); leaving page empty",
                    i + 1, type(exc).__name__, exc,
                )
                continue
            ocr_md = str(page_result.get("markdown") or "").strip()
            # Preserve the existing page record's keys (e.g. blocks)
            # but overwrite markdown + page_number with the OCR'd
            # values. Keep blocks empty -- the legacy local pages
            # already stored an empty list, and the OCR client does
            # not return a layout block list from the per-page API.
            new_page = dict(pages[i])
            new_page["page_number"] = page_result.get(
                "page_number", i + 1
            )
            new_page["markdown"] = ocr_md
            new_page["blocks"] = (
                [{"type": "Text", "text": ocr_md, "bbox": {}}]
                if ocr_md
                else []
            )
            pages[i] = new_page
            if ocr_md:
                recovered.append(i)

        # Rebuild the consolidated markdown to mirror the existing
        # pypdf join format (``"\n\n".join(f"## Page {i}\n\n{md}")``
        # for non-empty pages only). Verified against
        # packages/parser/pypdf.py and
        # packages/parser/qwen3_6_omni.py: both build the same
        # ``## Page N``-prefixed body and skip empty slots.
        md_parts: list[str] = []
        for p in pages:
            md = str(p.get("markdown") or "").strip()
            if md:
                page_num = p.get("page_number")
                md_parts.append(f"## Page {page_num}\n\n{md}")
        local_result["pages"] = pages
        local_result["markdown"] = "\n\n".join(md_parts)
        local_result["surgical_pages_recovered"] = recovered


__all__ = ["HybridParser", "lossy_score"]
