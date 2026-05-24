"""KB grounding for the NER pipeline.

Walks the Pydantic :class:`LLMExtraction` produced by the LLM and
attaches grounded identifiers in-place:

* every ``statute_ref`` span is re-parsed with the existing
  :data:`packages.extractor.base.ARTICLE_RE` and looked up in the
  ``legal_dict`` (phapdien) ``(law_short_code, article_number) →
  article_anchor`` index;
* every ``legal_term`` span is NFC-folded and looked up in the
  ``legal_term`` (tnpl) gazetteer (exact lookup first, then a
  rapidfuzz ``WRatio`` fuzzy fallback at score ≥ 92).

Per-doc coverage stats are returned alongside the linked extraction
so the driver can persist them in the output file. Both the entity
mutation and the stats computation are pure functions of the input
:class:`LLMExtraction` + :class:`KnowledgeBase`; given the same
inputs they produce the same outputs (deterministic).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from packages.extractor.base import ARTICLE_RE
from packages.extractor.ner.kb import KnowledgeBase
from packages.extractor.ner.schema import (
    EntityAttributes,
    ExtractedEntity,
    ExtractionStats,
    KbCoverage,
    LLMExtraction,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- result


@dataclass
class GroundingResult:
    """Output of :func:`ground` — the linked extraction + coverage stats."""

    extraction: LLMExtraction
    stats: ExtractionStats


# --------------------------------------------------------------------- helpers


def _parse_statute(span: str) -> tuple[str, int] | None:
    """Re-parse a ``statute_ref`` surface form with :data:`ARTICLE_RE`.

    Returns ``(law_short_code, article_number)`` if both can be
    extracted; ``None`` otherwise (the span is then left without a
    grounded anchor).
    """
    if not span:
        return None
    m = ARTICLE_RE.search(span)
    if m is None:
        return None
    code = m.group("code")
    art = m.group("article")
    if code is None or art is None:
        return None
    return code.upper(), int(art)


def _ground_statute_ref(ent: ExtractedEntity, kb: KnowledgeBase) -> bool:
    """Attach ``linked_article_anchor`` to a ``statute_ref`` entity.

    Returns ``True`` iff the entity got grounded.
    """
    parsed = _parse_statute(ent.text)
    if parsed is None:
        return False
    code, article_no = parsed
    anchor = kb.legal_dict.by_code_article.get((code, article_no))
    if anchor is None:
        return False
    title_meta = kb.legal_dict.by_anchor.get(anchor, {})
    new_attrs = ent.attributes.model_copy(update={
        "linked_article_anchor": anchor,
        "linked_law_code": code,
        "linked_article_number": article_no,
        "linked_article_title": title_meta.get("article_title"),
    })
    ent.attributes = new_attrs
    return True


def _ground_legal_term(ent: ExtractedEntity, kb: KnowledgeBase) -> bool:
    """Attach ``linked_term_id`` (+ score) to a ``legal_term`` entity.

    Returns ``True`` iff the entity got grounded.
    """
    if not ent.text:
        return False
    tid = kb.legal_term.lookup_exact(ent.text)
    score: int | None = None
    if tid is None:
        fuzzy = kb.legal_term.lookup_fuzzy(ent.text)
        if fuzzy is None:
            return False
        tid, score = fuzzy
    new_attrs = ent.attributes.model_copy(update={
        "linked_term_id": tid,
        "linked_match_score": score,
    })
    ent.attributes = new_attrs
    return True


# --------------------------------------------------------------------- driver


def ground(
    extraction: LLMExtraction,
    kb: KnowledgeBase,
) -> GroundingResult:
    """Ground every linkable entity against the bundled KBs.

    Walks both :attr:`metadata` and :attr:`maindata` lists. The two
    grounded entity types (``statute_ref``, ``legal_term``) only
    appear under ``maindata`` per the partition in
    ``wiki/EXTRACTION.md § 4``, but we walk ``metadata`` defensively
    anyway in case an LLM mis-files an entity. Phapdien
    (``legal_dict``) is consulted first for ``statute_ref`` spans,
    then tnpl (``legal_term``) is consulted for ``legal_term`` spans.

    Mutates the entities in place and returns a
    :class:`GroundingResult` that bundles the (now-grounded)
    extraction with coverage stats. Coverage stats use the canonical
    KB names from ``wiki/EXTRACTION.md § 0``.
    """
    legal_dict_total = 0
    legal_dict_linked = 0
    legal_term_total = 0
    legal_term_linked = 0

    for ent in extraction.all_entities:
        if ent.type == "statute_ref":
            legal_dict_total += 1
            if _ground_statute_ref(ent, kb):
                legal_dict_linked += 1
        elif ent.type == "legal_term":
            legal_term_total += 1
            if _ground_legal_term(ent, kb):
                legal_term_linked += 1

    def _pct(num: int, denom: int) -> float:
        if denom == 0:
            return 0.0
        return round(100.0 * num / denom, 2)

    n_metadata = len(extraction.metadata)
    n_maindata = len(extraction.maindata)
    stats = ExtractionStats(
        n_entities=n_metadata + n_maindata,
        n_metadata=n_metadata,
        n_maindata=n_maindata,
        legal_dict=KbCoverage(
            n_total=legal_dict_total,
            n_linked=legal_dict_linked,
            coverage_pct=_pct(legal_dict_linked, legal_dict_total),
        ),
        legal_term=KbCoverage(
            n_total=legal_term_total,
            n_linked=legal_term_linked,
            coverage_pct=_pct(legal_term_linked, legal_term_total),
        ),
    )
    return GroundingResult(extraction=extraction, stats=stats)


__all__ = [
    "EntityAttributes",
    "GroundingResult",
    "ground",
]
