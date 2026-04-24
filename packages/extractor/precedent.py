"""Vietnamese precedent ("án lệ") extractor.

Layer-2 normalization that maps generic extractions + scraper metadata
+ raw markdown into the ``vila.precedents`` shape declared in
``docs/05-data-infrastructure.md``. Emitted only when
``cfg.extractor.run_site_layer`` is ``True`` (anle portal turns this on;
congbobanan court-judgments portal leaves it off).
"""

from __future__ import annotations

from typing import Any

from packages.extractor.base import (
    ARTICLE_RE,
    DATE_RE,
    PRECEDENT_NUMBER_RE,
    ExtractorAlgorithm,
    GenericRecord,
    PrecedentRecord,
    StatuteRef,
)


class PrecedentExtractor(ExtractorAlgorithm):
    """Maps generic extractions + metadata -> ``vila.precedents`` shape."""

    name = "precedent"

    def extract(
        self,
        doc_id: str,
        markdown: str,
        scraper_metadata: dict[str, Any],
        generic: GenericRecord,
    ) -> PrecedentRecord:
        precedent_number = scraper_metadata.get("precedent_number")
        if not precedent_number:
            m = PRECEDENT_NUMBER_RE.search(markdown)
            if m:
                precedent_number = (
                    f"Án lệ số {m.group('num')}/{m.group('year')}/{m.group('suffix')}"
                )

        adopted_date = _parse_vn_date(scraper_metadata.get("adopted_date"))
        if not adopted_date:
            m = DATE_RE.search(markdown)
            if m:
                adopted_date = _iso_date(m.group("d"), m.group("m"), m.group("y"))

        # Applied article: take the most-referenced statute ref (see
        # :func:`_pick_applied_article` for the tiebreak rules).
        applied = _pick_applied_article(generic.statute_refs, scraper_metadata)

        principle = scraper_metadata.get("principle_text") or _principle_block(markdown)

        return PrecedentRecord(
            doc_id=doc_id,
            precedent_number=precedent_number,
            adopted_date=adopted_date,
            applied_article_code=applied.get("code") if applied else None,
            applied_article_number=applied.get("article") if applied else None,
            applied_article_clause=applied.get("clause") if applied else None,
            principle_text=principle,
            source_case_ref=scraper_metadata.get("source_judgment")
            or scraper_metadata.get("source_case"),
            text_hash=generic.text_hash,
        )


def _pick_applied_article(
    refs: list[StatuteRef],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if metadata.get("applied_article"):
        m = ARTICLE_RE.search(str(metadata["applied_article"]))
        if m:
            return {
                "article": int(m.group("article")),
                "clause": int(m.group("clause")) if m.group("clause") else None,
                "code": m.group("code"),
            }
    if not refs:
        return None
    counts: dict[tuple[int, str | None], int] = {}
    for r in refs:
        key = (r.article, r.code)
        counts[key] = counts.get(key, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])
    article, code = best[0]
    return {"article": article, "clause": None, "code": code}


def _principle_block(markdown: str) -> str | None:
    """Heuristic: find the 'Nội dung án lệ' or 'Nguyên tắc' paragraph."""
    for marker in ("Nội dung án lệ", "Nguyên tắc", "Khái quát"):
        idx = markdown.find(marker)
        if idx >= 0:
            # Take the next ~600 chars of the block.
            tail = markdown[idx + len(marker) : idx + len(marker) + 600]
            tail = tail.strip().lstrip(":").strip()
            if tail:
                return tail
    return None


def _parse_vn_date(value: Any) -> str | None:
    if not value:
        return None
    m = DATE_RE.search(str(value))
    if not m:
        return None
    return _iso_date(m.group("d"), m.group("m"), m.group("y"))


def _iso_date(d: str, m: str, y: str) -> str:
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"


__all__ = ["PrecedentExtractor"]
