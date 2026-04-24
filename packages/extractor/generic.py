"""Generic (site-agnostic) extractor.

Runs regex + dictionary NER and statute linking over a document's
markdown. Produces one :class:`~packages.extractor.base.GenericRecord`
per document, serialized as JSONL at
``data/<host>/jsonl/generic_extracted.jsonl``.

Layer-2 site normalization (Vietnamese precedents -> vila.precedents)
lives in :mod:`packages.extractor.precedent` and is gated by
``cfg.extractor.run_site_layer``.
"""

from __future__ import annotations

from typing import Iterator

from packages.extractor.base import (
    ARTICLE_RE,
    COURT_RE,
    DATE_RE,
    PRECEDENT_NUMBER_RE,
    Entity,
    ExtractorAlgorithm,
    GenericRecord,
    Relation,
    StatuteRef,
    text_hash,
)


class GenericExtractor(ExtractorAlgorithm):
    """Site-agnostic regex-based NER + statute linker.

    MVP implementation: ships regex-and-dictionary defaults so the
    pipeline runs without any ML dependency. A later patch wires
    ``packages/nlp/`` when that package materializes.
    """

    name = "generic"

    def extract(self, doc_id: str, markdown: str) -> GenericRecord:
        record = GenericRecord(
            doc_id=doc_id,
            text_hash=text_hash(markdown),
            char_len=len(markdown),
        )
        record.entities = list(self._extract_entities(markdown))
        record.statute_refs = list(self._extract_statutes(markdown))
        record.relations = list(self._extract_relations(markdown, record.entities))
        return record

    def _extract_entities(self, text: str) -> Iterator[Entity]:
        for m in DATE_RE.finditer(text):
            yield Entity(tag="DATE", text=m.group(0), start=m.start(), end=m.end())
        for m in COURT_RE.finditer(text):
            yield Entity(
                tag="ORG-COURT",
                text=m.group(0).strip(",. "),
                start=m.start(),
                end=m.end(),
            )
        for m in ARTICLE_RE.finditer(text):
            yield Entity(tag="ARTICLE", text=m.group(0), start=m.start(), end=m.end())
        for m in PRECEDENT_NUMBER_RE.finditer(text):
            yield Entity(tag="PRECEDENT", text=m.group(0), start=m.start(), end=m.end())

    def _extract_statutes(self, text: str) -> Iterator[StatuteRef]:
        for m in ARTICLE_RE.finditer(text):
            year = m.group("year")
            clause = m.group("clause")
            yield StatuteRef(
                article=int(m.group("article")),
                clause=int(clause) if clause else None,
                point=m.group("point"),
                code=m.group("code"),
                year=int(year) if year else None,
                span=(m.start(), m.end()),
            )

    def _extract_relations(
        self, text: str, entities: list[Entity]
    ) -> Iterator[Relation]:
        # MVP: "charge cites_article" relation inferred by proximity.
        charges: list[Entity] = [e for e in entities if e.tag == "CHARGE"]
        articles: list[Entity] = [e for e in entities if e.tag == "ARTICLE"]
        for ch in charges:
            for art in articles:
                if 0 < (art.start - ch.end) < 200:
                    yield Relation(
                        src=ch.text,
                        rel="cites_article",
                        dst=art.text,
                        evidence_span=(ch.start, art.end),
                    )
                    break


__all__ = ["GenericExtractor"]
