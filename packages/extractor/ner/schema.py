"""Pydantic output schema for the LLM NER call.

The LLM returns a JSON object whose top-level shape is
:class:`LLMExtraction`; the parser (``model_validate_json``) hard-fails
on any deviation, which gates malformed runs out of the cache and
into the manifest's ``status="parse_error"`` lane for replay.

Entity-type taxonomy is the 22-class catalogue documented in
``wiki/EXTRACTION.md § 4``. Every type carries a stable
:data:`EntityType` literal id used both as the JSON tag and as a
column key downstream.

Entities are partitioned into two lists per the
``wiki/EXTRACTION.md § 4`` contract:

* :data:`METADATA_TYPES` — procedural / court-side identifiers (case
  number, judge, court, prosecutor, …). Logistic information about
  *how* the case was processed.
* :data:`MAINDATA_TYPES` — substantive content of the case (parties,
  facts, dates, money, locations, statutes, terms, crimes,
  sentences). The "*what* was decided".

The persisted JSON exposes both lists as named keys
(``metadata`` and ``maindata``); the LLM is instructed to emit the
same shape directly so we never need a post-hoc re-classification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------- types

EntityType = Literal[
    # Parties (persons + organisations) — the three role-pairs that
    # can be filled by either a natural person or a legal entity.
    "per_defendant",
    "per_plaintiff",
    "per_victim",
    "org_defendant",
    "org_plaintiff",
    "org_victim",
    # Procedural personnel (always natural persons)
    "per_judge",
    "per_prosecutor",
    "per_lawyer",
    "per_witness",
    # Procedural organisations
    "org_court",
    "org_agency",
    # Locations (Vietnamese admin-unit hierarchy)
    "loc_province",
    "loc_district",
    "loc_commune",
    "loc_address",
    # Time / quantity / identifiers
    "date",
    "date_relative",
    "money",
    "id_number",
    "plate_number",
    "case_number",
    # Legal references
    "statute_ref",
    "legal_term",
    "crime",
    # Sentencing
    "sentence_prison",
    "sentence_fine",
]


#: Tuple form of :data:`EntityType` for runtime iteration / validation.
ENTITY_TYPES: tuple[str, ...] = (
    "per_defendant",
    "per_plaintiff",
    "per_victim",
    "org_defendant",
    "org_plaintiff",
    "org_victim",
    "per_judge",
    "per_prosecutor",
    "per_lawyer",
    "per_witness",
    "org_court",
    "org_agency",
    "loc_province",
    "loc_district",
    "loc_commune",
    "loc_address",
    "date",
    "date_relative",
    "money",
    "id_number",
    "plate_number",
    "case_number",
    "statute_ref",
    "legal_term",
    "crime",
    "sentence_prison",
    "sentence_fine",
)


#: Procedural / court-side entity types. These describe *how* the
#: case was processed, not *what* was decided. Persisted under the
#: ``metadata`` key in the output JSON.
METADATA_TYPES: frozenset[str] = frozenset({
    "case_number",
    "per_judge",
    "per_prosecutor",
    "per_lawyer",
    "per_witness",
    "org_court",
    "org_agency",
})


#: Substantive content of the case — parties (per_*/org_* pairs),
#: facts, locations, dates, money, identifiers, and the legal layer
#: (statute / term / crime / sentence). Persisted under the
#: ``maindata`` key in the output JSON.
MAINDATA_TYPES: frozenset[str] = frozenset({
    "per_defendant",
    "per_plaintiff",
    "per_victim",
    "org_defendant",
    "org_plaintiff",
    "org_victim",
    "loc_province",
    "loc_district",
    "loc_commune",
    "loc_address",
    "date",
    "date_relative",
    "money",
    "id_number",
    "plate_number",
    "statute_ref",
    "legal_term",
    "crime",
    "sentence_prison",
    "sentence_fine",
})


def section_for(entity_type: str) -> str:
    """Return ``"metadata"`` or ``"maindata"`` for an entity type id.

    Raises :class:`ValueError` for unknown ids — this guards against
    silent misclassification when the entity catalogue is extended
    without updating the partition above.
    """
    if entity_type in METADATA_TYPES:
        return "metadata"
    if entity_type in MAINDATA_TYPES:
        return "maindata"
    raise ValueError(
        f"unknown entity type {entity_type!r}; not in METADATA_TYPES "
        f"or MAINDATA_TYPES",
    )


# Sanity check: the partition is a complete cover of ENTITY_TYPES.
assert set(ENTITY_TYPES) == METADATA_TYPES | MAINDATA_TYPES
assert not (METADATA_TYPES & MAINDATA_TYPES)


# --------------------------------------------------------------------- models


class EntityAttributes(BaseModel):
    """Optional per-entity attributes filled by the linker (post-LLM).

    The LLM is *not* asked to fill these — they are populated by
    :mod:`packages.extractor.ner.linker` against the tnpl gazetteer
    and the phapdien article index. Pre-population happens during
    parse only when the LLM volunteers a ``linked_*`` field; the
    linker overwrites either way.
    """

    model_config = ConfigDict(extra="allow")

    linked_term_id: int | None = None
    linked_match_score: int | None = None
    linked_article_anchor: str | None = None
    linked_article_title: str | None = None
    linked_law_code: str | None = None
    linked_article_number: int | None = None


class ExtractedEntity(BaseModel):
    """One typed span produced by the LLM."""

    model_config = ConfigDict(extra="ignore")

    type: EntityType
    text: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)
    page: int | None = Field(default=None, ge=1)
    attributes: EntityAttributes = Field(default_factory=EntityAttributes)


class CaseSummary(BaseModel):
    """Top-level case-level summary; sibling of the entity list."""

    model_config = ConfigDict(extra="ignore")

    case_type: str | None = None
    primary_offence: str | None = None
    applied_statutes: list[str] = Field(default_factory=list)
    outcome: str | None = None


class LLMExtraction(BaseModel):
    """Raw shape returned by the LLM (no ViLA-side metadata yet).

    Entities are split into two lists per the wiki § 4 contract:

    * :attr:`metadata` — procedural / court-side identifiers
      (:data:`METADATA_TYPES`).
    * :attr:`maindata` — substantive case content
      (:data:`MAINDATA_TYPES`).

    The driver in :mod:`packages.extractor.ner.extract` wraps this
    into :class:`PersistedExtraction` after KB linking, adding the
    determinism-contract fields (``cache_key``, ``kb_version``,
    ``model_id``, ``run_id``, …).
    """

    model_config = ConfigDict(extra="ignore")

    metadata: list[ExtractedEntity] = Field(default_factory=list)
    maindata: list[ExtractedEntity] = Field(default_factory=list)
    summary: CaseSummary = Field(default_factory=CaseSummary)

    @property
    def all_entities(self) -> list[ExtractedEntity]:
        """Flat ``metadata + maindata`` list (read-only convenience).

        Useful for iteration without caring about the partition; the
        partition itself is a function of ``entity.type`` and is
        validated by :func:`section_for`.
        """
        return [*self.metadata, *self.maindata]


class KbCoverage(BaseModel):
    """Per-KB grounding coverage stats.

    One instance for ``legal_dict`` (phapdien) and one for
    ``legal_term`` (tnpl); see ``wiki/EXTRACTION.md § 0`` for the
    canonical KB names this pipeline uses.
    """

    model_config = ConfigDict(extra="ignore")

    n_total: int = 0
    n_linked: int = 0
    coverage_pct: float = 0.0


class ExtractionStats(BaseModel):
    """Linker-derived coverage statistics persisted with the result.

    Carries:

    * Per-section counts (``n_metadata`` / ``n_maindata``) so the
      metadata / maindata partition surfaces in the manifest without
      needing to load the cache file.
    * The two KB-coverage blocks named for the canonical KBs in
      ``wiki/EXTRACTION.md § 0``: ``legal_dict`` for phapdien-grounded
      ``statute_ref`` spans (always under ``maindata``), ``legal_term``
      for tnpl-grounded ``legal_term`` spans (always under
      ``maindata``).
    """

    model_config = ConfigDict(extra="ignore")

    n_entities: int = 0
    n_metadata: int = 0
    n_maindata: int = 0
    legal_dict: KbCoverage = Field(default_factory=KbCoverage)
    legal_term: KbCoverage = Field(default_factory=KbCoverage)


class PersistedExtraction(BaseModel):
    """The full record written to ``entities/cache/<cache_key>.json``.

    Mirrors the metadata / maindata partition from
    :class:`LLMExtraction`. Both lists are persisted verbatim so
    downstream consumers can use either the partitioned view
    (``record.metadata`` / ``record.maindata``) or the flat view
    (``record.all_entities``).
    """

    model_config = ConfigDict(extra="ignore")

    doc_name: str
    model_id: str
    prompt_version: str
    kb_version: str
    input_text_hash: str
    cache_key: str
    run_id: str
    cached_at: str

    metadata: list[ExtractedEntity] = Field(default_factory=list)
    maindata: list[ExtractedEntity] = Field(default_factory=list)
    summary: CaseSummary = Field(default_factory=CaseSummary)
    stats: ExtractionStats = Field(default_factory=ExtractionStats)

    @property
    def all_entities(self) -> list[ExtractedEntity]:
        """Flat ``metadata + maindata`` list (read-only convenience)."""
        return [*self.metadata, *self.maindata]


__all__ = [
    "ENTITY_TYPES",
    "MAINDATA_TYPES",
    "METADATA_TYPES",
    "CaseSummary",
    "EntityAttributes",
    "EntityType",
    "ExtractedEntity",
    "ExtractionStats",
    "KbCoverage",
    "LLMExtraction",
    "PersistedExtraction",
    "section_for",
]
