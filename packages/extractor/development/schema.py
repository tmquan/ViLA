"""Pydantic schema for the case-development view.

A *case development* is a deterministic, JSON-renderable projection
of a single ban-án's NER record onto the **procedural-development
arc** of the case — the ordered sequence of macro-structural
phases a Vietnamese court judgment passes through, and at each
phase the per-lane (metadata / maindata) entity *delta*:

* which entities **enter** the phase for the first time
  (``*_introduced``), and
* which entities have been seen on a previous phase but reappear
  here (``*_carried``).

This is the second of two complementary projections living under
``packages/extractor``:

* :mod:`packages.extractor.timeline` — date-anchored event-line
  view (*when* did things happen).
* :mod:`packages.extractor.development` — phase-anchored
  development-arc view (*how* the case develops; which lane of
  information grows in which phase).

The two are siblings; neither depends on the other at runtime.
``wiki/DEVELOPMENT.md`` is the source-of-truth spec for the field
names and the determinism contract; this module is the
implementation that the wiki tracks.

See ``wiki/DEVELOPMENT.md`` §§ 3-5 for the schema walk-through,
phase taxonomy, and determinism contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Bump on any change to the builder algorithm that affects the
#: persisted JSON bytes. The version participates in the per-doc
#: cache key so any algorithm edit invalidates only the affected
#: outputs instead of silently shadowing past runs.
#:
#: * ``v1`` — initial implementation: cue-driven phase segmentation
#:   (7 phases: preamble / narrative / investigation / hearing /
#:   reasoning / ruling / signature), literal-substring entity
#:   relocalisation, per-phase (metadata, maindata) by (introduced,
#:   carried) delta lists, deterministic sort by
#:   ``(char_start, type, text)`` within each delta list.
BUILDER_VERSION = "v1"


#: Stable schema_version stamped into the on-disk JSON for
#: downstream consumers that want to gate on shape changes without
#: inspecting the builder version.
#:
#: * ``v1`` — initial development-arc schema (per-phase introduced
#:   / carried entity lists for both NER lanes, plus a minimal
#:   inline ``CaseHeader`` and a stats block).
SCHEMA_VERSION = "v1"


# --------------------------------------------------------------------- types

#: Macro-structural phase id used for routing. The seven phases
#: model the standard sectional layout of a Vietnamese ban-án:
#:
#: * ``preamble`` — header card (case number, court, judges,
#:   parties).
#: * ``narrative`` — "Nội dung vụ án": alleged facts,
#:   defendants / plaintiffs / victims, locations, money, crimes.
#: * ``investigation`` — "Cơ quan điều tra" / "Thụ lý vụ án":
#:   investigation agency, additional facts, witness mentions
#:   (criminal pre-trial / civil case-acceptance).
#: * ``hearing`` — "Tại phiên toà": witness testimony, lawyer
#:   arguments, restated panel.
#: * ``reasoning`` — "Nhận định của Toà án": heavy
#:   ``legal_term`` / ``statute_ref`` density.
#: * ``ruling`` — "Quyết định": operative sentence(s), applied
#:   statutes, costs, appeal window.
#: * ``signature`` — final court mention and signatures.
#:
#: Not every doc has every phase. Phases whose cue is absent are
#: dropped; the degenerate fallback is a single ``preamble``
#: covering the whole document.
PhaseId = Literal[
    "preamble",
    "narrative",
    "investigation",
    "hearing",
    "reasoning",
    "ruling",
    "signature",
]


#: Tuple form of :data:`PhaseId` for runtime iteration / ordering
#: enforcement. The order here is the *canonical procedural order*
#: a ban-án unfolds in; the segmenter never reorders phases.
PHASE_ORDER: tuple[PhaseId, ...] = (
    "preamble",
    "narrative",
    "investigation",
    "hearing",
    "reasoning",
    "ruling",
    "signature",
)


#: NER lane id, mirroring the upstream metadata / maindata
#: partition from :mod:`packages.extractor.ner.schema`. Used as a
#: key inside :class:`Phase` so consumers can iterate the two
#: lanes uniformly.
Lane = Literal["metadata", "maindata"]


# --------------------------------------------------------------------- models


class EntityRef(BaseModel):
    """One entity mention attached to a phase.

    Carries enough context for the downstream UI to display the
    entity and optionally re-link it against the upstream KB:

    * ``type`` — the original NER entity type id (e.g.,
      ``per_judge``, ``statute_ref``).
    * ``text`` — verbatim surface form from the source.
    * ``char_start`` / ``char_end`` — char offsets into the
      NFC-normalised source. ``None`` for entities the locator
      could not place (those go into the unrouted bucket and do
      not appear on any phase).
    * ``kb_link_anchor`` — populated for ``statute_ref`` entities
      that grounded to phapdien (mirror of the NER attribute
      ``linked_article_anchor``).
    * ``kb_link_term_id`` — populated for ``legal_term`` entities
      that grounded to tnpl (mirror of ``linked_term_id``).

    The two KB-link fields are nullable for every other entity
    type; they are persisted alongside the mention so the
    development view is self-contained (a consumer can render
    KB-grounded badges without re-joining to the NER cache).
    """

    model_config = ConfigDict(extra="ignore")

    type: str
    text: str
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    kb_link_anchor: str | None = None
    kb_link_term_id: int | None = None


class CaseHeader(BaseModel):
    """Minimal case-level header.

    Inlined into the development record so a consumer rendering
    one ``CaseDevelopment.json`` does not have to join back to the
    timeline / NER cache for the case-card identifiers.

    Derived from ``record.metadata`` + ``record.summary`` (see
    :func:`packages.extractor.development.build._build_case_header`).
    """

    model_config = ConfigDict(extra="ignore")

    case_number: str | None = None
    court: str | None = None
    case_type: str | None = None
    primary_offence: str | None = None
    judges: list[str] = Field(default_factory=list)
    prosecutors: list[str] = Field(default_factory=list)


class Phase(BaseModel):
    """One macro-structural phase of the case-development arc.

    Each phase covers a contiguous ``[char_start, char_end)`` span
    of the NFC-normalised source. The four entity lists are the
    *per-lane delta* against earlier phases:

    * ``metadata_introduced`` — metadata-lane entities (case
      number, judges, prosecutors, …) seen here for the first
      time.
    * ``metadata_carried``    — metadata-lane entities that were
      already introduced in an earlier phase and reappear here.
    * ``maindata_introduced`` — maindata-lane entities (parties,
      facts, money, statutes, …) seen here for the first time.
    * ``maindata_carried``    — maindata-lane entities already
      introduced earlier that reappear here.

    The "first-time" key is the pair ``(entity.type, entity.text)``
    after NFC normalisation. Lists are sorted deterministically by
    ``(char_start, type, text)`` so the on-disk JSON is byte-stable
    across re-runs.
    """

    model_config = ConfigDict(extra="ignore")

    phase: PhaseId
    cue: str | None = None
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    metadata_introduced: list[EntityRef] = Field(default_factory=list)
    metadata_carried: list[EntityRef] = Field(default_factory=list)
    maindata_introduced: list[EntityRef] = Field(default_factory=list)
    maindata_carried: list[EntityRef] = Field(default_factory=list)


class DevelopmentStats(BaseModel):
    """Per-doc counts useful for filtering / sanity in dashboards.

    Per-phase counts live in :attr:`per_phase` keyed by
    :data:`PhaseId`; the top-level fields are derived sums kept
    for convenience.
    """

    model_config = ConfigDict(extra="ignore")

    n_entities_total: int = 0
    n_entities_routed: int = 0
    n_unrouted: int = 0
    n_phases: int = 0

    n_metadata_introduced: int = 0
    n_metadata_carried: int = 0
    n_maindata_introduced: int = 0
    n_maindata_carried: int = 0

    #: ``{phase_id: n_entities_attached_to_phase}``. Keys appear
    #: only for phases the segmenter actually emitted, so the dict
    #: is also a phase-coverage indicator.
    per_phase: dict[str, int] = Field(default_factory=dict)


class CaseDevelopment(BaseModel):
    """Top-level on-disk record per doc.

    Stamped with both :data:`SCHEMA_VERSION` (shape contract) and
    :data:`BUILDER_VERSION` (algorithm contract), plus the
    upstream NER cache identifiers so consumers can join back to
    the entity record without ambiguity.

    See ``wiki/DEVELOPMENT.md § 4`` for the field-by-field walk.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: str = SCHEMA_VERSION
    builder_version: str = BUILDER_VERSION

    doc_name: str
    source_cache_key: str
    source_kb_version: str
    source_prompt_version: str
    source_input_text_hash: str
    built_at: str

    case_header: CaseHeader = Field(default_factory=CaseHeader)
    phases: list[Phase] = Field(default_factory=list)
    stats: DevelopmentStats = Field(default_factory=DevelopmentStats)


__all__ = [
    "BUILDER_VERSION",
    "PHASE_ORDER",
    "SCHEMA_VERSION",
    "CaseDevelopment",
    "CaseHeader",
    "DevelopmentStats",
    "EntityRef",
    "Lane",
    "Phase",
    "PhaseId",
]
