"""Pydantic schema for the case-timeline view.

A *case timeline* is a deterministic, JSON-renderable projection of a
single ban-án's NER record onto a **single chronological lane** of
dated events plus a static :class:`CaseHeader` that holds the
case's logistical roster (case number, court, judges, prosecutors,
lawyers, witnesses, agencies, parties).

Rationale (``v3``): every callout's lane is determined by its NER
section per ``wiki/EXTRACTION.md § 4``:

* :data:`packages.extractor.ner.schema.METADATA_TYPES` — the
  *logistics* of the case: case_number, per_judge, per_prosecutor,
  per_lawyer, per_witness, org_court, org_agency. These are
  STATIC HEADER information about *how* the case is processed; they
  do NOT belong on a chronological lane and are aggregated into
  :class:`CaseHeader` at the case level.
* :data:`packages.extractor.ner.schema.MAINDATA_TYPES` — the
  *development arc* of the case: parties (per_*/org_*), loc_*,
  date, date_relative, money, id_number, plate_number, statute_ref,
  legal_term, crime, sentence_*. These are how the case
  substantively develops; they are the timeline.

The previous :class:`TimelineTrack` (``"meta"`` / ``"main"``)
abstraction is gone. The kind classifier (``filing`` / ``hearing``
/ ``verdict`` / ``sentence`` / ``fact`` / ``unknown``) keeps
running and labels each event purely as an annotation; it does NOT
decide which lane an event lives on.

Determinism: every output that reaches disk is a function of

* the upstream NER cache record's ``cache_key``
  (``cache_key`` already digests ``doc_name``, ``model_id``,
  ``prompt_version``, ``kb_version`` and ``input_text_hash``), and
* the timeline package's :data:`BUILDER_VERSION`.

Re-runs that hit no algorithm change are byte-for-byte identical;
:mod:`tests.unit.test_timeline_determinism` pins this.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

#: Bump on any change to the builder algorithm that affects the
#: persisted JSON bytes. Anything that re-orders events, reclassifies
#: kinds, changes clustering, or rewrites the date parser counts as
#: a builder change. The version participates in the per-doc cache
#: key so any algorithm edit invalidates only the affected outputs
#: instead of silently shadowing past runs.
#:
#: * ``v1`` — initial implementation: char-offset re-localisation in
#:   the source md, date-anchored clustering with a proximity window,
#:   heuristic event-kind classifier (fact / filing / hearing /
#:   verdict / sentence / ambient / unknown), and the **dual-track
#:   split** (procedural ``meta`` vs substantive ``main``) that
#:   mirrored the NER ``METADATA_TYPES`` / ``MAINDATA_TYPES``
#:   partition by routing on event-kind.
#: * ``v2`` — add **relative temporal resolution** *and* **time-of-day
#:   resolution**: a regex pre-pass that scavenges ``Trước đó X
#:   ngày``, ``X phút sau``, ``Cùng ngày``, ``Hôm qua``, etc. from
#:   the source markdown, resolves each against the most-recent
#:   preceding absolute date anchor, and re-emits the resolved span
#:   as a synthetic ``date`` member of the located-entity stream.
#:   The absolute date parser is renamed ``datetimes.py`` and is
#:   extended to recognise Vietnamese clock-time surface forms;
#:   :class:`WhenAnchor` gains time + relative-temporal provenance
#:   fields; :class:`TimelineStats` gains ``n_relative_*`` counters.
#: * ``v3`` — **collapse the meta/main two-lane model into a single
#:   development lane; logistics moves entirely into the static
#:   :class:`CaseHeader`**. The previous track-by-event-kind
#:   dispatch (``track_for_kind``) was wrong — it dragged
#:   substantive callouts (crime, statute_ref, sentence_*) onto the
#:   meta lane whenever the cluster's classifier label was
#:   ``verdict`` / ``hearing`` / ``sentence``. The fix is
#:   structural: each cluster's members are filtered to MAINDATA
#:   types only when forming the event's callouts; metadata-typed
#:   members (judge, prosecutor, court, …) feed the case header at
#:   case scope, never an event. The kind classifier is preserved
#:   as a descriptive label.
BUILDER_VERSION = "v3"


#: Stable schema_version stamped into the on-disk JSON for downstream
#: consumers that want to gate on shape changes without inspecting
#: the builder version.
#:
#: * ``v1`` — initial dual-track schema (``meta`` / ``main`` lanes,
#:   one ``WhenAnchor`` per event with absolute-date provenance
#:   only).
#: * ``v2`` — :class:`WhenAnchor` extended with **time-of-day** and
#:   **relative-temporal** provenance: ``iso_time``, ``iso_datetime``,
#:   ``is_relative`` / ``anchor_event_id`` / ``magnitude`` /
#:   ``unit`` / ``direction`` / ``iso_max``. ``sort_key`` is widened
#:   from ``"YYYY-MM-DD"`` to ``"YYYY-MM-DDTHH:MM:SS"``.
#: * ``v3`` — JSON shape collapses to a single chronological
#:   ``events`` list plus an optional ``ambient`` bucket. The
#:   ``meta`` and ``main`` :class:`TimelineTrack` fields are
#:   removed. :class:`CaseHeader` gains ``witnesses`` and
#:   ``agencies`` fields so the full logistics roster has a home.
#:   :class:`TimelineEvent` drops its ``track`` field.
#:   :class:`TimelineStats` drops the per-lane counters.
SCHEMA_VERSION = "v3"


# --------------------------------------------------------------------- types

#: Closed vocabulary of event-kind labels. The classifier in
#: :mod:`packages.extractor.timeline.classify` stamps each event
#: with one of these — they are purely descriptive labels for the
#: UI to render with kind-specific styling. **Lane assignment does
#: NOT depend on the kind**: every event lives on the single
#: chronological lane regardless of its kind label.
EventKind = Literal[
    "fact",      # alleged offence / underlying real-world event
    "filing",    # case opened (sơ thẩm / phúc thẩm filing date)
    "hearing",   # in-court session ("Tại phiên toà")
    "verdict",   # ruling / decision date
    "sentence",  # explicit prison or fine sentence event
    "unknown",   # dated event that the heuristic could not type
]


#: Closed vocabulary of actor roles that may appear on an
#: :class:`Actor`. Event-level actors are constrained to
#: ``defendant`` / ``plaintiff`` / ``victim`` (the MAINDATA
#: substantive parties); the ``witness`` role is reserved for
#: :class:`CaseHeader.witnesses`, which holds the procedural-side
#: ``per_witness`` mentions deduped by text. The builder enforces
#: the event-vs-header partition; the role union is widened only
#: so the static header card has a typed home for witnesses.
PartyRole = Literal[
    "defendant", "plaintiff", "victim", "witness",
]


# --------------------------------------------------------------------- models


#: Direction tag for relative temporal expressions. ``"before"`` for
#: backward-relative (``Trước đó 3 ngày``), ``"after"`` for
#: forward-relative (``05 phút sau``), ``"same"`` for same-time
#: deixis (``Cùng ngày``).
RelativeDirection = Literal["before", "after", "same"]


#: Closed unit vocabulary for relative deltas. Mirrors the units the
#: regex scanner in :mod:`packages.extractor.timeline.datetimes`
#: accepts. ``None`` for entries the parser could not pin to a
#: specific unit (rare; e.g. ``Hôm sau`` resolves with implicit
#: unit ``ngày``).
RelativeUnit = Literal["giây", "phút", "giờ", "ngày", "tuần", "tháng", "năm"]


class WhenAnchor(BaseModel):
    """Resolved time anchor for an event.

    See ``wiki/TIMELINE.md § 3a`` for the full provenance spec.
    Field semantics are unchanged from ``v2``; only the surrounding
    schema collapsed in ``v3``.
    """

    model_config = ConfigDict(extra="ignore")

    iso: str | None = None
    iso_partial: str | None = None
    iso_time: str | None = None
    iso_datetime: str | None = None
    raw: str
    page: int | None = Field(default=None, ge=1)
    sort_key: str

    is_relative: bool = False
    anchor_event_id: str | None = None
    magnitude: float | None = None
    unit: RelativeUnit | None = None
    direction: RelativeDirection | None = None
    iso_max: str | None = None


class Actor(BaseModel):
    """A party mention attached to an event or to the case header.

    Event actors (``TimelineEvent.actors``) are constrained at
    build time to substantive party types only — ``per_defendant``
    / ``per_plaintiff`` / ``per_victim`` and the matching
    ``org_*`` triple — because the timeline's chronological lane
    is the *development* arc of the case (see
    ``wiki/EXTRACTION.md § 4``).

    The header roster (``CaseHeader.witnesses``) reuses this
    model with ``role = "witness"`` so the same shape can describe
    both event-level actors and header-level witness mentions.
    Procedural personnel that are NOT stored as :class:`Actor`
    (judges, prosecutors, lawyers — natural-person strings; courts
    / agencies — organisation strings) live on
    :class:`CaseHeader` as plain ``list[str]`` for the same reason
    the static card holds a flat roster: there's no per-event
    role variation to capture.
    """

    model_config = ConfigDict(extra="ignore")

    role: PartyRole
    kind: Literal["person", "organization"]
    type: str
    text: str


class Place(BaseModel):
    """A location entity attached to an event."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["loc_province", "loc_district", "loc_commune", "loc_address"]
    text: str


class MoneyRef(BaseModel):
    """A monetary amount attached to an event (raw surface text)."""

    model_config = ConfigDict(extra="ignore")

    text: str


class StatuteRef(BaseModel):
    """A statute citation attached to an event, with optional KB grounding."""

    model_config = ConfigDict(extra="ignore")

    text: str
    linked_anchor: str | None = None
    linked_law_code: str | None = None
    linked_article_number: int | None = None


class TermRef(BaseModel):
    """A legal term attached to an event, with optional tnpl grounding."""

    model_config = ConfigDict(extra="ignore")

    text: str
    linked_term_id: int | None = None


class SentenceRef(BaseModel):
    """A sentence pronouncement (prison or fine) attached to an event."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["prison", "fine"]
    text: str


class TimelineEvent(BaseModel):
    """One row in the case timeline.

    A *dated* event has a non-null :attr:`when`; the *ambient*
    bucket is a single :class:`TimelineEvent` with ``when is None``
    that aggregates maindata entities the builder could not anchor
    to any date in the source. The bucket is exposed as
    :attr:`CaseTimeline.ambient` (singular — there is only one,
    since lanes were collapsed in ``v3``).
    """

    model_config = ConfigDict(extra="ignore")

    event_id: str
    when: WhenAnchor | None
    kind: EventKind

    actors: list[Actor] = Field(default_factory=list)
    places: list[Place] = Field(default_factory=list)
    money: list[MoneyRef] = Field(default_factory=list)
    statutes: list[StatuteRef] = Field(default_factory=list)
    terms: list[TermRef] = Field(default_factory=list)
    crimes: list[str] = Field(default_factory=list)
    sentences: list[SentenceRef] = Field(default_factory=list)

    span_text: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)


class PartySummary(BaseModel):
    """Per-role roster of substantive parties for the case header."""

    model_config = ConfigDict(extra="ignore")

    defendants: list[Actor] = Field(default_factory=list)
    plaintiffs: list[Actor] = Field(default_factory=list)
    victims: list[Actor] = Field(default_factory=list)


class CaseHeader(BaseModel):
    """Static logistics card for the case.

    Holds the procedural / court-side identifiers that don't change
    across events — the home for every ``METADATA_TYPES`` entity
    plus the substantive parties (deduped by text).

    The case header is computed once per doc by aggregating the
    NER ``record.metadata`` list AND any METADATA-typed entity that
    happened to be located near a date in the source (the cluster
    pre-pass). Both flows feed the same dedup; the partition rule
    is :func:`packages.extractor.ner.schema.section_for`.
    """

    model_config = ConfigDict(extra="ignore")

    case_number: str | None = None
    court: str | None = None
    case_type: str | None = None
    primary_offence: str | None = None
    judges: list[str] = Field(default_factory=list)
    prosecutors: list[str] = Field(default_factory=list)
    lawyers: list[str] = Field(default_factory=list)
    witnesses: list[Actor] = Field(default_factory=list)
    agencies: list[str] = Field(default_factory=list)
    parties: PartySummary = Field(default_factory=PartySummary)


class CaseOutcome(BaseModel):
    """Outcome panel — the operative ruling and its statute / sentence backing."""

    model_config = ConfigDict(extra="ignore")

    summary_text: str | None = None
    applied_statutes: list[str] = Field(default_factory=list)
    sentences: list[SentenceRef] = Field(default_factory=list)


class TimelineStats(BaseModel):
    """Per-doc counts, useful for filtering / sanity in dashboards.

    The ``v3`` collapse removed the per-lane counters; everything
    is reported once over the single chronological lane plus the
    optional ambient bucket.
    """

    model_config = ConfigDict(extra="ignore")

    n_events: int = 0
    n_dated: int = 0
    n_ambient: int = 0

    n_actors: int = 0
    n_places: int = 0
    n_money: int = 0
    n_statutes: int = 0
    n_terms: int = 0
    n_crimes: int = 0
    n_sentences: int = 0
    n_unlocated_entities: int = 0

    #: Total number of relative temporal expressions detected in
    #: the source (regex scan + any ``date_relative`` entities the
    #: NER model emitted). Counted before resolution.
    n_relative_total: int = 0
    #: Of :attr:`n_relative_total`, how many were successfully
    #: anchored to a preceding absolute date and resolved into a
    #: concrete :class:`WhenAnchor`.
    n_relative_resolved: int = 0
    #: Of :attr:`n_relative_total`, how many were left unresolved
    #: (no preceding anchor, ambiguous magnitude, or unparseable
    #: surface form). Unresolved expressions are routed to the
    #: ambient bucket so consumers still see the text.
    n_relative_unresolved: int = 0


class CaseTimeline(BaseModel):
    """Top-level on-disk record per doc.

    The case develops along a single chronological lane:

    * :attr:`case` — :class:`CaseHeader`, the static logistics
      roster (case_number, court, judges, prosecutors, lawyers,
      witnesses, agencies, parties).
    * :attr:`events` — :class:`TimelineEvent` list, the development
      arc, sorted by :attr:`WhenAnchor.sort_key` (then by
      ``char_start`` for stability).
    * :attr:`ambient` — optional single :class:`TimelineEvent`
      gathering maindata entities the builder could not anchor to
      any date. ``None`` when there are no orphan substantive
      mentions.
    * :attr:`outcome` — :class:`CaseOutcome` panel; the operative
      ruling and its statute/sentence backing.

    Stamped with both :data:`SCHEMA_VERSION` (shape) and the upstream
    NER cache identifiers so consumers can join back to the entity
    record without ambiguity.
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

    case: CaseHeader = Field(default_factory=CaseHeader)
    events: list[TimelineEvent] = Field(default_factory=list)
    ambient: TimelineEvent | None = None
    outcome: CaseOutcome = Field(default_factory=CaseOutcome)
    stats: TimelineStats = Field(default_factory=TimelineStats)


__all__ = [
    "BUILDER_VERSION",
    "SCHEMA_VERSION",
    "Actor",
    "CaseHeader",
    "CaseOutcome",
    "CaseTimeline",
    "EventKind",
    "MoneyRef",
    "PartyRole",
    "PartySummary",
    "Place",
    "RelativeDirection",
    "RelativeUnit",
    "SentenceRef",
    "StatuteRef",
    "TermRef",
    "TimelineEvent",
    "TimelineStats",
    "WhenAnchor",
]
