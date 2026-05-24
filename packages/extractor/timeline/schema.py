"""Pydantic schema for the case-timeline view.

A *case timeline* is a deterministic, JSON-renderable projection of a
single ban-án's NER record onto **two parallel swimlanes** of dated
events. Mirrors the upstream NER ``metadata`` / ``maindata`` split
(see ``wiki/EXTRACTION.md § 4``):

* :class:`TimelineTrack` ``meta`` — *history + logistics of the
  case*: when it was filed, when each hearing happened, when the
  verdict and sentence were pronounced. The court machinery view.
* :class:`TimelineTrack` ``main`` — *substantive content of the
  case*: when the underlying facts occurred, who was involved, what
  money / statutes / locations entered the record. The "what really
  happened" view.

Each track carries an ordered list of dated events plus an optional
*ambient* bucket for entities that could not be anchored to any
date in the source. Together with the static :class:`CaseHeader`
and :class:`CaseOutcome`, this is everything a visualisation needs
to render two horizontal swimlanes (vis-timeline / react-chrono /
Apache ECharts) without further processing.

The shape is documented in detail in ``wiki/TIMELINE.md`` (§ 3
schema, § 4 determinism contract). This module is the source of
truth for the field names; the wiki tracks it.

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
#: key (see :func:`make_timeline_cache_key`) so any algorithm edit
#: invalidates only the affected outputs instead of silently
#: shadowing past runs.
#:
#: * ``v1`` — initial implementation: char-offset re-localisation in
#:   the source md, date-anchored clustering with a proximity window,
#:   heuristic event-kind classifier (fact / filing / hearing /
#:   verdict / sentence / ambient / unknown), and the **dual-track
#:   split** (procedural ``meta`` vs substantive ``main``) that
#:   mirrors the NER ``METADATA_TYPES`` / ``MAINDATA_TYPES`` partition
#:   in ``packages/extractor/ner/schema.py``.
#: * ``v2`` — add **relative temporal resolution** *and* **time-of-day
#:   resolution**: a regex pre-pass that scavenges ``Trước đó X
#:   ngày``, ``X phút sau``, ``Cùng ngày``, ``Hôm qua``, etc. from
#:   the source markdown, resolves each against the most-recent
#:   preceding absolute date anchor, and re-emits the resolved span
#:   as a synthetic ``date`` member of the located-entity stream.
#:   The absolute date parser is renamed ``datetimes.py`` and is
#:   extended to recognise Vietnamese clock-time surface forms
#:   (``"22 giờ 30 phút ngày 14/3/2023"``, ``"khoảng 22 giờ ngày
#:   14/3/2023"``, ``"lúc 14:25"``); :class:`WhenAnchor` gains
#:   ``iso_time`` / ``iso_datetime`` fields plus
#:   ``is_relative`` / ``anchor_event_id`` / ``magnitude`` /
#:   ``unit`` / ``direction`` / ``iso_max`` provenance fields; the
#:   :class:`TimelineStats` gains the ``n_relative_*`` counters. See
#:   ``wiki/TIMELINE.md § 3a`` for the full spec.
BUILDER_VERSION = "v2"


#: Stable schema_version stamped into the on-disk JSON for downstream
#: consumers that want to gate on shape changes without inspecting
#: the builder version.
#:
#: * ``v1`` — initial dual-track schema (``meta`` / ``main`` lanes,
#:   one ``WhenAnchor`` per event with absolute-date provenance
#:   only).
#: * ``v2`` — :class:`WhenAnchor` extended with **time-of-day** and
#:   **relative-temporal** provenance: ``iso_time`` (``HH:MM[:SS]``
#:   when the surface form carried a clock), ``iso_datetime``
#:   (convenience ``YYYY-MM-DDTHH:MM[:SS]`` when both halves are
#:   populated), ``is_relative`` / ``anchor_event_id`` /
#:   ``magnitude`` / ``unit`` / ``direction`` / ``iso_max`` for
#:   resolved relative spans. ``sort_key`` is widened from
#:   ``"YYYY-MM-DD"`` to ``"YYYY-MM-DDTHH:MM:SS"`` so timed events
#:   sort before untimed events on the same day. :class:`TimelineStats`
#:   gains ``n_relative_total`` / ``n_relative_resolved`` /
#:   ``n_relative_unresolved``. All new fields are nullable /
#:   default-zero so v1 readers degrade gracefully.
SCHEMA_VERSION = "v2"


# --------------------------------------------------------------------- types

EventKind = Literal[
    "fact",      # alleged offence / underlying real-world event
    "filing",    # case opened (sơ thẩm / phúc thẩm filing date)
    "hearing",   # in-court session ("Tại phiên toà")
    "verdict",   # ruling / decision date
    "sentence",  # explicit prison or fine sentence event
    "ambient",   # case-level facts that have no date anchor
    "unknown",   # dated event that the heuristic could not type
]


#: Track id — selects which swimlane of the dual-panel timeline an
#: event lives on. ``"meta"`` is the procedural / court-machinery
#: track (filing, hearings, verdict, sentence). ``"main"`` is the
#: substantive content track (alleged facts, parties, money,
#: statutes). The mapping ``EventKind`` → ``Track`` is fixed in
#: :data:`META_KINDS` / :data:`MAIN_KINDS` below; ``unknown`` events
#: route to ``main`` by default.
Track = Literal["meta", "main"]


#: Event kinds that always live on the procedural ``meta`` track —
#: the *history + logistics* of how the case was processed.
META_KINDS: frozenset[str] = frozenset({"filing", "hearing", "verdict", "sentence"})


#: Event kinds that always live on the substantive ``main`` track —
#: the *content* of the case. The ``unknown`` kind also routes here
#: by default (a date with no clear procedural cue is more often a
#: substantive fact than a logistical one in this corpus).
MAIN_KINDS: frozenset[str] = frozenset({"fact", "unknown"})


def track_for_kind(kind: str) -> Track:
    """Return the swimlane (``"meta"`` or ``"main"``) for an event kind.

    ``ambient`` is intentionally rejected: ambient events are split
    into per-track buckets by the builder based on entity-section
    composition rather than by kind, so callers should never ask for
    a track from a literal ``ambient`` kind.
    """
    if kind in META_KINDS:
        return "meta"
    if kind in MAIN_KINDS:
        return "main"
    raise ValueError(
        f"event kind {kind!r} has no fixed track; ambient events "
        "are split by entity-section composition, not kind",
    )


PartyRole = Literal[
    # substantive (in maindata)
    "defendant", "plaintiff", "victim",
    # procedural (in metadata)
    "judge", "prosecutor", "lawyer", "witness",
    # organisational
    "court", "agency",
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

    * ``iso`` — full ISO date ``YYYY-MM-DD`` if the surface form
      yields a complete date. For *resolved relative anchors* this
      is the absolute date computed from the anchor + delta — the
      relative provenance is preserved in :attr:`is_relative` and
      friends so consumers can render the original phrasing
      alongside the resolved point.
    * ``iso_partial`` — partial ISO ``YYYY-MM`` or ``YYYY`` when the
      day or both the day and month are missing.
    * ``iso_time`` — clock time as ``HH:MM:SS`` (``v2``+) when the
      surface form carried a clock — e.g. ``"22 giờ 30 phút ngày
      14/3/2023"`` populates both ``iso`` and ``iso_time``,
      ``"khoảng 22 giờ ngày 14/3/2023"`` populates ``iso_time =
      "22:00:00"``, ``"14 giờ 25 phút"`` (no date) populates only
      ``iso_time = "14:25:00"``. The relative resolver also writes
      this field when a sub-day delta (``X phút sau`` /
      ``X giờ sau``) is added to a dated + timed anchor.
    * ``iso_datetime`` — convenience full ISO
      ``YYYY-MM-DDTHH:MM:SS`` when both ``iso`` and ``iso_time``
      are populated; ``None`` otherwise. Computed at construction.
    * ``raw`` — the original surface text from the entity.
    * ``page`` — 1-based page number from the entity (may be null
      when the LLM omitted it).
    * ``sort_key`` — lexicographically sortable key used by the
      builder to order events. In ``v2`` the shape is
      ``"YYYY-MM-DDTHH:MM:SS"``: timed events on the same calendar
      day sort before untimed events (which use ``T99:99:99``),
      and partial / unresolved dates sort to the end of the
      track. See
      :func:`packages.extractor.timeline.datetimes.parse_date_to_anchor`
      for the construction.

    Relative-temporal provenance (added in :data:`SCHEMA_VERSION`
    ``v2``; absent on absolute anchors):

    * ``is_relative`` — ``True`` iff this anchor was synthesised by
      resolving a relative expression against a preceding absolute
      date. Renderers can use the flag to draw the marker
      differently (dashed line, lighter colour, …).
    * ``anchor_event_id`` — id of the event whose ``when`` was used
      as the anchor for resolution. ``None`` for absolute anchors.
    * ``magnitude`` / ``unit`` / ``direction`` — the parsed delta
      that produced the resolved date. ``magnitude`` is a float to
      accommodate decimal-comma OCR forms (``"1,2 phút sau"``).
    * ``iso_max`` — upper-bound ISO date for vague magnitudes
      (``"vài ngày sau"``, ``"khoảng 5 tuần sau"``); ``iso`` then
      carries the lower bound and ``iso_max`` carries the upper.
      Single-point resolutions leave ``iso_max`` ``None``.
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
    """A party / personnel mention attached to an event.

    ``role`` is the procedural / substantive role, derived from the
    entity ``type`` (``per_defendant`` → ``defendant`` etc.).
    ``kind`` distinguishes natural persons from legal entities; this
    matters because the v3 NER schema pairs ``per_*`` with ``org_*``
    for the three party roles.
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

    A *dated* event has a non-null :attr:`when`; an *ambient* event
    has ``when is None`` and gathers case-level entities that the
    builder could not anchor to any date in the source. Ambient
    events are still emitted (for completeness) under
    :attr:`TimelineTrack.ambient`.

    The :attr:`track` field is redundant with the parent
    :class:`TimelineTrack.track` — it is stamped on the event so
    flattened views (CSV exports, joined tables) preserve the
    partition without losing information.
    """

    model_config = ConfigDict(extra="ignore")

    event_id: str
    track: Track
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
    """Per-role roster of parties for the case header."""

    model_config = ConfigDict(extra="ignore")

    defendants: list[Actor] = Field(default_factory=list)
    plaintiffs: list[Actor] = Field(default_factory=list)
    victims: list[Actor] = Field(default_factory=list)


class CaseHeader(BaseModel):
    """Case-level header, derived from the metadata + summary.

    Cards on a timeline UI usually show a static header plus the
    event list; this model is the contract for that card.
    """

    model_config = ConfigDict(extra="ignore")

    case_number: str | None = None
    court: str | None = None
    case_type: str | None = None
    primary_offence: str | None = None
    judges: list[str] = Field(default_factory=list)
    prosecutors: list[str] = Field(default_factory=list)
    lawyers: list[str] = Field(default_factory=list)
    parties: PartySummary = Field(default_factory=PartySummary)


class CaseOutcome(BaseModel):
    """Outcome panel — the operative ruling and its statute / sentence backing."""

    model_config = ConfigDict(extra="ignore")

    summary_text: str | None = None
    applied_statutes: list[str] = Field(default_factory=list)
    sentences: list[SentenceRef] = Field(default_factory=list)


class TimelineTrack(BaseModel):
    """One swimlane (procedural ``meta`` or substantive ``main``).

    Each track carries:

    * ``events`` — the dated events on this lane, sorted by
      :attr:`WhenAnchor.sort_key` (then by event id for stability).
    * ``ambient`` — a single optional bucket aggregating entities of
      this track's section that did not anchor to any date in the
      source. ``None`` when there are no orphans for this track.
    * ``n_dated`` / ``n_events`` — convenience counts so
      visualisations can show track totals without iterating.

    The track id (``"meta"`` or ``"main"``) is also stamped onto each
    :class:`TimelineEvent` via ``track`` so consumers that flatten
    both lanes into a single CSV / table can preserve the partition.
    """

    model_config = ConfigDict(extra="ignore")

    track: Track
    events: list[TimelineEvent] = Field(default_factory=list)
    ambient: TimelineEvent | None = None
    n_events: int = 0
    n_dated: int = 0


class TimelineStats(BaseModel):
    """Per-doc counts, useful for filtering / sanity in dashboards.

    The per-track counts (``n_meta_*`` / ``n_main_*``) are the
    primary source of truth; the totals are derived sums kept for
    convenience.
    """

    model_config = ConfigDict(extra="ignore")

    n_events: int = 0
    n_dated: int = 0
    n_ambient: int = 0

    n_meta_events: int = 0
    n_meta_dated: int = 0
    n_main_events: int = 0
    n_main_dated: int = 0

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
    #: substantive ambient bucket so consumers still see the text.
    n_relative_unresolved: int = 0


class CaseTimeline(BaseModel):
    """Top-level on-disk record per doc.

    Two parallel swimlanes:

    * :attr:`meta` — procedural / court-machinery track (history +
      logistics of the case). Events here are filings, hearings,
      verdicts, and sentences.
    * :attr:`main` — substantive content track (what really
      happened). Events here are alleged facts plus any dated
      events the classifier could not pin to a procedural cue.

    The static :class:`CaseHeader` (parties, judges, court) and the
    :class:`CaseOutcome` panel are shared across both swimlanes.

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
    meta: TimelineTrack = Field(default_factory=lambda: TimelineTrack(track="meta"))
    main: TimelineTrack = Field(default_factory=lambda: TimelineTrack(track="main"))
    outcome: CaseOutcome = Field(default_factory=CaseOutcome)
    stats: TimelineStats = Field(default_factory=TimelineStats)


__all__ = [
    "BUILDER_VERSION",
    "MAIN_KINDS",
    "META_KINDS",
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
    "TimelineTrack",
    "Track",
    "WhenAnchor",
    "track_for_kind",
]
