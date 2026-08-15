"""Per-doc timeline builder — pure function over (cache, source).

Reads one ``entities/cache/<key>.json`` (the canonical NER record
for a doc) plus the source ``md/<doc>.md``, produces one
:class:`packages.extractor.timeline.schema.CaseTimeline`. No LLM
calls; no network; no randomness; deterministic given those two
inputs and :data:`packages.extractor.timeline.schema.BUILDER_VERSION`.

See ``wiki/TIMELINE.md`` for the procedural spec; this file is the
canonical implementation that the wiki tracks.

Routing rule (``v3``):

* Every entity's lane is determined by its NER section per
  :func:`packages.extractor.ner.schema.section_for`. METADATA-typed
  entities (``case_number``, ``per_judge``, …) are *logistics* and
  feed the :class:`CaseHeader` card at case scope. They never
  appear as event callouts and never enter the ambient bucket.
* MAINDATA-typed entities (parties, dates, money, statute_ref,
  sentence_*, …) are *development*: when located near a date in
  the source they become an event's callouts; otherwise they go
  into the single ambient bucket.
* The event-kind classifier (``filing`` / ``hearing`` / ``verdict``
  / ``sentence`` / ``fact`` / ``unknown``) keeps running but
  produces a descriptive label only — it does NOT drive lane
  selection any more.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.extractor.ner.schema import (
    METADATA_TYPES,
    ExtractedEntity,
    PersistedExtraction,
    section_for,
)
from packages.extractor.timeline.classify import classify_event_kind
from packages.extractor.timeline.cluster import (
    Cluster,
    cluster_by_date_proximity,
)
from packages.extractor.timeline.datetimes import (
    find_relative_expressions,
    parse_date_to_anchor,
    parse_relative_to_anchor,
)
from packages.extractor.timeline.locator import (
    LocatedEntity,
    locate_entities,
    location_stats,
)
from packages.extractor.timeline.schema import (
    Actor,
    CaseHeader,
    CaseOutcome,
    CaseTimeline,
    MoneyRef,
    PartySummary,
    Place,
    SentenceRef,
    StatuteRef,
    TermRef,
    TimelineEvent,
    TimelineStats,
    WhenAnchor,
)

logger = logging.getLogger("packages.extractor.timeline")


# --------------------------------------------------------------------- maps

#: NER entity type → (party role, party kind) for MAINDATA-section
#: parties only. Procedural personnel (judge, prosecutor, lawyer,
#: witness, court, agency) are NOT mapped here — they live on the
#: case header rather than as event actors. The split between
#: metadata and maindata is owned by
#: :func:`packages.extractor.ner.schema.section_for`.
_ENTITY_TO_ACTOR: dict[str, tuple[str, str]] = {
    "per_defendant":  ("defendant",  "person"),
    "per_plaintiff":  ("plaintiff",  "person"),
    "per_victim":     ("victim",     "person"),
    "org_defendant":  ("defendant",  "organization"),
    "org_plaintiff":  ("plaintiff",  "organization"),
    "org_victim":     ("victim",     "organization"),
}


#: Sentence type → kind for :class:`SentenceRef`.
_SENTENCE_KIND: dict[str, str] = {
    "sentence_prison": "prison",
    "sentence_fine":   "fine",
}


# --------------------------------------------------------------------- helpers


def _entity_to_actor(ent: ExtractedEntity) -> Actor | None:
    rk = _ENTITY_TO_ACTOR.get(ent.type)
    if rk is None:
        return None
    role, kind = rk
    return Actor(role=role, kind=kind, type=ent.type, text=ent.text)


def _entity_to_place(ent: ExtractedEntity) -> Place | None:
    if ent.type in {"loc_province", "loc_district", "loc_commune", "loc_address"}:
        return Place(type=ent.type, text=ent.text)
    return None


def _entity_to_money(ent: ExtractedEntity) -> MoneyRef | None:
    return MoneyRef(text=ent.text) if ent.type == "money" else None


def _entity_to_statute(ent: ExtractedEntity) -> StatuteRef | None:
    if ent.type != "statute_ref":
        return None
    a = ent.attributes
    return StatuteRef(
        text=ent.text,
        linked_anchor=a.linked_article_anchor,
        linked_law_code=a.linked_law_code,
        linked_article_number=a.linked_article_number,
    )


def _entity_to_term(ent: ExtractedEntity) -> TermRef | None:
    if ent.type != "legal_term":
        return None
    return TermRef(text=ent.text, linked_term_id=ent.attributes.linked_term_id)


def _entity_to_sentence(ent: ExtractedEntity) -> SentenceRef | None:
    kind = _SENTENCE_KIND.get(ent.type)
    if kind is None:
        return None
    return SentenceRef(kind=kind, text=ent.text)


def _entity_to_crime(ent: ExtractedEntity) -> str | None:
    return ent.text if ent.type == "crime" else None


def _section_of(ent: ExtractedEntity) -> str:
    """Return ``"metadata"`` or ``"maindata"`` for an entity, defaulting safely.

    The NER schema's :func:`section_for` raises on unknown ids; the
    builder must not fail on schema drift, so we default to
    ``"maindata"`` (kept in the event lane / ambient bucket) on
    unrecognised types.
    """
    try:
        return section_for(ent.type)
    except ValueError:
        return "maindata"


def _dedup_actors(actors: list[Actor]) -> list[Actor]:
    """Preserve first-seen order; dedupe by ``(role, text)``."""
    seen: set[tuple[str, str]] = set()
    out: list[Actor] = []
    for a in actors:
        key = (a.role, a.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


# --------------------------------------------------------------------- header


def _build_case_header(
    *,
    record: PersistedExtraction,
    extra_metadata_entities: list[ExtractedEntity],
) -> CaseHeader:
    """Aggregate all logistics entities into the per-case card.

    ``record.metadata`` is the LLM's metadata partition;
    ``extra_metadata_entities`` collects any METADATA-typed entity
    that the builder also saw via the cluster pre-pass (i.e. it
    appeared near a date in the source). Both flows feed the same
    deduper so a judge mentioned both in the preamble and beside
    a hearing date appears exactly once on the header.

    Substantive parties (the maindata-section ``per_*`` /
    ``org_*`` triple) come from ``record.maindata`` and dedupe on
    ``(role, text)``.
    """
    case_number: str | None = None
    court: str | None = None
    judges: list[str] = []
    prosecutors: list[str] = []
    lawyers: list[str] = []
    witnesses: list[Actor] = []
    agencies: list[str] = []

    seen_judge: set[str] = set()
    seen_pros: set[str] = set()
    seen_lawyer: set[str] = set()
    seen_witness: set[str] = set()
    seen_agency: set[str] = set()

    def _ingest(ent: ExtractedEntity) -> None:
        nonlocal case_number, court
        t, txt = ent.type, ent.text
        if t == "case_number" and case_number is None:
            case_number = txt
        elif t == "org_court" and court is None:
            court = txt
        elif t == "per_judge" and txt not in seen_judge:
            judges.append(txt)
            seen_judge.add(txt)
        elif t == "per_prosecutor" and txt not in seen_pros:
            prosecutors.append(txt)
            seen_pros.add(txt)
        elif t == "per_lawyer" and txt not in seen_lawyer:
            lawyers.append(txt)
            seen_lawyer.add(txt)
        elif t == "per_witness" and txt not in seen_witness:
            witnesses.append(Actor(
                role="witness",
                kind="person",
                type="per_witness",
                text=txt,
            ))
            seen_witness.add(txt)
        elif t == "org_agency" and txt not in seen_agency:
            agencies.append(txt)
            seen_agency.add(txt)

    for ent in record.metadata:
        _ingest(ent)
    # The cluster pre-pass may have surfaced additional METADATA
    # entities the LLM listed under maindata or that appear only
    # near a date in the source — pass them through the same
    # deduper.
    for ent in extra_metadata_entities:
        _ingest(ent)

    parties = PartySummary()
    seen_party: set[tuple[str, str]] = set()
    for ent in record.maindata:
        actor = _entity_to_actor(ent)
        if actor is None or actor.role not in {"defendant", "plaintiff", "victim"}:
            continue
        key = (actor.role, actor.text)
        if key in seen_party:
            continue
        seen_party.add(key)
        bucket = getattr(parties, actor.role + "s")
        bucket.append(actor)

    return CaseHeader(
        case_number=case_number,
        court=court,
        case_type=record.summary.case_type,
        primary_offence=record.summary.primary_offence,
        judges=judges,
        prosecutors=prosecutors,
        lawyers=lawyers,
        witnesses=_dedup_actors(witnesses),
        agencies=agencies,
        parties=parties,
    )


def _build_outcome(record: PersistedExtraction) -> CaseOutcome:
    """Extract the case outcome (summary text + sentences seen in maindata)."""
    sentences: list[SentenceRef] = []
    seen_sent: set[tuple[str, str]] = set()
    for ent in record.maindata:
        sent = _entity_to_sentence(ent)
        if sent is None:
            continue
        key = (sent.kind, sent.text)
        if key in seen_sent:
            continue
        seen_sent.add(key)
        sentences.append(sent)
    return CaseOutcome(
        summary_text=record.summary.outcome,
        applied_statutes=list(record.summary.applied_statutes),
        sentences=sentences,
    )


# --------------------------------------------------------------------- events


def _build_event(
    *,
    cluster: Cluster,
    source_text: str,
    event_id: str,
    is_ambient: bool,
) -> TimelineEvent | None:
    """Build a :class:`TimelineEvent` from a cluster.

    Returns ``None`` if the cluster contributes no MAINDATA callouts
    AND is not the ambient bucket — i.e. a date with only
    procedural personnel near it. Those clusters are silently
    dropped from the events list (the personnel still feed the
    case header at case scope).
    """
    actors: list[Actor] = []
    places: list[Place] = []
    money: list[MoneyRef] = []
    statutes: list[StatuteRef] = []
    terms: list[TermRef] = []
    crimes: list[str] = []
    sentences: list[SentenceRef] = []

    seen_actor: set[tuple[str, str]] = set()
    seen_place: set[tuple[str, str]] = set()
    seen_money: set[str] = set()
    seen_stat: set[str] = set()
    seen_term: set[str] = set()
    seen_crime: set[str] = set()
    seen_sent: set[tuple[str, str]] = set()

    for le in cluster.members:
        ent = le.entity
        if _section_of(ent) != "maindata":
            continue
        if (a := _entity_to_actor(ent)) is not None:
            key = (a.role, a.text)
            if key not in seen_actor:
                actors.append(a)
                seen_actor.add(key)
        if (p := _entity_to_place(ent)) is not None:
            key = (p.type, p.text)
            if key not in seen_place:
                places.append(p)
                seen_place.add(key)
        if (m := _entity_to_money(ent)) is not None and m.text not in seen_money:
            money.append(m)
            seen_money.add(m.text)
        if (s := _entity_to_statute(ent)) is not None and s.text not in seen_stat:
            statutes.append(s)
            seen_stat.add(s.text)
        if (t := _entity_to_term(ent)) is not None and t.text not in seen_term:
            terms.append(t)
            seen_term.add(t.text)
        if (c := _entity_to_crime(ent)) is not None and c not in seen_crime:
            crimes.append(c)
            seen_crime.add(c)
        if (sn := _entity_to_sentence(ent)) is not None:
            key2 = (sn.kind, sn.text)
            if key2 not in seen_sent:
                sentences.append(sn)
                seen_sent.add(key2)

    has_metadata_members = any(
        _section_of(le.entity) == "metadata" for le in cluster.members
    )
    has_maindata_callouts = bool(
        actors or places or money or statutes or terms or crimes or sentences,
    )
    if (
        not is_ambient
        and has_metadata_members
        and not has_maindata_callouts
    ):
        # Purely procedural cluster — judge / prosecutor / court
        # near a date with no other content. The personnel
        # mentions still feed the case header via the parallel
        # pre-pass; they don't need a chronological row.
        return None

    when: WhenAnchor | None = None
    kind: str = "unknown"
    span_text: str | None = None
    char_start = cluster.char_start
    char_end = cluster.char_end

    if not is_ambient and cluster.anchor is not None:
        anchor_le = cluster.anchor
        anchor_ent = anchor_le.entity
        if anchor_le.pre_resolved is not None:
            when = anchor_le.pre_resolved
        else:
            when = parse_date_to_anchor(anchor_ent.text, page=anchor_ent.page)
        kind = classify_event_kind(cluster, source_text=source_text)
        if char_start is not None and char_end is not None:
            a = max(0, char_start - 80)
            b = min(len(source_text), char_end + 80)
            span_text = source_text[a:b]

    return TimelineEvent(
        event_id=event_id,
        when=when,
        kind=kind,  # type: ignore[arg-type]
        actors=actors,
        places=places,
        money=money,
        statutes=statutes,
        terms=terms,
        crimes=crimes,
        sentences=sentences,
        span_text=span_text,
        char_start=char_start,
        char_end=char_end,
    )


# --------------------------------------------------------------------- ambient


def _filter_ambient_to_maindata(ambient: Cluster) -> Cluster:
    """Return a copy of ``ambient`` keeping only MAINDATA-typed members.

    Procedural personnel (METADATA-typed) never enter the ambient
    bucket — they are aggregated into the case header by the
    parallel pre-pass.
    """
    out = Cluster(anchor=None, members=[])
    for le in ambient.members:
        if _section_of(le.entity) == "maindata":
            out.members.append(le)
    return out


# --------------------------------------------------------------------- relatives

_UNRESOLVED_SORT_KEY = "9999-99-99T99:99:99"

#: Placeholder anchor-event-id used in pass 1 of the cross-link.
#: After we assign final event ids, we walk all events and patch
#: ``__src_anchor_<idx>__`` markers to the actual anchor event id.
_ANCHOR_PLACEHOLDER_PREFIX = "__src_anchor_"
_ANCHOR_PLACEHOLDER_SUFFIX = "__"


def _make_anchor_placeholder(idx: int) -> str:
    return f"{_ANCHOR_PLACEHOLDER_PREFIX}{idx}{_ANCHOR_PLACEHOLDER_SUFFIX}"


def _resolve_relatives(
    *,
    located: list[LocatedEntity],
    nfc_source: str,
) -> tuple[list[LocatedEntity], int, int, int]:
    """Walk located entities in source order and resolve relative spans.

    See ``wiki/TIMELINE.md § 3a`` for the algorithm. Unchanged in
    ``v3``; only the surrounding builder collapsed lanes.
    """
    existing_spans = {
        (le.start, le.end)
        for le in located
        if le.start is not None and le.end is not None
    }
    regex_spans = find_relative_expressions(nfc_source)
    synthetic: list[LocatedEntity] = []
    for s, e, _ in regex_spans:
        if (s, e) in existing_spans:
            continue
        raw = nfc_source[s:e]
        ent = ExtractedEntity(type="date_relative", text=raw)
        synthetic.append(LocatedEntity(entity=ent, start=s, end=e))

    combined = list(located) + synthetic
    combined.sort(key=lambda le: (
        le.start if le.start is not None else 1 << 30,
        0 if le.pre_resolved is None else 1,
    ))

    merged: list[LocatedEntity] = []
    current_anchor: WhenAnchor | None = None
    current_anchor_idx: int | None = None
    abs_anchor_counter = 0
    n_total = 0
    n_resolved = 0
    n_unresolved = 0

    for le in combined:
        ent = le.entity
        if ent.type == "date":
            current_anchor = parse_date_to_anchor(ent.text, page=ent.page)
            current_anchor_idx = abs_anchor_counter
            abs_anchor_counter += 1
            merged.append(le)
            continue

        if ent.type == "date_relative":
            n_total += 1
            placeholder = (
                _make_anchor_placeholder(current_anchor_idx)
                if current_anchor_idx is not None
                else None
            )
            resolved = parse_relative_to_anchor(
                ent.text,
                anchor=current_anchor,
                anchor_event_id=placeholder,
                page=ent.page,
            )
            if resolved is not None and resolved.iso is not None:
                n_resolved += 1
                promoted = ExtractedEntity(
                    type="date",
                    text=ent.text,
                    page=ent.page,
                    attributes=ent.attributes,
                )
                merged.append(LocatedEntity(
                    entity=promoted,
                    start=le.start,
                    end=le.end,
                    pre_resolved=resolved,
                ))
            else:
                n_unresolved += 1
                merged.append(le)
            continue

        merged.append(le)

    return merged, n_total, n_resolved, n_unresolved


def _patch_anchor_event_ids(
    *,
    located: list[LocatedEntity],
    events: list[TimelineEvent],
) -> None:
    """Replace ``__src_anchor_<idx>__`` placeholders with real event ids.

    Walks every dated event in the single events list. For absolute
    anchors (``when.is_relative=False``), records the mapping
    ``(char_start of anchor_entity) → event_id``. Then on a second
    pass, for every relative event, finds its anchor's char_start
    by re-walking ``located`` to recover the (anchor_idx → start)
    map, and rewrites ``when.anchor_event_id`` from the placeholder
    to the real id. Placeholders that fail to resolve (e.g. the
    anchor event was clustered into ambient) are cleared to
    ``None`` so on-disk readers don't see internal sentinels.
    """
    anchor_idx_to_start: dict[int, int] = {}
    abs_counter = 0
    for le in located:
        if le.entity.type != "date":
            continue
        if le.pre_resolved is not None:
            continue
        if le.start is not None:
            anchor_idx_to_start[abs_counter] = le.start
        abs_counter += 1

    start_to_event_id: dict[int, str] = {}
    for ev in events:
        if (
            ev.when is not None
            and not ev.when.is_relative
            and ev.char_start is not None
        ):
            start_to_event_id[ev.char_start] = ev.event_id

    for ev in events:
        if ev.when is None or not ev.when.is_relative:
            continue
        placeholder = ev.when.anchor_event_id
        if placeholder is None or not placeholder.startswith(
            _ANCHOR_PLACEHOLDER_PREFIX,
        ):
            continue
        try:
            idx = int(
                placeholder[len(_ANCHOR_PLACEHOLDER_PREFIX):
                            -len(_ANCHOR_PLACEHOLDER_SUFFIX)],
            )
        except ValueError:
            ev.when.anchor_event_id = None
            continue
        anchor_start = anchor_idx_to_start.get(idx)
        if anchor_start is None:
            ev.when.anchor_event_id = None
            continue
        ev.when.anchor_event_id = start_to_event_id.get(anchor_start)


# --------------------------------------------------------------------- entry


def build_timeline(
    *,
    record: PersistedExtraction,
    source_text: str,
    cluster_window_chars: int = 1500,
    built_at: str | None = None,
) -> CaseTimeline:
    """Build a :class:`CaseTimeline` for a single doc.

    Pure function: identical (record, source_text, cluster_window) →
    identical bytes when serialised with sorted-keys JSON. The
    :data:`built_at` argument lets callers pin the timestamp for
    reproducibility tests; if ``None``, the current UTC time is
    stamped (and the bytes will then naturally vary across calls).

    Algorithm overview (see ``wiki/TIMELINE.md § 6``):

    1. NFC-normalise the source markdown.
    2. Re-localise every NER entity in the source (NFC, greedy
       left-to-right) so each gets a char-offset.
    3. Pre-pass — scan the source for relative temporal expressions
       (regex), synthesise ``date_relative`` entities for hits the
       NER missed, then walk the merged stream in source order to
       resolve each ``date_relative`` against the most-recent
       preceding absolute ``date``. Resolved relatives are promoted
       to type ``"date"`` carrying their pre-computed
       :class:`WhenAnchor`; unresolved ones stay as
       ``date_relative`` and flow into the ambient bucket.
    4. Cluster located entities by date proximity within
       ``cluster_window_chars``.
    5. Walk every cluster: split members into header_members
       (METADATA-typed; collected for the case header card) and
       event_members (MAINDATA-typed; the event's callouts). Emit
       at most one :class:`TimelineEvent` per cluster, on the
       single chronological lane. A cluster with no MAINDATA members
       is silently skipped — its date had only logistics around it.
    6. Build the :class:`CaseHeader` from ``record.metadata`` plus
       any METADATA entities the cluster pre-pass surfaced.
    7. Sort emitted events by ``(when.sort_key, char_start)`` and
       stamp ids ``<doc>:E000``, ``E001``, …
    8. Filter the ambient cluster to MAINDATA only; emit it as the
       optional :class:`CaseTimeline.ambient` event.
    9. Patch each resolved relative event's ``anchor_event_id`` to
       its absolute-anchor event id.
    10. Emit the :class:`CaseTimeline` record.
    """
    nfc_source = unicodedata.normalize("NFC", source_text)

    all_entities = list(record.all_entities)
    located = locate_entities(source_text=nfc_source, entities=all_entities)

    located, n_rel_total, n_rel_resolved, n_rel_unresolved = _resolve_relatives(
        located=located,
        nfc_source=nfc_source,
    )

    dated_clusters, ambient = cluster_by_date_proximity(
        located,
        cluster_window_chars=cluster_window_chars,
    )

    # Collect METADATA entities surfaced by the cluster pre-pass —
    # these are entities the locator placed near a date in the
    # source. They may overlap with ``record.metadata`` (typical
    # for a judge mentioned in the preamble); the header builder
    # dedupes by (type, text).
    header_extra: list[ExtractedEntity] = []
    for cluster in dated_clusters:
        for le in cluster.members:
            if le.entity.type in METADATA_TYPES:
                header_extra.append(le.entity)
    for le in ambient.members:
        if le.entity.type in METADATA_TYPES:
            header_extra.append(le.entity)

    events: list[TimelineEvent] = []
    for cluster in dated_clusters:
        ev = _build_event(
            cluster=cluster,
            source_text=nfc_source,
            event_id="__pending__",
            is_ambient=False,
        )
        if ev is None:
            continue
        events.append(ev)

    events.sort(key=lambda e: (
        e.when.sort_key if e.when else _UNRESOLVED_SORT_KEY,
        e.char_start if e.char_start is not None else 1 << 30,
    ))
    for idx, ev in enumerate(events):
        ev.event_id = f"{record.doc_name}:E{idx:03d}"

    main_ambient_cluster = _filter_ambient_to_maindata(ambient)
    ambient_event: TimelineEvent | None = None
    if main_ambient_cluster.members:
        ambient_event = _build_event(
            cluster=main_ambient_cluster,
            source_text=nfc_source,
            event_id=f"{record.doc_name}:EA00",
            is_ambient=True,
        )

    _patch_anchor_event_ids(
        located=located,
        events=events,
    )

    case_header = _build_case_header(
        record=record,
        extra_metadata_entities=header_extra,
    )
    outcome = _build_outcome(record)

    all_events: list[TimelineEvent] = list(events)
    if ambient_event is not None:
        all_events.append(ambient_event)

    _, n_unloc = location_stats(located)
    stats = TimelineStats(
        n_events=len(all_events),
        n_dated=len(events),
        n_ambient=1 if ambient_event is not None else 0,
        n_actors=sum(len(e.actors) for e in all_events),
        n_places=sum(len(e.places) for e in all_events),
        n_money=sum(len(e.money) for e in all_events),
        n_statutes=sum(len(e.statutes) for e in all_events),
        n_terms=sum(len(e.terms) for e in all_events),
        n_crimes=sum(len(e.crimes) for e in all_events),
        n_sentences=sum(len(e.sentences) for e in all_events),
        n_unlocated_entities=n_unloc,
        n_relative_total=n_rel_total,
        n_relative_resolved=n_rel_resolved,
        n_relative_unresolved=n_rel_unresolved,
    )

    return CaseTimeline(
        doc_name=record.doc_name,
        source_cache_key=record.cache_key,
        source_kb_version=record.kb_version,
        source_prompt_version=record.prompt_version,
        source_input_text_hash=record.input_text_hash,
        built_at=built_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        case=case_header,
        events=events,
        ambient=ambient_event,
        outcome=outcome,
        stats=stats,
    )


# --------------------------------------------------------------------- io


def write_timeline(
    *,
    timeline: CaseTimeline,
    output_root: Path,
) -> Path:
    """Write ``<output_root>/timelines/<doc_name>.json`` atomically.

    The bytes are produced with sorted keys + indent=2 so visual
    diffs across runs read cleanly. Atomic write via .tmp + rename.
    """
    out_dir = output_root / "timelines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{timeline.doc_name}.json"
    payload = timeline.model_dump(mode="json")
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(body + "\n", encoding="utf-8")
    tmp.replace(out_path)
    return out_path


def read_canonical_record(
    *,
    canonical_dir: Path,
    doc_name: str,
) -> PersistedExtraction:
    """Load the canonical NER record for a doc.

    ``canonical_dir`` is ``<entities>/canonical/`` — the directory
    of symlinks to the canonical-model cache files materialised by
    the NER pipeline.
    """
    src = canonical_dir / f"{doc_name}.json"
    return PersistedExtraction.model_validate_json(
        src.read_text(encoding="utf-8"),
    )


def read_source_text(*, md_dir: Path, doc_name: str) -> str:
    """Read ``<md_dir>/<doc_name>.md`` and NFC-normalise."""
    body = (md_dir / f"{doc_name}.md").read_text(encoding="utf-8")
    return unicodedata.normalize("NFC", body)


def list_doc_names(canonical_dir: Path) -> list[str]:
    """Return doc names with a canonical extraction, in lex order."""
    return sorted(p.stem for p in canonical_dir.glob("*.json"))


def aggregate_timelines_jsonl(
    *,
    output_root: Path,
    doc_names: list[str],
) -> Path:
    """Concatenate per-doc timelines into ``timelines.jsonl``.

    One JSON line per doc, in lex order. Useful for downstream tools
    that prefer a single streamable artefact.
    """
    timelines_dir = output_root / "timelines"
    out_path = output_root / "timelines.jsonl"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for doc_name in sorted(doc_names):
            src = timelines_dir / f"{doc_name}.json"
            if not src.exists():
                continue
            obj = json.loads(src.read_text(encoding="utf-8"))
            fh.write(json.dumps(obj, ensure_ascii=False, sort_keys=True))
            fh.write("\n")
    tmp.replace(out_path)
    return out_path


def build_one(
    *,
    doc_name: str,
    canonical_dir: Path,
    md_dir: Path,
    output_root: Path,
    cluster_window_chars: int,
    built_at: str | None,
) -> CaseTimeline:
    """Build + persist one timeline. Returns the in-memory object."""
    record = read_canonical_record(canonical_dir=canonical_dir, doc_name=doc_name)
    source_text = read_source_text(md_dir=md_dir, doc_name=doc_name)
    timeline = build_timeline(
        record=record,
        source_text=source_text,
        cluster_window_chars=cluster_window_chars,
        built_at=built_at,
    )
    write_timeline(timeline=timeline, output_root=output_root)
    return timeline


@dataclass
class TimelineBuildStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Per-document timeline build as a Curator stage.

    ``process`` runs :func:`build_one` over the batch's ``doc_name`` column
    (read the canonical NER record + source markdown, cluster events, persist
    ``timelines/<doc>.json``) and returns the in-memory
    :class:`~packages.extractor.timeline.schema.CaseTimeline` objects in an
    object-typed ``timeline`` column. The ``__main__`` driver is a thin wrapper
    that feeds one :class:`DocumentBatch` through this stage — mirroring
    :class:`packages.extractor.ner.extract.NerExtractStage`.
    """

    canonical_dir: Path
    md_dir: Path
    output_root: Path
    cluster_window_chars: int
    built_at: str | None = None
    name: str = "timeline_build"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["doc_name"])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["timeline"])

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df = task.to_pandas().copy()
        df["timeline"] = [
            build_one(
                doc_name=str(d),
                canonical_dir=self.canonical_dir,
                md_dir=self.md_dir,
                output_root=self.output_root,
                cluster_window_chars=self.cluster_window_chars,
                built_at=self.built_at,
            )
            for d in df["doc_name"]
        ]
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


__all__ = [
    "TimelineBuildStage",
    "aggregate_timelines_jsonl",
    "build_one",
    "build_timeline",
    "list_doc_names",
    "read_canonical_record",
    "read_source_text",
    "write_timeline",
]
