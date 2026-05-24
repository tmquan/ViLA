"""Per-doc timeline builder — pure function over (cache, source).

Reads one ``entities/cache/<key>.json`` (the canonical NER record
for a doc) plus the source ``md/<doc>.md``, produces one
:class:`packages.extractor.timeline.schema.CaseTimeline`. No LLM
calls; no network; no randomness; deterministic given those two
inputs and :data:`packages.extractor.timeline.schema.BUILDER_VERSION`.

See ``wiki/TIMELINE.md`` for the procedural spec; this file is the
canonical implementation that the wiki tracks.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

from packages.extractor.ner.schema import (
    ExtractedEntity,
    PersistedExtraction,
    section_for,
)
from packages.extractor.timeline.classify import classify_event_kind
from packages.extractor.timeline.cluster import (
    Cluster,
    cluster_by_date_proximity,
)
from packages.extractor.timeline.dates import parse_date_to_anchor
from packages.extractor.timeline.locator import (
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
    TimelineTrack,
    Track,
    track_for_kind,
)

logger = logging.getLogger("packages.extractor.timeline")


# --------------------------------------------------------------------- maps

#: NER entity type → (party role, party kind). Only the subset of
#: types that map to actors. Other types route through their own
#: collectors.
_ENTITY_TO_ACTOR: dict[str, tuple[str, str]] = {
    # substantive (in maindata)
    "per_defendant":  ("defendant",  "person"),
    "per_plaintiff":  ("plaintiff",  "person"),
    "per_victim":     ("victim",     "person"),
    "org_defendant":  ("defendant",  "organization"),
    "org_plaintiff":  ("plaintiff",  "organization"),
    "org_victim":     ("victim",     "organization"),
    # procedural (in metadata)
    "per_judge":      ("judge",      "person"),
    "per_prosecutor": ("prosecutor", "person"),
    "per_lawyer":     ("lawyer",     "person"),
    "per_witness":    ("witness",    "person"),
    "org_court":      ("court",      "organization"),
    "org_agency":     ("agency",     "organization"),
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


def _build_case_header(
    record: PersistedExtraction,
) -> CaseHeader:
    """Aggregate metadata + summary into the per-case card.

    De-duplication is exact-match-on-text after NFC; sort order
    preserved as encountered for byte-stability.
    """
    case_number: str | None = None
    court: str | None = None
    judges: list[str] = []
    prosecutors: list[str] = []
    lawyers: list[str] = []

    seen_judge: set[str] = set()
    seen_pros: set[str] = set()
    seen_lawyer: set[str] = set()

    for ent in record.metadata:
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


def _build_event(
    *,
    cluster: Cluster,
    source_text: str,
    event_id: str,
    track: Track,
    is_ambient: bool,
) -> TimelineEvent:
    """Build a :class:`TimelineEvent` from a cluster."""
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

    when = None
    kind = "ambient"
    span_text = None
    char_start = cluster.char_start
    char_end = cluster.char_end

    if not is_ambient and cluster.anchor is not None:
        anchor_ent = cluster.anchor.entity
        when = parse_date_to_anchor(anchor_ent.text, page=anchor_ent.page)
        kind = classify_event_kind(cluster, source_text=source_text)
        if char_start is not None and char_end is not None:
            # 240-char neighbourhood for UI tooltips.
            a = max(0, char_start - 80)
            b = min(len(source_text), char_end + 80)
            span_text = source_text[a:b]

    return TimelineEvent(
        event_id=event_id,
        track=track,
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


# --------------------------------------------------------------------- entry


def _split_ambient_by_section(
    ambient: Cluster,
) -> tuple[Cluster, Cluster]:
    """Partition an ambient cluster into ``(meta, main)`` sub-clusters.

    Each member is routed by :func:`section_for` on its NER type id —
    metadata-typed entities (``case_number``, ``per_judge``, …) go to
    the procedural ambient bucket; maindata-typed entities (parties,
    money, statute_ref, …) go to the substantive ambient bucket.
    Order within each sub-cluster is preserved.
    """
    meta = Cluster(anchor=None, members=[])
    main = Cluster(anchor=None, members=[])
    for le in ambient.members:
        try:
            sect = section_for(le.entity.type)
        except ValueError:
            sect = "maindata"
        if sect == "metadata":
            meta.members.append(le)
        else:
            main.members.append(le)
    return meta, main


def _finalise_track(
    *,
    track: Track,
    dated_events: list[TimelineEvent],
    ambient_cluster: Cluster,
    source_text: str,
    doc_name: str,
) -> TimelineTrack:
    """Sort, re-id, and box up one track's events + ambient."""
    dated_events.sort(key=lambda e: (
        e.when.sort_key if e.when else "9999-99-99",
        e.char_start if e.char_start is not None else 1 << 30,
    ))
    prefix = "M" if track == "meta" else "X"
    for idx, ev in enumerate(dated_events):
        ev.event_id = f"{doc_name}:{prefix}{idx:03d}"

    ambient_event: TimelineEvent | None = None
    if ambient_cluster.members:
        ambient_event = _build_event(
            cluster=ambient_cluster,
            source_text=source_text,
            event_id=f"{doc_name}:{prefix}A00",
            track=track,
            is_ambient=True,
        )

    n_dated = len(dated_events)
    n_events = n_dated + (1 if ambient_event is not None else 0)
    return TimelineTrack(
        track=track,
        events=dated_events,
        ambient=ambient_event,
        n_events=n_events,
        n_dated=n_dated,
    )


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

    1. Re-localise every NER entity in the source markdown (NFC,
       greedy left-to-right) so each gets a char-offset.
    2. Cluster located entities by date proximity within
       ``cluster_window_chars``.
    3. For each dated cluster: classify the event kind, then route
       to the procedural ``meta`` track or the substantive ``main``
       track via :func:`track_for_kind`.
    4. Split the ambient cluster (un-anchored entities) by NER
       section (``METADATA_TYPES`` vs ``MAINDATA_TYPES``) into per-
       track ambient buckets.
    5. Stamp stable, sortable event ids per track and emit the
       :class:`CaseTimeline` record.
    """
    nfc_source = unicodedata.normalize("NFC", source_text)

    # Locate every entity (metadata + maindata) once. Metadata
    # entities (case_number, judge, court, ...) DO participate in
    # clustering — their proximity to a date in the source is what
    # tells us "Judge A presided at the hearing on date X". The
    # CaseHeader card aggregates them at the case level
    # independently.
    all_entities = list(record.all_entities)
    located = locate_entities(source_text=nfc_source, entities=all_entities)

    dated_clusters, ambient = cluster_by_date_proximity(
        located,
        cluster_window_chars=cluster_window_chars,
    )

    # Build dated events, then partition into meta vs main by event
    # kind. Provisional event_ids are replaced in _finalise_track
    # after we know the final order per track.
    meta_events: list[TimelineEvent] = []
    main_events: list[TimelineEvent] = []
    for cluster in dated_clusters:
        ev = _build_event(
            cluster=cluster,
            source_text=nfc_source,
            event_id="__pending__",
            track="main",  # placeholder; corrected immediately below
            is_ambient=False,
        )
        track = track_for_kind(ev.kind)
        ev.track = track
        if track == "meta":
            meta_events.append(ev)
        else:
            main_events.append(ev)

    meta_ambient, main_ambient = _split_ambient_by_section(ambient)

    meta_track = _finalise_track(
        track="meta",
        dated_events=meta_events,
        ambient_cluster=meta_ambient,
        source_text=nfc_source,
        doc_name=record.doc_name,
    )
    main_track = _finalise_track(
        track="main",
        dated_events=main_events,
        ambient_cluster=main_ambient,
        source_text=nfc_source,
        doc_name=record.doc_name,
    )

    case_header = _build_case_header(record)
    outcome = _build_outcome(record)

    all_events: list[TimelineEvent] = []
    all_events.extend(meta_track.events)
    if meta_track.ambient is not None:
        all_events.append(meta_track.ambient)
    all_events.extend(main_track.events)
    if main_track.ambient is not None:
        all_events.append(main_track.ambient)

    _, n_unloc = location_stats(located)
    stats = TimelineStats(
        n_events=len(all_events),
        n_dated=meta_track.n_dated + main_track.n_dated,
        n_ambient=(
            (1 if meta_track.ambient is not None else 0)
            + (1 if main_track.ambient is not None else 0)
        ),
        n_meta_events=meta_track.n_events,
        n_meta_dated=meta_track.n_dated,
        n_main_events=main_track.n_events,
        n_main_dated=main_track.n_dated,
        n_actors=sum(len(e.actors) for e in all_events),
        n_places=sum(len(e.places) for e in all_events),
        n_money=sum(len(e.money) for e in all_events),
        n_statutes=sum(len(e.statutes) for e in all_events),
        n_terms=sum(len(e.terms) for e in all_events),
        n_crimes=sum(len(e.crimes) for e in all_events),
        n_sentences=sum(len(e.sentences) for e in all_events),
        n_unlocated_entities=n_unloc,
    )

    return CaseTimeline(
        doc_name=record.doc_name,
        source_cache_key=record.cache_key,
        source_kb_version=record.kb_version,
        source_prompt_version=record.prompt_version,
        source_input_text_hash=record.input_text_hash,
        built_at=built_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        case=case_header,
        meta=meta_track,
        main=main_track,
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


__all__ = [
    "aggregate_timelines_jsonl",
    "build_one",
    "build_timeline",
    "list_doc_names",
    "read_canonical_record",
    "read_source_text",
    "write_timeline",
]
