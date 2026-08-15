"""Per-doc case-development builder — pure function over (record, source).

Reads one canonical NER record + the source ``md/<doc>.md`` and
produces one :class:`packages.extractor.development.schema.CaseDevelopment`.
No LLM call, no network, no randomness — pure function of
``(record, source_text, BUILDER_VERSION)``.

The development package is a sibling of the timeline package. It
does NOT import any code from :mod:`packages.extractor.timeline`
at runtime; the only NER dependency is the *schema* in
:mod:`packages.extractor.ner.schema` (the data contract).

See ``wiki/DEVELOPMENT.md`` for the procedural spec; this module
is the canonical implementation that the wiki tracks.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.extractor.development.schema import (
    BUILDER_VERSION,
    SCHEMA_VERSION,
    CaseDevelopment,
    CaseHeader,
    DevelopmentStats,
    EntityRef,
    Lane,
    Phase,
    PhaseId,
)
from packages.extractor.development.segmenter import PhaseSpan, segment_phases
from packages.extractor.ner.schema import (
    ExtractedEntity,
    PersistedExtraction,
    section_for,
)

logger = logging.getLogger("packages.extractor.development")


# --------------------------------------------------------------------- locator
#
# We re-implement a tiny literal-substring locator here rather than
# importing from packages.extractor.timeline.locator: the development
# package stays independent of the timeline package at runtime
# (sibling, not child).


@dataclass(frozen=True)
class _Located:
    entity: ExtractedEntity
    char_start: int | None
    char_end: int | None


def _locate_entities(
    *, nfc_source: str, entities: Iterable[ExtractedEntity],
) -> list[_Located]:
    """Greedy left-to-right NFC literal-substring locator.

    Identical semantics to the timeline locator: walk entities in
    persisted order, find the first occurrence at-or-after a
    global cursor, advance the cursor on hit. Misses fall back to
    a search from offset 0 (without advancing the cursor) so
    entities the LLM emitted out of document order — typical for
    case-header entities — still get located. Unfindable entities
    are recorded with ``(None, None)`` and excluded from routing.
    """
    located: list[_Located] = []
    cursor = 0
    for ent in entities:
        needle = unicodedata.normalize("NFC", ent.text)
        if not needle:
            located.append(_Located(entity=ent, char_start=None, char_end=None))
            continue
        idx = nfc_source.find(needle, cursor)
        if idx == -1:
            idx = nfc_source.find(needle, 0)
            if idx == -1:
                located.append(_Located(entity=ent, char_start=None, char_end=None))
                continue
            end = idx + len(needle)
            located.append(_Located(entity=ent, char_start=idx, char_end=end))
            continue
        end = idx + len(needle)
        located.append(_Located(entity=ent, char_start=idx, char_end=end))
        cursor = end
    return located


# --------------------------------------------------------------------- helpers


def _entity_ref(loc: _Located) -> EntityRef:
    ent = loc.entity
    attrs = ent.attributes
    return EntityRef(
        type=ent.type,
        text=ent.text,
        char_start=loc.char_start,
        char_end=loc.char_end,
        kb_link_anchor=attrs.linked_article_anchor,
        kb_link_term_id=attrs.linked_term_id,
    )


def _build_case_header(record: PersistedExtraction) -> CaseHeader:
    """Aggregate metadata + summary into the per-case card.

    De-duplication is exact-match-on-text after NFC; sort order
    preserved as encountered for byte-stability — mirroring the
    timeline builder's case-header convention so the two artifacts
    show the same identifiers.
    """
    case_number: str | None = None
    court: str | None = None
    judges: list[str] = []
    prosecutors: list[str] = []

    seen_judge: set[str] = set()
    seen_pros: set[str] = set()

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

    return CaseHeader(
        case_number=case_number,
        court=court,
        case_type=record.summary.case_type,
        primary_offence=record.summary.primary_offence,
        judges=judges,
        prosecutors=prosecutors,
    )


def _route_into_phase(
    char_start: int,
    spans: list[PhaseSpan],
) -> int | None:
    """Return index of the phase span covering ``char_start``, or ``None``.

    Binary-search-friendly span layout (contiguous, ordered, half-
    open intervals) but we use a linear scan because the typical
    span count is ≤ 7. None is returned when the offset is outside
    every span (only possible when the entity offset is itself
    ``None`` — i.e. the entity was unlocated).
    """
    for i, span in enumerate(spans):
        if span.char_start <= char_start < span.char_end:
            return i
        # Boundary case: the very last span is half-open at the
        # right; an offset exactly equal to len(source) is treated
        # as belonging to the last span.
        if i == len(spans) - 1 and char_start == span.char_end:
            return i
    return None


def _lane_for(entity_type: str) -> Lane:
    """Return the NER lane (``"metadata"`` / ``"maindata"``) for a type.

    Falls back to ``"maindata"`` if the ner schema rejects the
    type — that branch is only hit when the upstream NER schema
    is extended without updating the partition assertion in
    :mod:`packages.extractor.ner.schema`, so it should never fire
    in practice. We tolerate it here to keep the development
    builder resilient against in-flight schema evolution.
    """
    try:
        return section_for(entity_type)  # type: ignore[return-value]
    except ValueError:
        return "maindata"


def _sort_refs(refs: list[EntityRef]) -> list[EntityRef]:
    """Deterministic in-place sort by ``(char_start, type, text)``.

    Entities with ``char_start = None`` cannot appear here (they
    are routed to the unrouted bucket before we ever build a Phase
    list); we still guard with a sentinel for defensive coding.
    """
    refs.sort(key=lambda r: (
        r.char_start if r.char_start is not None else 1 << 30,
        r.type,
        r.text,
    ))
    return refs


# --------------------------------------------------------------------- entry


def build_development(
    *,
    record: PersistedExtraction,
    source_text: str,
    built_at: str | None = None,
) -> CaseDevelopment:
    """Build a :class:`CaseDevelopment` for a single doc.

    Pure function: identical ``(record, source_text)`` →
    identical sorted-keys JSON bytes when ``built_at`` is also
    pinned. The :data:`built_at` argument lets callers pin the
    timestamp for byte-stable runs; if ``None``, the current UTC
    time is stamped (and the bytes will then naturally vary across
    calls — the rest of the record is still deterministic).

    Algorithm (see ``wiki/DEVELOPMENT.md § 6``):

    1. NFC-normalise the source.
    2. Segment the source into phase spans (cue-driven).
    3. Locate every NER entity in the source.
    4. Route each located entity into the phase whose span covers
       its ``char_start``; unlocated / out-of-span entities go to
       the unrouted bucket and increment ``stats.n_unrouted``.
    5. For each phase in source order, compute the per-lane
       introduced / carried deltas against a running set of
       ``(type, text)`` keys seen on earlier phases.
    6. Sort each delta list by ``(char_start, type, text)`` so
       JSON bytes are stable.
    7. Stamp the upstream NER cache identifiers and the schema /
       builder version constants.
    """
    nfc_source = unicodedata.normalize("NFC", source_text)
    spans = segment_phases(nfc_source)

    all_entities = list(record.all_entities)
    located = _locate_entities(nfc_source=nfc_source, entities=all_entities)

    # Pre-allocate per-phase lane buckets (just the raw refs; we
    # split into introduced / carried below).
    per_phase: list[dict[Lane, list[EntityRef]]] = [
        {"metadata": [], "maindata": []} for _ in spans
    ]
    n_unrouted = 0

    for loc in located:
        if loc.char_start is None:
            n_unrouted += 1
            continue
        idx = _route_into_phase(loc.char_start, spans)
        if idx is None:
            n_unrouted += 1
            continue
        lane = _lane_for(loc.entity.type)
        per_phase[idx][lane].append(_entity_ref(loc))

    # Per-phase delta against a running set.
    seen_meta: set[tuple[str, str]] = set()
    seen_main: set[tuple[str, str]] = set()
    phases: list[Phase] = []
    n_meta_intro = 0
    n_meta_carry = 0
    n_main_intro = 0
    n_main_carry = 0
    per_phase_counts: dict[str, int] = {}

    for span, lane_buckets in zip(spans, per_phase, strict=True):
        meta_intro: list[EntityRef] = []
        meta_carry: list[EntityRef] = []
        for ref in lane_buckets["metadata"]:
            key = (ref.type, ref.text)
            if key in seen_meta:
                meta_carry.append(ref)
            else:
                meta_intro.append(ref)
                seen_meta.add(key)
        main_intro: list[EntityRef] = []
        main_carry: list[EntityRef] = []
        for ref in lane_buckets["maindata"]:
            key = (ref.type, ref.text)
            if key in seen_main:
                main_carry.append(ref)
            else:
                main_intro.append(ref)
                seen_main.add(key)

        _sort_refs(meta_intro)
        _sort_refs(meta_carry)
        _sort_refs(main_intro)
        _sort_refs(main_carry)

        phases.append(Phase(
            phase=span.phase,
            cue=span.cue,
            char_start=span.char_start,
            char_end=span.char_end,
            metadata_introduced=meta_intro,
            metadata_carried=meta_carry,
            maindata_introduced=main_intro,
            maindata_carried=main_carry,
        ))
        n_meta_intro += len(meta_intro)
        n_meta_carry += len(meta_carry)
        n_main_intro += len(main_intro)
        n_main_carry += len(main_carry)
        per_phase_counts[span.phase] = (
            len(meta_intro) + len(meta_carry)
            + len(main_intro) + len(main_carry)
        )

    n_routed = (
        n_meta_intro + n_meta_carry + n_main_intro + n_main_carry
    )
    stats = DevelopmentStats(
        n_entities_total=len(all_entities),
        n_entities_routed=n_routed,
        n_unrouted=n_unrouted,
        n_phases=len(phases),
        n_metadata_introduced=n_meta_intro,
        n_metadata_carried=n_meta_carry,
        n_maindata_introduced=n_main_intro,
        n_maindata_carried=n_main_carry,
        per_phase=per_phase_counts,
    )

    case_header = _build_case_header(record)

    return CaseDevelopment(
        schema_version=SCHEMA_VERSION,
        builder_version=BUILDER_VERSION,
        doc_name=record.doc_name,
        source_cache_key=record.cache_key,
        source_kb_version=record.kb_version,
        source_prompt_version=record.prompt_version,
        source_input_text_hash=record.input_text_hash,
        built_at=built_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        case_header=case_header,
        phases=phases,
        stats=stats,
    )


# --------------------------------------------------------------------- io


def write_development(
    *,
    development: CaseDevelopment,
    output_root: Path,
) -> Path:
    """Write ``<output_root>/development/<doc_name>.json`` atomically.

    Bytes are produced with sorted keys + indent=2 so visual diffs
    across runs read cleanly. Atomic write via .tmp + rename.
    """
    out_dir = output_root / "development"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{development.doc_name}.json"
    payload = development.model_dump(mode="json")
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


def aggregate_developments_jsonl(
    *,
    output_root: Path,
    doc_names: list[str],
) -> Path:
    """Concatenate per-doc developments into ``developments.jsonl``.

    One JSON line per doc, in lex order. Useful for downstream
    tools that prefer a single streamable artefact.
    """
    dev_dir = output_root / "development"
    out_path = output_root / "developments.jsonl"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for doc_name in sorted(doc_names):
            src = dev_dir / f"{doc_name}.json"
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
    built_at: str | None,
) -> CaseDevelopment:
    """Build + persist one development. Returns the in-memory object."""
    record = read_canonical_record(
        canonical_dir=canonical_dir, doc_name=doc_name,
    )
    source_text = read_source_text(md_dir=md_dir, doc_name=doc_name)
    development = build_development(
        record=record,
        source_text=source_text,
        built_at=built_at,
    )
    write_development(development=development, output_root=output_root)
    return development


@dataclass
class DevelopmentBuildStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Per-document development build as a Curator stage.

    ``process`` runs :func:`build_one` over the batch's ``doc_name`` column
    (read the canonical NER record + source markdown, route entities into
    phases, persist ``developments/<doc>.json``) and returns the in-memory
    :class:`~packages.extractor.development.schema.CaseDevelopment` objects in
    an object-typed ``development`` column. The ``__main__`` driver is a thin
    wrapper that feeds one :class:`DocumentBatch` through this stage — mirroring
    :class:`packages.extractor.ner.extract.NerExtractStage`.
    """

    canonical_dir: Path
    md_dir: Path
    output_root: Path
    built_at: str | None = None
    name: str = "development_build"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["doc_name"])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["development"])

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df = task.to_pandas().copy()
        df["development"] = [
            build_one(
                doc_name=str(d),
                canonical_dir=self.canonical_dir,
                md_dir=self.md_dir,
                output_root=self.output_root,
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
    "DevelopmentBuildStage",
    "aggregate_developments_jsonl",
    "build_development",
    "build_one",
    "list_doc_names",
    "read_canonical_record",
    "read_source_text",
    "write_development",
]


# Make Phase / PhaseId re-exports unused-import-safe.
_ = (Phase, PhaseId)
