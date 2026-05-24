"""Deterministic case-timeline builder for visual analytics.

Projects the NER ``maindata`` entities for one ban-án onto a
sequence of dated events suitable for vis-timeline / react-chrono /
Apache ECharts timeline / Gantt swimlane renderers. No LLM call,
no network, no randomness — pure function of the upstream NER cache
record + the source markdown.

See ``wiki/TIMELINE.md`` for the spec and reproduction recipe.

Sub-modules:

* :mod:`packages.extractor.timeline.schema` — Pydantic models +
  :data:`SCHEMA_VERSION` and :data:`BUILDER_VERSION` constants.
* :mod:`packages.extractor.timeline.dates` — Vietnamese date
  surface-form → ISO + sortable key.
* :mod:`packages.extractor.timeline.locator` — re-localise entity
  texts to char offsets in the source markdown (NFC, greedy left-
  to-right).
* :mod:`packages.extractor.timeline.cluster` — date-anchored
  proximity clustering (single-knob ``window_chars``).
* :mod:`packages.extractor.timeline.classify` — heuristic event-kind
  classifier (fact / filing / hearing / verdict / sentence / ambient
  / unknown).
* :mod:`packages.extractor.timeline.build` — per-doc builder, IO,
  aggregation.
* :mod:`packages.extractor.timeline.__main__` — CLI entry point;
  ``python -m packages.extractor.timeline --help``.

Determinism: every output that reaches disk is a function of
``(source_cache_key, builder_version, cluster_window_chars)`` —
re-runs that hit no algorithm change are byte-for-byte identical.
:mod:`tests.unit.test_timeline_determinism` pins this contract.
"""

from packages.extractor.timeline.build import (
    aggregate_timelines_jsonl,
    build_one,
    build_timeline,
    list_doc_names,
    read_canonical_record,
    read_source_text,
    write_timeline,
)
from packages.extractor.timeline.dates import parse_date_to_anchor
from packages.extractor.timeline.schema import (
    BUILDER_VERSION,
    SCHEMA_VERSION,
    Actor,
    CaseHeader,
    CaseOutcome,
    CaseTimeline,
    EventKind,
    MoneyRef,
    PartyRole,
    PartySummary,
    Place,
    SentenceRef,
    StatuteRef,
    TermRef,
    TimelineEvent,
    TimelineStats,
    WhenAnchor,
)

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
    "SentenceRef",
    "StatuteRef",
    "TermRef",
    "TimelineEvent",
    "TimelineStats",
    "WhenAnchor",
    "aggregate_timelines_jsonl",
    "build_one",
    "build_timeline",
    "list_doc_names",
    "parse_date_to_anchor",
    "read_canonical_record",
    "read_source_text",
    "write_timeline",
]
