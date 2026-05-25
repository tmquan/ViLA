"""Deterministic case-timeline builder for visual analytics.

Projects the NER ``maindata`` entities for one ban-án onto a
sequence of dated events suitable for vis-timeline / react-chrono /
Apache ECharts timeline / Gantt swimlane renderers. The ``v3``
shape collapses the prior dual-track (``meta`` / ``main``) layout
into a single chronological lane plus a static
:class:`CaseHeader` — see ``wiki/TIMELINE.md § 2`` for the routing
rule (``METADATA_TYPES`` → header, ``MAINDATA_TYPES`` → events).

No LLM call, no network, no randomness — pure function of the
upstream NER cache record + the source markdown.

See ``wiki/TIMELINE.md`` for the spec and reproduction recipe.

Sub-modules:

* :mod:`packages.extractor.timeline.schema` — Pydantic models +
  :data:`SCHEMA_VERSION` and :data:`BUILDER_VERSION` constants.
* :mod:`packages.extractor.timeline.datetimes` — Vietnamese date
  *and time* surface-form → ISO + sortable key. Also hosts the
  relative-temporal parser (``X phút sau``, ``Cùng ngày``,
  ``Hôm qua``, …) and the source-text scanner.
* :mod:`packages.extractor.timeline.locator` — re-localise entity
  texts to char offsets in the source markdown (NFC, greedy left-
  to-right).
* :mod:`packages.extractor.timeline.cluster` — date-anchored
  proximity clustering (single-knob ``window_chars``).
* :mod:`packages.extractor.timeline.classify` — heuristic event-kind
  classifier (fact / filing / hearing / verdict / sentence /
  unknown). Pure label, never decides which lane an event is on.
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
from packages.extractor.timeline.datetimes import (
    find_relative_expressions,
    parse_date_to_anchor,
    parse_relative_to_anchor,
)
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
    "find_relative_expressions",
    "list_doc_names",
    "parse_date_to_anchor",
    "parse_relative_to_anchor",
    "read_canonical_record",
    "read_source_text",
    "write_timeline",
]
