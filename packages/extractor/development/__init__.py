"""Deterministic case-development builder for procedural-arc analytics.

Projects each ban-án's NER record onto its **procedural-development
arc** — the ordered sequence of macro-structural phases a court
judgment passes through (``preamble`` → ``narrative`` →
``investigation`` → ``hearing`` → ``reasoning`` → ``ruling`` →
``signature``) and the per-lane (metadata / maindata) entity
delta at each phase. Sibling of :mod:`packages.extractor.timeline`
— complementary projection, not a replacement.

No LLM call, no network, no randomness — pure function of the
upstream NER cache record + the source markdown.

See ``wiki/DEVELOPMENT.md`` for the spec and reproduction recipe.

Sub-modules:

* :mod:`packages.extractor.development.schema` — Pydantic models +
  :data:`SCHEMA_VERSION` / :data:`BUILDER_VERSION` constants.
* :mod:`packages.extractor.development.classify` — cue tables
  used by the segmenter.
* :mod:`packages.extractor.development.segmenter` — source text
  → ordered list of contiguous phase spans (cue-driven).
* :mod:`packages.extractor.development.build` — per-doc builder,
  IO, aggregation; entry point :func:`build_development`.
* :mod:`packages.extractor.development.__main__` — CLI entry
  point; ``python -m packages.extractor.development --help``.

Determinism: every output that reaches disk is a function of
``(source_cache_key, builder_version)`` — re-runs that hit no
algorithm change are byte-for-byte identical.
:mod:`tests.unit.test_development_determinism` pins this contract.
"""

from packages.extractor.development.build import (
    aggregate_developments_jsonl,
    build_development,
    build_one,
    list_doc_names,
    read_canonical_record,
    read_source_text,
    write_development,
)
from packages.extractor.development.schema import (
    BUILDER_VERSION,
    PHASE_ORDER,
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
    "PhaseSpan",
    "aggregate_developments_jsonl",
    "build_development",
    "build_one",
    "list_doc_names",
    "read_canonical_record",
    "read_source_text",
    "segment_phases",
    "write_development",
]
