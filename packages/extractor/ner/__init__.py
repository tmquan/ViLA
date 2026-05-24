"""LLM-driven NER + KB-grounding pipeline for Vietnamese ban-án.

Sub-package layout (see ``wiki/EXTRACTION.md`` for the procedure-as-spec):

* :mod:`packages.extractor.ner.kb` — tnpl gazetteer + phapdien article
  index builders, with on-disk pickle cache keyed on the input file
  hashes.
* :mod:`packages.extractor.ner.schema` — Pydantic models for the LLM
  JSON output (22 entity types + per-doc summary + grounding
  attributes).
* :mod:`packages.extractor.ner.prompts` — versioned system / user
  prompts.
* :mod:`packages.extractor.ner.client` — thin synchronous NIM
  chat-completions client with per-model reasoning toggles.
* :mod:`packages.extractor.ner.linker` — exact + fuzzy KB linking from
  extracted spans to ``term_id`` / ``article_anchor``.
* :mod:`packages.extractor.ner.extract` — per-doc pipeline (read →
  build cache key → LLM call → parse → link → persist + manifest row).
* :mod:`packages.extractor.ner.__main__` — CLI entry point;
  ``python -m packages.extractor.ner --help``.

Determinism: every output that reaches disk is a function of
``(doc_name, model_id, prompt_version, kb_version, input_text_hash)``;
re-runs that hit the cache are byte-for-byte identical. See
``tests/unit/test_ner_determinism.py`` for the regression tests.
"""

from packages.extractor.ner.prompts import PROMPT_VERSION
from packages.extractor.ner.schema import (
    ENTITY_TYPES,
    MAINDATA_TYPES,
    METADATA_TYPES,
    EntityType,
    ExtractedEntity,
    ExtractionStats,
    KbCoverage,
    LLMExtraction,
    PersistedExtraction,
    section_for,
)

__all__ = [
    "ENTITY_TYPES",
    "MAINDATA_TYPES",
    "METADATA_TYPES",
    "PROMPT_VERSION",
    "EntityType",
    "ExtractedEntity",
    "ExtractionStats",
    "KbCoverage",
    "LLMExtraction",
    "PersistedExtraction",
    "section_for",
]
