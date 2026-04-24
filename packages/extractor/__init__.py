"""Stage 3 (extractor) module layout.

    base.py        - :class:`ExtractorAlgorithm` ABC + record types + regex set
    generic.py     - :class:`GenericExtractor` (regex NER + statute linker)
    precedent.py   - :class:`PrecedentExtractor` (Vietnamese án lệ normalizer)
    stage.py       - :class:`LegalExtractStage` (``ProcessingStage``)
"""

from packages.extractor.base import (
    ARTICLE_RE,
    COURT_RE,
    DATE_RE,
    PRECEDENT_NUMBER_RE,
    Entity,
    ExtractorAlgorithm,
    GenericRecord,
    PrecedentRecord,
    Relation,
    StatuteRef,
    text_hash,
)
from packages.extractor.generic import GenericExtractor
from packages.extractor.precedent import PrecedentExtractor
from packages.extractor.stage import LegalExtractStage

__all__ = [
    "ARTICLE_RE",
    "COURT_RE",
    "DATE_RE",
    "Entity",
    "ExtractorAlgorithm",
    "GenericExtractor",
    "GenericRecord",
    "LegalExtractStage",
    "PRECEDENT_NUMBER_RE",
    "PrecedentExtractor",
    "PrecedentRecord",
    "Relation",
    "StatuteRef",
    "text_hash",
]
