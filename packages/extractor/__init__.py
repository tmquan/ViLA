"""Stage 3 (extractor) module layout.

    base.py            - :class:`ExtractorAlgorithm` ABC + record types + regex set
    generic.py         - :class:`GenericExtractor` (regex NER + statute linker)
    precedent.py       - :class:`PrecedentExtractor` (Vietnamese án lệ normalizer)
    structure.py       - :class:`LegalStructureExtractor` (hierarchical doc model)
    normalization.py   - NFC + Vietnamese tone-mark canonicalizer (ftfy-backed)
    stage.py           - :class:`LegalExtractStage` (``ProcessingStage``)
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
from packages.extractor.normalization import (
    VietnameseTextNormalizer,
    normalize_text,
)
from packages.extractor.precedent import PrecedentExtractor
from packages.extractor.stage import LegalExtractStage
from packages.extractor.structure import (
    DocumentMeta,
    DocumentStats,
    DocumentStructure,
    LegalStructureExtractor,
    PARAGRAPH_KINDS,
    Paragraph,
    SCHEMA_VERSION,
    SECTION_KINDS,
    Section,
    Sentence,
)

__all__ = [
    "ARTICLE_RE",
    "COURT_RE",
    "DATE_RE",
    "DocumentMeta",
    "DocumentStats",
    "DocumentStructure",
    "Entity",
    "ExtractorAlgorithm",
    "GenericExtractor",
    "GenericRecord",
    "LegalExtractStage",
    "LegalStructureExtractor",
    "PARAGRAPH_KINDS",
    "PRECEDENT_NUMBER_RE",
    "Paragraph",
    "PrecedentExtractor",
    "PrecedentRecord",
    "Relation",
    "SCHEMA_VERSION",
    "SECTION_KINDS",
    "Section",
    "Sentence",
    "StatuteRef",
    "VietnameseTextNormalizer",
    "normalize_text",
    "text_hash",
]
