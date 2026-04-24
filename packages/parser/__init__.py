"""Stage 2 (parser) module layout.

    base.py     - :class:`ParserAlgorithm` ABC (backend contract)
    nemotron.py - :class:`NemotronParser` (NIM ``nvidia/nemotron-parse``)
    pypdf.py    - :class:`PypdfParser`    (local pypdf + docx2txt)
    hybrid.py   - :class:`HybridParser`   (local first, NIM fallback on empty)
    stage.py    - :class:`PdfParseStage`  (``ProcessingStage[DocumentBatch, DocumentBatch]``)

Composed into a :class:`nemo_curator.pipeline.Pipeline` by
:mod:`packages.datasites.<site>.pipeline`.
"""

from packages.parser.base import ParserAlgorithm
from packages.parser.hybrid import HybridParser
from packages.parser.nemotron import NemoretrieverParser, NemotronParser
from packages.parser.pypdf import PypdfParser
from packages.parser.stage import PdfParseStage, build_parser

__all__ = [
    "HybridParser",
    "NemoretrieverParser",
    "NemotronParser",  # back-compat alias; same class as NemoretrieverParser
    "ParserAlgorithm",
    "PdfParseStage",
    "PypdfParser",
    "build_parser",
]
