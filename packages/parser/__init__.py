"""Stage 2 (parser) module layout.

    base.py     - :class:`ParserAlgorithm` ABC (backend contract)
    nemotron.py - :class:`NemotronParser` (NIM ``nvidia/nemotron-parse``)
    pypdf.py    - :class:`PypdfParser`    (local pypdf + docx2txt)
    stage.py    - :class:`PdfParseStage`  (``ProcessingStage[DocumentBatch, DocumentBatch]``)

Composed into a :class:`nemo_curator.pipeline.Pipeline` by
:mod:`packages.datasites.<site>.pipeline`.
"""

from packages.parser.base import ParserAlgorithm
from packages.parser.nemotron import NemotronParser
from packages.parser.pypdf import PypdfParser
from packages.parser.stage import PdfParseStage, build_parser

__all__ = [
    "NemotronParser",
    "ParserAlgorithm",
    "PdfParseStage",
    "PypdfParser",
    "build_parser",
]
