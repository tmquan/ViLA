"""Stage 2 (parser) module layout.

    base.py          - :class:`ParserAlgorithm` ABC (backend contract)
    pypdf.py         - :class:`PypdfParser` (local pypdf + docx2txt)
    nemotron.py      - :class:`NemotronParseClient` (NIM ``nvidia/nemotron-parse`` layout OCR)
    nemotron_omni.py - :class:`NemotronOmniClient` (NIM Nemotron-3 Nano Omni VLM OCR)
    qwen3_6_omni.py  - :class:`Qwen36OmniClient` (self-hosted Qwen3.6 Omni VLM OCR; default fallback)
    hybrid.py        - :class:`HybridParser` (local first; VLM fallback on empty/lossy/corrupt)
    triage.py, native_interleaved.py, cmap_healer.py, types.py, ocr_models.py - support
    stage.py         - :class:`PdfParseStage` (``ProcessingStage[DocumentBatch, DocumentBatch]``)

Composed into a :class:`nemo_curator.pipeline.Pipeline` by
:mod:`packages.datasites.<site>.pipeline`.
"""

from packages.parser.base import ParserAlgorithm
from packages.parser.hybrid import HybridParser
from packages.parser.nemotron import (
    NemoretrieverParser,
    NemotronParseClient,
    NemotronParser,
)
from packages.parser.nemotron_omni import NemotronOmniClient
from packages.parser.pypdf import PypdfParser
from packages.parser.qwen3_6_omni import Qwen36OmniClient
from packages.parser.stage import PdfParseStage, build_parser

__all__ = [
    "HybridParser",
    "NemoretrieverParser",  # back-compat alias for NemotronParseClient
    "NemotronOmniClient",
    "NemotronParseClient",
    "NemotronParser",  # back-compat alias for NemotronParseClient
    "ParserAlgorithm",
    "PdfParseStage",
    "PypdfParser",
    "Qwen36OmniClient",
    "build_parser",
]
