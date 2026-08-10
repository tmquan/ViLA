"""anle.toaan.gov.vn datasite -- Vietnamese án lệ + bản án corpus.

Canonical NeMo Curator structure:

    components/         the four abstract-base subclasses
      url_generator.py  -> AnleURLGenerator  (URLGenerator)
      downloader.py     -> AnlePDFDownloader (PDFDownloader)
      iterator.py       -> AnleIterator      (DocumentIterator)
      extractor.py      -> AnleExtractor     (DocumentExtractor)
    pipeline.py         -> AnleDownloadExtractStage + paced main()
    hf_export.py        -> JSONL -> hf/ (parquet + README + manifest)
    push_to_hf.py       -> hf/ -> HuggingFace dataset repo
"""

from packages.datasites.anle.components import (
    AnleExtractor,
    AnleIterator,
    AnlePDFDownloader,
    AnleURLGenerator,
)
from packages.datasites.anle.pipeline import AnleConfig, AnleDownloadExtractStage

__all__ = [
    "AnleConfig",
    "AnleDownloadExtractStage",
    "AnleExtractor",
    "AnleIterator",
    "AnlePDFDownloader",
    "AnleURLGenerator",
]
