"""Anle Curator primitives (the four abstract-base subclasses).

    url_generator.py  -- AnleURLGenerator  (URLGenerator)
    downloader.py     -- AnlePDFDownloader (PDFDownloader)
    iterator.py       -- AnleIterator      (DocumentIterator)
    extractor.py      -- AnleExtractor     (DocumentExtractor)
"""

from packages.datasites.anle.components.downloader import AnlePDFDownloader
from packages.datasites.anle.components.extractor import AnleExtractor
from packages.datasites.anle.components.iterator import AnleIterator
from packages.datasites.anle.components.url_generator import (
    AnleURLGenerator,
    absolutize,
    extract_doc_name,
    extract_doc_name_from_url,
)

__all__ = [
    "AnleExtractor",
    "AnleIterator",
    "AnlePDFDownloader",
    "AnleURLGenerator",
    "absolutize",
    "extract_doc_name",
    "extract_doc_name_from_url",
]
