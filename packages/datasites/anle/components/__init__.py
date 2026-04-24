"""Anle Curator primitives (URLGenerator / DocumentDownloader / ...).

The four Curator abstract bases have one anle subclass each:

    url_generator.py  -- AnleURLGenerator     (URLGenerator)
    downloader.py     -- AnleDocumentDownloader (DocumentDownloader)
    iterator.py       -- AnleDocumentIterator (DocumentIterator)
    extractor.py      -- AnleDocumentExtractor (DocumentExtractor)

These are the site-specific bricks the top-level pipeline factories
in ``packages.datasites.anle.{download,parse,extract,embed,reduce}``
compose into :class:`nemo_curator.pipeline.Pipeline` instances.
"""

from packages.datasites.anle.components.downloader import AnleDocumentDownloader
from packages.datasites.anle.components.extractor import AnleDocumentExtractor
from packages.datasites.anle.components.iterator import AnleDocumentIterator
from packages.datasites.anle.components.url_generator import (
    AnleURLGenerator,
    absolutize,
    extract_doc_name,
    extract_doc_name_from_url,
)

__all__ = [
    "AnleDocumentDownloader",
    "AnleDocumentExtractor",
    "AnleDocumentIterator",
    "AnleURLGenerator",
    "absolutize",
    "extract_doc_name",
    "extract_doc_name_from_url",
]
