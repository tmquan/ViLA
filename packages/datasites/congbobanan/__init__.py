"""congbobanan.toaan.gov.vn datasite (canonical NeMo Curator structure).

Four Curator abstract-base subclasses under
:mod:`packages.datasites.congbobanan.components`:

    url_generator.py  -- CBBADocumentURLGenerator  (URLGenerator)
    downloader.py     -- CBBADocumentPDFDownloader  (PDFDownloader)
    iterator.py       -- CBBADocumentIterator       (DocumentIterator)
    extractor.py      -- CBBADocumentExtractor       (DocumentExtractor)

wired by :class:`packages.datasites.congbobanan.pipeline.CBBADocumentDownloadExtractStage`
(+ a single-IP paced ``main()`` runner).
"""

from packages.datasites.congbobanan.components import (
    CBBADocumentExtractor,
    CBBADocumentIterator,
    CBBADocumentPDFDownloader,
    CBBADocumentURLGenerator,
    doc_id_from_url,
)
from packages.datasites.congbobanan.pipeline import (
    CBBADocumentDownloadExtractStage,
    main,
)

__all__ = [
    "CBBADocumentDownloadExtractStage",
    "CBBADocumentExtractor",
    "CBBADocumentIterator",
    "CBBADocumentPDFDownloader",
    "CBBADocumentURLGenerator",
    "doc_id_from_url",
    "main",
]
