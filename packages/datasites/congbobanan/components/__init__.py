"""congbobanan Curator primitives.

One subclass of each Curator abstract base:

    url_generator.py  -- CBBADocumentURLGenerator  (URLGenerator)
    downloader.py     -- CBBADocumentPDFDownloader  (PDFDownloader)
    iterator.py       -- CBBADocumentIterator       (DocumentIterator)
    extractor.py      -- CBBADocumentExtractor       (DocumentExtractor)

Composed into a :class:`nemo_curator.pipeline.Pipeline` by
:class:`packages.datasites.congbobanan.pipeline.CBBADocumentDownloadExtractStage`.
"""

from packages.datasites.congbobanan.components.downloader import (
    ACCEPTED_BODY_EXTENSIONS,
    CBBADocumentPDFDownloader,
    page_has_metadata,
)
from packages.datasites.congbobanan.components.extractor import (
    CBBADocumentExtractor,
)
from packages.datasites.congbobanan.components.iterator import (
    CBBADocumentIterator,
)
from packages.datasites.congbobanan.components.url_generator import (
    CBBADocumentURLGenerator,
    doc_id_from_url,
)

__all__ = [
    "ACCEPTED_BODY_EXTENSIONS",
    "CBBADocumentPDFDownloader",
    "CBBADocumentExtractor",
    "CBBADocumentIterator",
    "CBBADocumentURLGenerator",
    "doc_id_from_url",
    "page_has_metadata",
]
