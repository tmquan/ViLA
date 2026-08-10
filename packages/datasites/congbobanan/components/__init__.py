"""congbobanan Curator primitives.

One subclass of each Curator abstract base:

    url_generator.py  -- CongbobananURLGenerator  (URLGenerator)
    downloader.py     -- CongbobananPDFDownloader  (PDFDownloader)
    iterator.py       -- CongbobananIterator       (DocumentIterator)
    extractor.py      -- CongbobananExtractor       (DocumentExtractor)

Composed into a :class:`nemo_curator.pipeline.Pipeline` by
:class:`packages.datasites.congbobanan.pipeline.CongbobananDownloadExtractStage`.
"""

from packages.datasites.congbobanan.components.downloader import (
    ACCEPTED_BODY_EXTENSIONS,
    CongbobananPDFDownloader,
    page_has_metadata,
)
from packages.datasites.congbobanan.components.extractor import (
    CongbobananExtractor,
)
from packages.datasites.congbobanan.components.iterator import (
    CongbobananIterator,
)
from packages.datasites.congbobanan.components.url_generator import (
    CongbobananURLGenerator,
    doc_id_from_url,
)

__all__ = [
    "ACCEPTED_BODY_EXTENSIONS",
    "CongbobananPDFDownloader",
    "CongbobananExtractor",
    "CongbobananIterator",
    "CongbobananURLGenerator",
    "doc_id_from_url",
    "page_has_metadata",
]
