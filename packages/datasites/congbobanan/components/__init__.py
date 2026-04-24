"""congbobanan Curator primitives.

One subclass of each Curator abstract base:

    url_generator.py  -- CongbobananURLGenerator    (URLGenerator)
    downloader.py     -- CongbobananDocumentDownloader (DocumentDownloader)
    iterator.py       -- CongbobananDocumentIterator (DocumentIterator)
    extractor.py      -- CongbobananDocumentExtractor (DocumentExtractor)

Composed into :class:`nemo_curator.pipeline.Pipeline` instances by the
top-level factories in ``packages.datasites.congbobanan.{download,parse,
extract,embed,reduce}``.
"""

from packages.datasites.congbobanan.components.downloader import (
    CongbobananDocumentDownloader,
)
from packages.datasites.congbobanan.components.extractor import (
    CongbobananDocumentExtractor,
)
from packages.datasites.congbobanan.components.iterator import (
    CongbobananDocumentIterator,
)
from packages.datasites.congbobanan.components.url_generator import (
    CongbobananURLGenerator,
    doc_id_from_url,
)

__all__ = [
    "CongbobananDocumentDownloader",
    "CongbobananDocumentExtractor",
    "CongbobananDocumentIterator",
    "CongbobananURLGenerator",
    "doc_id_from_url",
]
