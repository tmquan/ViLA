"""congbobanan.toaan.gov.vn datasite (canonical NeMo Curator structure).

Four Curator abstract-base subclasses under
:mod:`packages.datasites.congbobanan.components`:

    url_generator.py  -- CongbobananURLGenerator  (URLGenerator)
    downloader.py     -- CongbobananPDFDownloader  (PDFDownloader)
    iterator.py       -- CongbobananIterator       (DocumentIterator)
    extractor.py      -- CongbobananExtractor       (DocumentExtractor)

wired by :class:`packages.datasites.congbobanan.pipeline.CongbobananDownloadExtractStage`
(+ a single-IP paced ``main()`` runner).
"""

from packages.datasites.congbobanan.components import (
    CongbobananExtractor,
    CongbobananIterator,
    CongbobananPDFDownloader,
    CongbobananURLGenerator,
    doc_id_from_url,
)
from packages.datasites.congbobanan.pipeline import (
    CongbobananDownloadExtractStage,
    main,
)

__all__ = [
    "CongbobananDownloadExtractStage",
    "CongbobananExtractor",
    "CongbobananIterator",
    "CongbobananPDFDownloader",
    "CongbobananURLGenerator",
    "doc_id_from_url",
    "main",
]
