"""luutru Curator primitives (URLGenerator / DocumentDownloader / ...).

The four Curator abstract bases have one luutru subclass each:

    url_generator.py  -- LuutruURLGenerator      (URLGenerator)
    downloader.py     -- LuutruDocumentDownloader (DocumentDownloader)
    iterator.py       -- LuutruDocumentIterator  (DocumentIterator)
    extractor.py      -- LuutruDocumentExtractor (DocumentExtractor)

These are the site-specific bricks the top-level pipeline factories
in ``packages.datasites.luutru.{download,parse,extract,embed,reduce}``
compose into :class:`nemo_curator.pipeline.Pipeline` instances.
"""

from packages.datasites.luutru.components.downloader import (
    LuutruDocumentDownloader,
)
from packages.datasites.luutru.components.extractor import (
    LuutruDocumentExtractor,
)
from packages.datasites.luutru.components.iterator import LuutruDocumentIterator
from packages.datasites.luutru.components.url_generator import (
    LuutruURLGenerator,
    absolutize,
    extract_doc_name,
    extract_doc_name_from_url,
)

__all__ = [
    "LuutruDocumentDownloader",
    "LuutruDocumentExtractor",
    "LuutruDocumentIterator",
    "LuutruURLGenerator",
    "absolutize",
    "extract_doc_name",
    "extract_doc_name_from_url",
]
