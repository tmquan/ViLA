"""thuvienphapluat hoi-dap Q&A Curator primitives.

One subclass of each Curator abstract base:

    url_generator.py  -- TVPLQAURLGenerator (URLGenerator)
    downloader.py     -- TVPLQADownloader   (HTMLDownloader / DocumentDownloader)
    iterator.py       -- TVPLQAIterator     (DocumentIterator)
    extractor.py      -- TVPLQAExtractor     (DocumentExtractor)

Composed into the ``TVPLQADownloadExtractStage`` in
``packages.datasites.thuvienphapluat_hdpl.pipeline``.
"""

from packages.datasites.thuvienphapluat_hdpl.components.downloader import TVPLQADownloader
from packages.datasites.thuvienphapluat_hdpl.components.extractor import TVPLQAExtractor
from packages.datasites.thuvienphapluat_hdpl.components.iterator import TVPLQAIterator
from packages.datasites.thuvienphapluat_hdpl.components.url_generator import TVPLQAURLGenerator

__all__ = [
    "TVPLQADownloader",
    "TVPLQAExtractor",
    "TVPLQAIterator",
    "TVPLQAURLGenerator",
]
