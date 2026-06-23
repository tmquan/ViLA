"""Curator download primitives for the dichvucong datasite."""

from __future__ import annotations

from packages.datasites.dichvucong.components.downloader import (
    DichvucongDocumentDownloader,
)
from packages.datasites.dichvucong.components.extractor import (
    DichvucongDocumentExtractor,
)
from packages.datasites.dichvucong.components.iterator import (
    DichvucongDocumentIterator,
)
from packages.datasites.dichvucong.components.url_generator import (
    DichvucongURLGenerator,
)

__all__ = [
    "DichvucongDocumentDownloader",
    "DichvucongDocumentExtractor",
    "DichvucongDocumentIterator",
    "DichvucongURLGenerator",
]
