"""congbobanan.toaan.gov.vn datasite.

Top-level files map 1-to-1 onto the five Curator pipelines:

    download.py  -> integer case IDs -> PDFs
    parse.py     -> PDFs             -> markdown
    extract.py   -> markdown         -> JSONL
    embed.py     -> JSONL            -> embeddings parquet
    reduce.py    -> embeddings       -> reduced parquet
    pipeline.py  -> registry + ``build_pipeline(cfg, name)`` dispatch

The four Curator abstract-base subclasses (URLGenerator,
DocumentDownloader, DocumentIterator, DocumentExtractor) live under
:mod:`packages.datasites.congbobanan.components`.
"""

from packages.datasites.congbobanan.components import (
    CongbobananDocumentDownloader,
    CongbobananDocumentExtractor,
    CongbobananDocumentIterator,
    CongbobananURLGenerator,
)
from packages.datasites.congbobanan.pipeline import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    build_download_pipeline,
    build_embed_pipeline,
    build_extract_pipeline,
    build_parse_pipeline,
    build_pipeline,
    build_reduce_pipeline,
)

__all__ = [
    "ALL_PIPELINES_ORDER",
    "CongbobananDocumentDownloader",
    "CongbobananDocumentExtractor",
    "CongbobananDocumentIterator",
    "CongbobananURLGenerator",
    "PIPELINES",
    "build_download_pipeline",
    "build_embed_pipeline",
    "build_extract_pipeline",
    "build_parse_pipeline",
    "build_pipeline",
    "build_reduce_pipeline",
]
