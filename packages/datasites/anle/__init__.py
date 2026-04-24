"""anle.toaan.gov.vn datasite.

Top-level files map 1-to-1 onto the five Curator pipelines:

    download.py  -> URLs     -> PDFs
    parse.py     -> PDFs     -> markdown
    extract.py   -> markdown -> JSONL
    embed.py     -> JSONL    -> embeddings parquet
    reduce.py    -> embeddings parquet -> reduced parquet
    pipeline.py  -> registry + ``build_pipeline(cfg, name)`` dispatch

The four Curator abstract-base subclasses (URLGenerator,
DocumentDownloader, DocumentIterator, DocumentExtractor) live under
:mod:`packages.datasites.anle.components`.
"""

from packages.datasites.anle.components import (
    AnleDocumentDownloader,
    AnleDocumentExtractor,
    AnleDocumentIterator,
    AnleURLGenerator,
)
from packages.datasites.anle.pipeline import (
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
    "AnleDocumentDownloader",
    "AnleDocumentExtractor",
    "AnleDocumentIterator",
    "AnleURLGenerator",
    "PIPELINES",
    "build_download_pipeline",
    "build_embed_pipeline",
    "build_extract_pipeline",
    "build_parse_pipeline",
    "build_pipeline",
    "build_reduce_pipeline",
]
