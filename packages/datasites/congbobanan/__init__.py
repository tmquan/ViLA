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

# Register site-specific normalizers (congbobanan_join_word_breaks,
# congbobanan_join_soft_wraps, congbobanan_strip_page_noise) into the
# global :data:`packages.extractor.normalizers.NORMALIZER_REGISTRY` so
# the YAML chain in ``configs/default.yaml`` resolves them by name.
# The eager import here covers both the driver process and remote
# Ray workers (which import this package at actor setup time).
from packages.datasites.congbobanan import normalizers as _normalizers  # noqa: F401

__all__ = [
    "ALL_PIPELINES_ORDER",
    "PIPELINES",
    "CongbobananDocumentDownloader",
    "CongbobananDocumentExtractor",
    "CongbobananDocumentIterator",
    "CongbobananURLGenerator",
    "build_download_pipeline",
    "build_embed_pipeline",
    "build_extract_pipeline",
    "build_parse_pipeline",
    "build_pipeline",
    "build_reduce_pipeline",
]
