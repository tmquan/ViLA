"""Cross-site pipeline helpers: executor factory + Ray client bootstrap.

Per-site :class:`nemo_curator.pipeline.Pipeline` factories live under
:mod:`packages.datasites.<site>.pipeline`; this module only hosts the
bits that are identical across every site.
"""

from packages.pipeline.executors import (
    EXECUTOR_CHOICES,
    build_executor,
    init_ray,
    shutdown_ray,
)
from packages.pipeline.io import (
    JSONL_EXTENSION,
    MARKDOWN_EXTENSION,
    META_EXTENSION,
    PARQUET_EXTENSION,
    JsonlPerDocWriter,
    MarkdownPerDocWriter,
    MarkdownReader,
    MarkdownReaderStage,
    ParquetPerDocWriter,
)

__all__ = [
    "EXECUTOR_CHOICES",
    "JSONL_EXTENSION",
    "JsonlPerDocWriter",
    "MARKDOWN_EXTENSION",
    "META_EXTENSION",
    "MarkdownPerDocWriter",
    "MarkdownReader",
    "MarkdownReaderStage",
    "PARQUET_EXTENSION",
    "ParquetPerDocWriter",
    "build_executor",
    "init_ray",
    "shutdown_ray",
]
