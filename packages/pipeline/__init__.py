"""Cross-site pipeline helpers: executor + Ray + IO + stage factories.

Per-site :class:`nemo_curator.pipeline.Pipeline` builders live under
:mod:`packages.datasites.<site>` and use the factories in this
package to share the boilerplate across sites.
"""

from packages.pipeline.executors import (
    EXECUTOR_CHOICES,
    build_executor,
    init_ray,
    shutdown_ray,
)
from packages.pipeline.factories import (
    DEFAULT_FPP,
    build_embed_pipeline,
    build_extract_pipeline,
    build_reduce_pipeline,
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
    "DEFAULT_FPP",
    "EXECUTOR_CHOICES",
    "JSONL_EXTENSION",
    "MARKDOWN_EXTENSION",
    "META_EXTENSION",
    "PARQUET_EXTENSION",
    "JsonlPerDocWriter",
    "MarkdownPerDocWriter",
    "MarkdownReader",
    "MarkdownReaderStage",
    "ParquetPerDocWriter",
    "build_embed_pipeline",
    "build_executor",
    "build_extract_pipeline",
    "build_reduce_pipeline",
    "init_ray",
    "shutdown_ray",
]
