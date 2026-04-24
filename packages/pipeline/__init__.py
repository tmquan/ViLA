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
    MARKDOWN_EXTENSION,
    META_EXTENSION,
    MarkdownPerDocWriter,
    MarkdownReader,
    MarkdownReaderStage,
)

__all__ = [
    "EXECUTOR_CHOICES",
    "MARKDOWN_EXTENSION",
    "META_EXTENSION",
    "MarkdownPerDocWriter",
    "MarkdownReader",
    "MarkdownReaderStage",
    "build_executor",
    "init_ray",
    "shutdown_ray",
]
