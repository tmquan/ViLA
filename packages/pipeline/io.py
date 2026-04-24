"""Markdown-per-document I/O stages.

Curator ships :class:`JsonlWriter` / :class:`ParquetWriter` that key
output files by ``task_id`` (one file per :class:`DocumentBatch`). The
parse stage needs the opposite: one ``<doc_name>.md`` file per row so
operators can grep / diff / regenerate a single document's markdown
without touching the rest.

Two stages here implement that pair:

    MarkdownPerDocWriter   DocumentBatch -> FileGroupTask
    MarkdownReader         _EmptyTask    -> DocumentBatch (composite)

Each row's markdown lands in ``<path>/<doc_name>.md``. Every other
non-empty column flows to a sibling ``<path>/<doc_name>.meta.json``
so downstream stages (e.g. :class:`LegalExtractStage`) see the full
row again after the markdown round-trip.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import CompositeStage, ProcessingStage
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, FileGroupTask, _EmptyTask

logger = logging.getLogger(__name__)


META_EXTENSION = ".meta.json"
MARKDOWN_EXTENSION = ".md"


# --------------------------------------------------------------------- writer


@dataclass
class MarkdownPerDocWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Write one ``<doc_name>.md`` + sibling ``<doc_name>.meta.json`` per row.

    The markdown payload is the value of ``markdown_field``; everything
    else on the row is JSON-serialised into the meta sidecar so
    downstream stages can rebuild the full DocumentBatch without
    re-reading upstream artifacts.

    Non-JSON-serialisable cells fall back to ``str(value)`` via
    ``json.dumps(..., default=str)``. Binary columns (e.g. ``pdf_bytes``)
    are dropped entirely to keep the meta sidecar small and greppable.
    """

    path: str
    doc_name_field: str = "doc_name"
    markdown_field: str = "markdown"
    drop_fields: tuple[str, ...] = ("pdf_bytes",)
    name: str = "markdown_per_doc_writer"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [self.doc_name_field, self.markdown_field])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def process(self, task: DocumentBatch) -> FileGroupTask:
        df = task.to_pandas()
        written: list[str] = []
        drop = set(self.drop_fields) | {self.markdown_field}

        for _, row in df.iterrows():
            doc_name = str(row.get(self.doc_name_field) or "").strip()
            if not doc_name:
                logger.warning(
                    "row missing %s; skipping markdown write",
                    self.doc_name_field,
                )
                continue
            md_path = Path(self.path) / f"{doc_name}{MARKDOWN_EXTENSION}"
            meta_path = Path(self.path) / f"{doc_name}{META_EXTENSION}"

            md_path.write_text(
                str(row.get(self.markdown_field) or ""),
                encoding="utf-8",
            )
            meta = {k: _jsonable(v) for k, v in row.items() if k not in drop}
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            written.append(str(md_path))
            written.append(str(meta_path))

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=written,
            _metadata={**task._metadata, "format": "markdown_per_doc"},
            _stage_perf=task._stage_perf,
        )


def _jsonable(value: Any) -> Any:
    """Coerce pandas / numpy scalars to JSON-friendly Python types."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    # pandas NA / NaN handling via pd.isna for scalar-only values.
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


# --------------------------------------------------------------------- reader


@dataclass
class MarkdownReaderStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Read one ``<doc_name>.md`` (+ sibling ``.meta.json``) per file path."""

    markdown_field: str = "markdown"
    doc_name_field: str = "doc_name"
    name: str = "markdown_reader_stage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [self.doc_name_field, self.markdown_field])

    def process(self, task: FileGroupTask) -> DocumentBatch:
        rows: list[dict[str, Any]] = []
        for p in task.data:
            path = Path(p)
            if path.suffix != MARKDOWN_EXTENSION:
                # File partitioning may hand us meta sidecars too; skip.
                continue
            doc_name = path.stem
            markdown = path.read_text(encoding="utf-8")

            meta_path = path.with_suffix(META_EXTENSION)
            meta: dict[str, Any] = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "invalid meta sidecar %s: %s; continuing with md-only row",
                        meta_path, exc,
                    )

            row: dict[str, Any] = {
                **meta,
                self.doc_name_field: meta.get(self.doc_name_field) or doc_name,
                self.markdown_field: markdown,
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata={**task._metadata, "format": "markdown_per_doc"},
            _stage_perf=task._stage_perf,
        )


@dataclass
class MarkdownReader(CompositeStage[_EmptyTask, DocumentBatch]):
    """Composite: partition ``*.md`` -> read each into one DocumentBatch."""

    file_paths: str | list[str]
    files_per_partition: int | None = 8
    doc_name_field: str = "doc_name"
    markdown_field: str = "markdown"
    name: str = "markdown_reader"

    def __post_init__(self) -> None:
        super().__init__()

    def decompose(self) -> list[ProcessingStage]:
        return [
            FilePartitioningStage(
                file_paths=self.file_paths,
                file_extensions=[MARKDOWN_EXTENSION],
                files_per_partition=self.files_per_partition,
            ),
            MarkdownReaderStage(
                markdown_field=self.markdown_field,
                doc_name_field=self.doc_name_field,
            ),
        ]

    def get_description(self) -> str:
        return f"Read {MARKDOWN_EXTENSION} files from {self.file_paths}"


__all__ = [
    "MARKDOWN_EXTENSION",
    "META_EXTENSION",
    "MarkdownPerDocWriter",
    "MarkdownReader",
    "MarkdownReaderStage",
]
