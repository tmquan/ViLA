"""Per-document I/O stages (markdown, JSONL, parquet).

Curator ships :class:`JsonlWriter` / :class:`ParquetWriter` that key
output files by ``task_id`` -- one file per :class:`DocumentBatch`,
which means many documents share one output filename. Every
pipeline stage in ViLA wants the opposite: one ``<doc_name>.<ext>``
file per row so operators can grep / diff / regenerate / resume a
single document's artifact without touching the rest.

Three writer stages here + one reader composite:

    MarkdownPerDocWriter   DocumentBatch -> FileGroupTask   (markdown body + meta sidecar)
    JsonlPerDocWriter      DocumentBatch -> FileGroupTask   (one-line-per-file JSONL)
    ParquetPerDocWriter    DocumentBatch -> FileGroupTask   (one-row-per-file parquet)
    MarkdownReader         _EmptyTask    -> DocumentBatch   (file-per-doc markdown composite)

Every writer drops non-serialisable byte columns (``pdf_bytes`` by
default) and optionally projects a user-supplied ``fields`` list to
keep the on-disk schema narrow.
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
JSONL_EXTENSION = ".jsonl"
PARQUET_EXTENSION = ".parquet"


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


# --------------------------------------------------------------------- JSONL-per-doc


@dataclass
class JsonlPerDocWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Write one ``<doc_name>.jsonl`` per row (one line per file).

    Mirrors Curator's :class:`JsonlWriter` API (``path`` + ``fields`` +
    ``mode="ignore"`` semantics) but keys files by ``doc_name`` so the
    pipeline is resume-friendly at document granularity.

    Bytes columns (``pdf_bytes`` by default) are dropped before
    serialisation. If ``fields`` is set, only those columns are
    projected; unknown columns in ``fields`` are silently skipped.
    """

    path: str
    doc_name_field: str = "doc_name"
    fields: list[str] | None = None
    drop_fields: tuple[str, ...] = ("pdf_bytes",)
    name: str = "jsonl_per_doc_writer"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [self.doc_name_field])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def process(self, task: DocumentBatch) -> FileGroupTask:
        df = task.to_pandas()
        df = _project_columns(df, fields=self.fields, drop_fields=self.drop_fields)

        written: list[str] = []
        for _, row in df.iterrows():
            doc_name = _doc_name_or_empty(row.get(self.doc_name_field))
            if not doc_name:
                logger.warning(
                    "row missing %s; skipping jsonl write",
                    self.doc_name_field,
                )
                continue
            out_path = Path(self.path) / f"{doc_name}{JSONL_EXTENSION}"
            obj = {k: _jsonable(v) for k, v in row.items()}
            out_path.write_text(
                json.dumps(obj, ensure_ascii=False, default=str) + "\n",
                encoding="utf-8",
            )
            written.append(str(out_path))

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=written,
            _metadata={**task._metadata, "format": "jsonl_per_doc"},
            _stage_perf=task._stage_perf,
        )


# --------------------------------------------------------------------- parquet-per-doc


@dataclass
class ParquetPerDocWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Write one ``<doc_name>.parquet`` per row (one row per file).

    Mirrors Curator's :class:`ParquetWriter` API but keys files by
    ``doc_name`` so the pipeline is resume-friendly at document
    granularity. Empty list / object columns round-trip cleanly via
    ``pandas.DataFrame.to_parquet`` (pyarrow backend) on a
    single-row frame.
    """

    path: str
    doc_name_field: str = "doc_name"
    fields: list[str] | None = None
    drop_fields: tuple[str, ...] = ("pdf_bytes",)
    name: str = "parquet_per_doc_writer"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [self.doc_name_field])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def process(self, task: DocumentBatch) -> FileGroupTask:
        df = task.to_pandas()
        df = _project_columns(df, fields=self.fields, drop_fields=self.drop_fields)

        written: list[str] = []
        for _, row in df.iterrows():
            doc_name = _doc_name_or_empty(row.get(self.doc_name_field))
            if not doc_name:
                logger.warning(
                    "row missing %s; skipping parquet write",
                    self.doc_name_field,
                )
                continue
            out_path = Path(self.path) / f"{doc_name}{PARQUET_EXTENSION}"
            # Build a 1-row DataFrame preserving column order from df.
            one_row_df = pd.DataFrame([row.to_dict()], columns=df.columns)
            one_row_df.to_parquet(out_path, index=False)
            written.append(str(out_path))

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=written,
            _metadata={**task._metadata, "format": "parquet_per_doc"},
            _stage_perf=task._stage_perf,
        )


# --------------------------------------------------------------------- helpers


def _doc_name_or_empty(value: Any) -> str:
    """Return a non-empty ``doc_name`` or ``""`` for skip-this-row semantics.

    Pandas coerces missing ``doc_name`` cells to ``NaN``; stringifying
    that blindly yields ``"nan"`` and lands a file called
    ``nan.<ext>`` on disk. Treat NaN / None / empty / whitespace as a
    skip signal instead.
    """
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if value is None:
        return ""
    return str(value).strip()


def _project_columns(
    df: pd.DataFrame,
    *,
    fields: list[str] | None,
    drop_fields: tuple[str, ...],
) -> pd.DataFrame:
    """Return ``df`` with user-requested column projection applied.

    * If ``fields`` is set, keep those (preserving declared order); any
      requested column missing from the frame is silently skipped.
    * Otherwise, drop ``drop_fields`` (typically binary columns like
      ``pdf_bytes`` that can't round-trip through JSON / parquet).
    """
    if fields is not None:
        keep = [c for c in fields if c in df.columns]
        return df[keep] if keep else df.iloc[:, :0]
    drop = [c for c in drop_fields if c in df.columns]
    return df.drop(columns=drop) if drop else df


__all__ = [
    "JSONL_EXTENSION",
    "JsonlPerDocWriter",
    "MARKDOWN_EXTENSION",
    "META_EXTENSION",
    "MarkdownPerDocWriter",
    "MarkdownReader",
    "MarkdownReaderStage",
    "PARQUET_EXTENSION",
    "ParquetPerDocWriter",
]
