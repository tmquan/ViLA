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
import re
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

# Lone UTF-16 surrogates (the U+D800..U+DFFF range) are reserved for
# UTF-16 codec use and are NOT valid characters in any well-formed
# Unicode string. Python's str type accepts them (str is internally
# code-point-oriented, not codec-validated) but UTF-8 forbids them at
# encode time, so any ``str.encode("utf-8")`` or ``.write_text(...,
# encoding="utf-8")`` on a surrogate-bearing string raises
# UnicodeEncodeError. We see this in practice from upstream PDFs whose
# /Info dict carries a malformed UTF-16BE field where one half of a
# surrogate pair got dropped or corrupted in transit. ~1 row per
# ~1M survived a 1.06M-doc parse before the writer crashed mid-run on
# a low surrogate (U+DF58); the fix is to replace lone surrogates with
# U+FFFD (REPLACEMENT CHARACTER, the Unicode-canonical "this codepoint
# was unrepresentable" glyph) before encoding.
_SURROGATE_RE = re.compile("[\ud800-\udfff]")
_REPLACEMENT_CHAR = "\ufffd"


def _scrub_surrogates(value: str) -> str:
    """Replace lone UTF-16 surrogates with U+FFFD, leaving clean strings untouched.

    Fast path: strings with no surrogates return the SAME object (no
    copy, no allocation). Strings with one or more surrogates return a
    new string with every U+D800..U+DFFF codepoint replaced by U+FFFD.

    Idempotent: U+FFFD is outside the surrogate range, so a re-scrub
    of an already-scrubbed string is a no-op.
    """
    if not value or _SURROGATE_RE.search(value) is None:
        return value
    return _SURROGATE_RE.sub(_REPLACEMENT_CHAR, value)


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

    ``skip_existing`` (default True) makes the writer idempotent: rows
    whose ``<doc_name>.md`` is already on disk are passed over without
    rewriting either the body or the meta sidecar. Disable it when the
    parser has been upgraded and you want every md re-emitted.
    """

    path: str
    doc_name_field: str = "doc_name"
    markdown_field: str = "markdown"
    drop_fields: tuple[str, ...] = ("pdf_bytes",)
    skip_existing: bool = True
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
            doc_name = _doc_name_or_empty(row.get(self.doc_name_field))
            if not doc_name:
                logger.warning(
                    "row missing %s; skipping markdown write",
                    self.doc_name_field,
                )
                continue
            md_path = Path(self.path) / f"{doc_name}{MARKDOWN_EXTENSION}"
            meta_path = Path(self.path) / f"{doc_name}{META_EXTENSION}"
            if self.skip_existing and md_path.exists():
                # Idempotent resume: leave the existing md untouched.
                # We still report the path in the FileGroupTask so
                # downstream stages can address the row.
                written.append(str(md_path))
                if meta_path.exists():
                    written.append(str(meta_path))
                continue
            markdown = str(row.get(self.markdown_field) or "")
            # Contract: upstream drops empty-markdown rows (see
            # PdfParseStage). Defensive: never write a 0-byte <doc>.md
            # so the extract / embed pipelines never observe one.
            if not markdown.strip():
                logger.warning(
                    "row doc_name=%s has empty %s; skipping markdown write",
                    doc_name, self.markdown_field,
                )
                continue

            # Scrub lone UTF-16 surrogates (would crash .write_text(...,
            # encoding="utf-8")). The scrub is a no-op on clean input;
            # we log iff it actually mutates the body so operators can
            # audit which docs had body-level surrogates.
            scrubbed = _scrub_surrogates(markdown)
            if scrubbed is not markdown:
                logger.warning(
                    "doc_name=%s: scrubbed UTF-16 surrogate(s) from markdown "
                    "body (replaced with U+FFFD)", doc_name,
                )
                markdown = scrubbed
            md_path.write_text(markdown, encoding="utf-8")
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
    """Coerce pandas / numpy scalars to JSON-friendly Python types.

    String values are passed through :func:`_scrub_surrogates` so any
    lone UTF-16 surrogate inherited from upstream PDF metadata is
    replaced with U+FFFD before the value reaches ``json.dumps`` +
    ``.encode("utf-8")``. The scrub is silent (operators audit by
    grepping output files for U+FFFD) and is a no-op on clean strings.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return _scrub_surrogates(value)
    if isinstance(value, (int, float, bool)):
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

            # Contract: "there must not be empty markdown" downstream.
            # Skip stale 0-byte / whitespace-only .md files (e.g. from
            # an earlier pipeline version that did not drop empty
            # parser output) rather than emit a ghost row.
            if not markdown.strip():
                logger.warning(
                    "skipping empty markdown file %s", path,
                )
                continue

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


_UNSAFE_DOC_NAME_CHARS = ("/", "\\", "\x00")


def _doc_name_or_empty(value: Any) -> str:
    """Return a safe, non-empty ``doc_name`` or ``""`` for skip semantics.

    Pandas coerces missing ``doc_name`` cells to ``NaN``; stringifying
    that blindly yields ``"nan"`` and lands a file called
    ``nan.<ext>`` on disk. Treat NaN / None / empty / whitespace as a
    skip signal instead.

    Also rejects values that could escape the writer's target directory
    (path separators, ``..``, absolute paths, NUL bytes, dot-only
    components). The contract is: the returned string -- when joined to
    a writer's ``self.path`` -- never resolves outside that directory.
    Sites that need exotic doc_name shapes must sanitize upstream
    (e.g. URL-escape) before reaching the writer.
    """
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if value is None:
        return ""
    candidate = str(value).strip()
    if not candidate:
        return ""
    if any(ch in candidate for ch in _UNSAFE_DOC_NAME_CHARS):
        logger.warning(
            "unsafe doc_name %r contains path separator or NUL; skipping",
            candidate,
        )
        return ""
    if candidate in (".", "..") or candidate.startswith(".."):
        logger.warning(
            "unsafe doc_name %r resolves outside writer directory; skipping",
            candidate,
        )
        return ""
    # Final invariant: the candidate must equal its own basename.
    if Path(candidate).name != candidate:
        logger.warning(
            "unsafe doc_name %r does not equal its basename; skipping",
            candidate,
        )
        return ""
    return candidate


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
    "MARKDOWN_EXTENSION",
    "META_EXTENSION",
    "PARQUET_EXTENSION",
    "JsonlPerDocWriter",
    "MarkdownPerDocWriter",
    "MarkdownReader",
    "MarkdownReaderStage",
    "ParquetPerDocWriter",
]
