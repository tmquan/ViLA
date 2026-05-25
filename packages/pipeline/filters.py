"""Pipeline filter stages.

Generic :class:`ProcessingStage` shapes that drop rows from a
:class:`DocumentBatch` based on cheap predicates. Used to short-circuit
expensive downstream stages (parser, embedder) on already-completed
work so a re-run is a tight no-op pass over the file list.

Resume contract: every stage here is *purely* in-memory. The on-disk
state (``md/<doc>.md``, ``parquet/embeddings/<doc>.parquet``, ...) is
the source of truth; the filter stage only looks at what already
exists and removes the corresponding row from the in-flight batch.
Nothing on disk is mutated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

logger = logging.getLogger(__name__)


@dataclass
class SkipExistingMarkdownFilter(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Drop rows whose ``<doc_name>.md`` already exists in ``md_dir``.

    Inserted between the iterator/extractor stage and ``PdfParseStage``,
    this turns a re-run of the parser into a tight scan: PDFs whose
    markdown sibling is already on disk never reach the parser, so
    pypdf doesn't re-read them, the writer doesn't rewrite them, and
    the only cost is one ``Path.exists()`` per row.

    The filter is fail-open: if ``md_dir`` doesn't exist (cold start),
    nothing is dropped. Empty markdown files (zero bytes / whitespace
    only) do *not* count as already-done -- those are dropped by the
    parser stage's existing empty-markdown invariant on the next pass,
    so they get re-attempted.

    All-rows-dropped batches return an empty :class:`DocumentBatch`,
    which downstream stages handle as a no-op.
    """

    md_dir: str
    doc_name_field: str = "doc_name"
    name: str = "skip_existing_markdown_filter"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))
    batch_size: int = 1

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [self.doc_name_field])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], [])

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self._md_dir = Path(self.md_dir)

    def process(self, task: DocumentBatch) -> DocumentBatch:
        df = task.to_pandas()
        if df.empty or self.doc_name_field not in df.columns:
            return task
        if not getattr(self, "_md_dir", None):
            self.setup(None)

        md_dir = self._md_dir
        if not md_dir.exists():
            return task

        def _has_md(doc_name: Any) -> bool:
            if doc_name is None:
                return False
            name = str(doc_name).strip()
            if not name:
                return False
            md_path = md_dir / f"{name}.md"
            try:
                # Treat zero-byte / whitespace-only md files as missing
                # so the parser re-attempts them. Stat is cheap; the
                # full read only happens on the (rare) zero-byte case.
                st = md_path.stat()
            except OSError:
                return False
            return st.st_size > 0

        keep_mask = df[self.doc_name_field].map(_has_md).map(lambda x: not x)
        n_in = len(df)
        df = df.loc[keep_mask].reset_index(drop=True)
        dropped = n_in - len(df)
        if dropped:
            logger.debug(
                "SkipExistingMarkdownFilter: dropped %d/%d rows already on disk",
                dropped, n_in,
            )

        return DocumentBatch(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=df,
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


__all__ = ["SkipExistingMarkdownFilter"]
