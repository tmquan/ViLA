"""NeMo Curator stage: stored ``<id>.html.gz`` pages -> rich Q&A records.

Composes the two hoi-dap Curator components — :class:`TVPLQAIterator`
(:class:`DocumentIterator`) then :class:`TVPLQAExtractor`
(:class:`DocumentExtractor`, incl. legal-citation extraction) — into a single
:class:`ProcessingStage`. Consumes a :class:`DocumentBatch` whose ``file_path``
column lists page files; emits a ``DocumentBatch`` of extracted Q&A records
(non-Q&A / unparseable pages are dropped). Driven in-process on the GB10 (the
same pattern the anle/congbobanan extractions use).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch

from packages.datasites.thuvienphapluat_hdpl.components.extractor import TVPLQAExtractor
from packages.datasites.thuvienphapluat_hdpl.components.iterator import TVPLQAIterator

logger = logging.getLogger(__name__)


@dataclass
class TVPLQAExtractStage(ProcessingStage[DocumentBatch, DocumentBatch]):
    """Iterate + extract stored hoi-dap pages into Q&A records."""

    name: str = "tvpl_qa_extract"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    batch_size: int = 1

    _iterator: TVPLQAIterator | None = field(default=None, init=False, repr=False)
    _extractor: TVPLQAExtractor | None = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return (["data"], ["file_path"])

    def outputs(self) -> tuple[list[str], list[str]]:
        return (["data"], TVPLQAExtractor._OUT)

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        self._iterator = TVPLQAIterator()
        self._extractor = TVPLQAExtractor()

    def process(self, task: DocumentBatch) -> DocumentBatch:
        if self._iterator is None:
            self.setup(None)
        assert self._iterator is not None and self._extractor is not None
        rows: list[dict] = []
        for file_path in task.to_pandas()["file_path"]:
            try:
                for rec in self._iterator.iterate(str(file_path)):
                    qa = self._extractor.extract(rec)
                    if qa:
                        rows.append(qa)
            except Exception as exc:  # noqa: BLE001 — one bad page never fails the batch
                logger.warning(f"extract failed for {file_path}: {type(exc).__name__}: {exc}")
        return DocumentBatch(
            task_id=task.task_id, dataset_name=task.dataset_name,
            data=pd.DataFrame(rows, columns=TVPLQAExtractor._OUT))


__all__ = ["TVPLQAExtractStage"]
