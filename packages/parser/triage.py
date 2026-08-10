"""Stage 2a: GPU-free PDF triage into native and deferred cohorts.

Runs :func:`packages.parser.types.classify_pdf` over a corpus and
splits it in two, without ever loading a vision model:

``manifests/native.jsonl``
    PDFs with a trustworthy embedded text layer. Consumed by the
    ``pdf_native`` pipeline (:mod:`packages.parser.native_interleaved`).
``manifests/deferred.jsonl``
    Everything else, annotated with the detection reason, the raw
    classification signals, and a ranked list of remediation models
    from :mod:`packages.parser.ocr_models`.

Both files are written in NeMo Curator's manifest format -- one JSON
object per line carrying at minimum ``file_name`` and ``url`` -- so
either can be fed straight back into Curator's own
:class:`~nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning.PDFPartitioningStage`.
In particular the deferred manifest drops directly into
:class:`NemotronParsePDFReader` when the OCR run eventually happens::

    NemotronParsePDFReader(
        manifest_path="data/<host>/manifests/deferred.jsonl",
        pdf_dir="data/<host>/pdf",
    )

Stage chain::

    PdfSourceManifestStage   _EmptyTask    -> FileGroupTask
    PdfTriageStage           FileGroupTask -> DocumentBatch
    TriageManifestWriter     DocumentBatch -> FileGroupTask
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import (
    PDFPartitioningStage,
)
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import DocumentBatch, FileGroupTask, _EmptyTask

from packages.parser.base import ParserAlgorithm
from packages.parser.ocr_models import primary_model, suggested_models_as_dicts
from packages.parser.types import (
    DEFAULT_MAX_LOSSY,
    DEFAULT_MIN_CHARS,
    DETECTION_REASON,
    VietnameseLegalPDFType,
    classify_pdf,
    deferred_class,
    is_native,
)

logger = logging.getLogger(__name__)

#: File extensions the triage pass considers. Mirrors the extension
#: list the existing parse pipelines pass to ``FilePartitioningStage``.
DEFAULT_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".doc")

#: Sidecar written next to each downloaded binary holding its source
#: URL (one line, no trailing structure). Produced by the datasite
#: downloaders; absent for corpora fetched by other means.
URL_SIDECAR_EXTENSION = ".url"

SOURCE_MANIFEST = "source.jsonl"
NATIVE_MANIFEST = "native.jsonl"
DEFERRED_MANIFEST = "deferred.jsonl"
TRIAGE_SUMMARY = "triage_summary.json"


def _read_url_sidecar(pdf_path: Path) -> str:
    """Return the source URL for ``pdf_path``, or ``""`` when unknown."""
    sidecar = pdf_path.with_suffix(URL_SIDECAR_EXTENSION)
    try:
        return sidecar.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


@dataclass
class PdfSourceManifestStage(ProcessingStage[_EmptyTask, FileGroupTask]):
    """Scan ``pdf_dir``, write ``source.jsonl``, emit partitioned tasks.

    Curator's :class:`PDFPartitioningStage` reads a manifest but cannot
    produce one, and a Curator pipeline can only have a single
    ``_EmptyTask`` source -- so scanning and partitioning have to happen
    in one stage rather than two chained ones.

    Rather than reimplement the packing, this stage writes the manifest
    and then delegates to a real :class:`PDFPartitioningStage` instance,
    so the ``FileGroupTask`` payloads are produced by Curator's own code
    and stay byte-compatible with what the deferred manifest will later
    feed to :class:`NemotronParsePDFReader`.

    Parameters
    ----------
    pdf_dir
        Directory holding the raw binaries.
    manifest_dir
        Where ``source.jsonl`` is written.
    extensions
        Which file extensions to enumerate.
    pdfs_per_task
        Documents packed into each downstream ``FileGroupTask``.
    max_pdfs
        Cap on total documents, for smoke runs.
    """

    pdf_dir: str
    manifest_dir: str
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    pdfs_per_task: int = 32
    max_pdfs: int | None = None
    dataset_name: str = "pdf_triage"
    name: str = "pdf_source_manifest"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers_per_node": 1}

    def _scan(self) -> list[dict[str, str]]:
        """Enumerate source documents in a stable, resumable order."""
        root = Path(self.pdf_dir)
        if not root.is_dir():
            msg = f"pdf_dir does not exist or is not a directory: {root}"
            raise FileNotFoundError(msg)

        wanted = {e.lower() for e in self.extensions}
        entries: list[dict[str, str]] = []
        # scandir + sort keeps ordering deterministic across runs so a
        # --limit smoke run always covers the same slice of the corpus.
        with os.scandir(root) as it:
            names = sorted(
                e.name for e in it if e.is_file() and Path(e.name).suffix.lower() in wanted
            )
        for file_name in names:
            entries.append(
                {
                    "file_name": file_name,
                    "url": _read_url_sidecar(root / file_name),
                }
            )
        return entries

    def _write_manifest(self) -> str:
        entries = self._scan()
        manifest_root = Path(self.manifest_dir)
        manifest_root.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_root / SOURCE_MANIFEST
        with manifest_path.open("w", encoding="utf-8") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(
            "PdfSourceManifestStage: wrote %d entries to %s",
            len(entries),
            manifest_path,
        )
        return str(manifest_path)

    def process(self, task: _EmptyTask) -> list[FileGroupTask]:
        manifest_path = self._write_manifest()
        partitioner = PDFPartitioningStage(
            manifest_path=manifest_path,
            pdfs_per_task=self.pdfs_per_task,
            max_pdfs=self.max_pdfs,
            dataset_name=self.dataset_name,
        )
        return partitioner.process(task)


@dataclass
class PdfTriageStage(ProcessingStage[FileGroupTask, DocumentBatch]):
    """Classify each document in a ``FileGroupTask``; emit one row each.

    Reads bytes from ``<pdf_dir>/<file_name>`` and runs the pypdf-only
    :func:`classify_pdf`. No document is ever rasterized here -- the
    whole point of the stage is to decide *whether* rasterization will
    be needed, cheaply and on CPU.

    Emits one row per input document carrying ``pdf_type``, the raw
    classification ``signals``, a ``route`` of ``"native"`` or
    ``"deferred"``, and (for deferred rows) the ranked model
    suggestions. Nothing is filtered out: the writer downstream needs
    to see every document to produce an exhaustive split.
    """

    pdf_dir: str
    min_chars: int = DEFAULT_MIN_CHARS
    max_lossy: float = DEFAULT_MAX_LOSSY
    name: str = "pdf_triage"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _local: ParserAlgorithm | None = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return (
            ["data"],
            ["file_name", "url", "pdf_type", "route", "signals"],
        )

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        # Imported here rather than at module scope so the pypdf
        # dependency is resolved on the worker, matching PdfParseStage.
        from packages.parser.pypdf import PypdfParser

        self._local = PypdfParser()

    def _read_bytes(self, file_name: str) -> tuple[bytes | None, str | None]:
        path = Path(self.pdf_dir) / file_name
        try:
            return path.read_bytes(), None
        except OSError as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def _row(self, entry: dict[str, Any]) -> dict[str, Any]:
        file_name = str(entry.get("file_name") or "")
        url = str(entry.get("url") or "")
        doc_name = Path(file_name).stem

        pdf_bytes, read_error = self._read_bytes(file_name)
        if pdf_bytes is None:
            # Unreadable on disk is a corpus-integrity problem, not a
            # document-content problem, but it still belongs in the
            # deferred manifest so it is not silently lost.
            pdf_type = VietnameseLegalPDFType.CORRUPTED
            signals: dict[str, Any] = {"error": read_error}
        else:
            pdf_type, signals = classify_pdf(
                pdf_bytes,
                local=self._local,
                min_chars=self.min_chars,
                max_lossy=self.max_lossy,
            )

        native = is_native(pdf_type)
        return {
            "file_name": file_name,
            "doc_name": doc_name,
            "url": url,
            "pdf_type": pdf_type.value,
            "route": "native" if native else "deferred",
            "deferred_class": deferred_class(pdf_type),
            "reason": DETECTION_REASON[pdf_type],
            "signals": signals,
            "suggested_model": None if native else primary_model(pdf_type),
            "suggested_models": [] if native else suggested_models_as_dicts(pdf_type),
            "file_size_bytes": len(pdf_bytes) if pdf_bytes else 0,
            "triaged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }

    def process(self, task: FileGroupTask) -> DocumentBatch | None:
        if self._local is None:
            self.setup(None)

        rows: list[dict[str, Any]] = []
        for raw in task.data:
            try:
                entry = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (TypeError, ValueError):
                logger.warning("PdfTriageStage: unparseable manifest entry %r", raw)
                continue
            rows.append(self._row(entry))

        if not rows:
            return None

        return DocumentBatch(
            task_id=f"{task.task_id}_triaged",
            dataset_name=task.dataset_name,
            data=pd.DataFrame(rows),
            _metadata=task._metadata,
            _stage_perf=task._stage_perf,
        )


@dataclass
class TriageManifestWriter(ProcessingStage[DocumentBatch, FileGroupTask]):
    """Split triaged rows into the native and deferred manifests.

    Pinned to a single worker (:meth:`num_workers` returns 1) so the two
    manifests can be appended through long-lived file handles without
    cross-worker interleaving. Sharding per task and merging afterwards
    would avoid the pin, but a Curator pipeline has no reduce step to
    hang the merge on, and the write itself is a few hundred bytes of
    JSON per document -- orders of magnitude cheaper than the pypdf
    parse feeding it, so the single writer never becomes the bottleneck.

    The handles are opened in :meth:`setup` with mode ``"w"``, so each
    run starts from empty manifests rather than accumulating duplicates
    across re-runs. ``triage_summary.json`` is rewritten after every
    batch (it is only a handful of counters) so a partial run still
    leaves usable totals behind if the pipeline is interrupted.
    """

    manifest_dir: str
    name: str = "triage_manifest_writer"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    _native_fh: TextIO | None = field(default=None, init=False, repr=False)
    _deferred_fh: TextIO | None = field(default=None, init=False, repr=False)
    _counts: Counter[str] = field(default_factory=Counter, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["file_name", "pdf_type", "route"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def num_workers(self) -> int | None:
        return 1

    def xenna_stage_spec(self) -> dict[str, Any]:
        return {"num_workers_per_node": 1}

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        root = Path(self.manifest_dir)
        root.mkdir(parents=True, exist_ok=True)
        self._native_fh = (root / NATIVE_MANIFEST).open("w", encoding="utf-8")
        self._deferred_fh = (root / DEFERRED_MANIFEST).open("w", encoding="utf-8")
        self._counts = Counter()

    def teardown(self) -> None:
        for fh in (self._native_fh, self._deferred_fh):
            if fh is not None and not fh.closed:
                fh.flush()
                fh.close()
        self._native_fh = None
        self._deferred_fh = None

    def _write_summary(self) -> None:
        total = sum(
            v for k, v in self._counts.items() if k.startswith("pdf_type.")
        )
        summary = {
            "total": total,
            "native": self._counts.get("route.native", 0),
            "deferred": self._counts.get("route.deferred", 0),
            "by_pdf_type": {
                k.removeprefix("pdf_type."): v
                for k, v in sorted(self._counts.items())
                if k.startswith("pdf_type.")
            },
            "by_deferred_class": {
                k.removeprefix("deferred_class."): v
                for k, v in sorted(self._counts.items())
                if k.startswith("deferred_class.")
            },
            "native_fraction": (
                self._counts.get("route.native", 0) / total if total else 0.0
            ),
            "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path = Path(self.manifest_dir) / TRIAGE_SUMMARY
        path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def process(self, task: DocumentBatch) -> FileGroupTask:
        if self._native_fh is None or self._deferred_fh is None:
            self.setup(None)
        assert self._native_fh is not None
        assert self._deferred_fh is not None

        df = task.to_pandas()
        for _, row in df.iterrows():
            native = str(row.get("route")) == "native"
            pdf_type = str(row.get("pdf_type"))
            self._counts[f"pdf_type.{pdf_type}"] += 1
            self._counts["route.native" if native else "route.deferred"] += 1

            # Both manifests lead with file_name + url so Curator's
            # PDFPartitioningStage can read either one unmodified.
            record: dict[str, Any] = {
                "file_name": row.get("file_name"),
                "url": row.get("url"),
                "doc_name": row.get("doc_name"),
                "pdf_type": pdf_type,
                "signals": row.get("signals"),
                "triaged_at": row.get("triaged_at"),
            }
            if native:
                fh = self._native_fh
            else:
                dclass = row.get("deferred_class")
                self._counts[f"deferred_class.{dclass}"] += 1
                record.update(
                    {
                        "deferred_class": dclass,
                        "reason": row.get("reason"),
                        "suggested_model": row.get("suggested_model"),
                        "suggested_models": row.get("suggested_models"),
                    }
                )
                fh = self._deferred_fh
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        self._native_fh.flush()
        self._deferred_fh.flush()
        self._write_summary()

        return FileGroupTask(
            task_id=task.task_id,
            dataset_name=task.dataset_name,
            data=[
                str(Path(self.manifest_dir) / NATIVE_MANIFEST),
                str(Path(self.manifest_dir) / DEFERRED_MANIFEST),
            ],
            _metadata={**task._metadata, "format": "triage_manifest"},
            _stage_perf=task._stage_perf,
        )


__all__ = [
    "DEFAULT_EXTENSIONS",
    "DEFERRED_MANIFEST",
    "NATIVE_MANIFEST",
    "SOURCE_MANIFEST",
    "TRIAGE_SUMMARY",
    "PdfSourceManifestStage",
    "PdfTriageStage",
    "TriageManifestWriter",
]
