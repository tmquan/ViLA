"""Stage 2b: native-PDF extraction onto Curator's interleaved schema.

``nemo_curator.stages.interleaved.pdf`` ships exactly one PDF path, and
it is entirely Nemotron-Parse: :class:`PDFPreprocessStage` rasterizes
every page to PNG and :class:`NemotronParseInferenceStage` runs a GPU
model over the images. There is no native text-layer reader in that
namespace.

For the ~94% of a Vietnamese legal corpus that already carries a clean
embedded text layer, rasterizing is pure waste -- it converts text we
already have into pixels so a GPU can convert them back. So this module
keeps Curator's *contracts* and swaps only the engine:

* input is a :class:`FileGroupTask` of JSON manifest entries, produced
  by Curator's own :class:`PDFPartitioningStage`,
* output is an :class:`InterleavedBatch` whose rows conform to
  :data:`INTERLEAVED_SCHEMA` and carry the same four user columns
  (``url`` / ``page_number`` / ``pdf_name`` / ``element_class``) that
  Curator's :func:`build_interleaved_rows` emits,
* the terminal writer is Curator's own
  :class:`InterleavedParquetWriterStage`.

The result is a drop-in substitute for :class:`NemotronParsePDFReader`
on the native cohort: same schema in, same schema out, no GPU.

Two stages::

    NativePdfExtractStage           FileGroupTask   -> InterleavedBatch
    InterleavedMarkdownSidecarStage InterleavedBatch -> InterleavedBatch
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import FileGroupTask, InterleavedBatch
from nemo_curator.tasks.interleaved import INTERLEAVED_SCHEMA

from packages.parser.base import ParserAlgorithm
from packages.pipeline.io import (
    MARKDOWN_EXTENSION,
    META_EXTENSION,
    _scrub_surrogates,
)

logger = logging.getLogger(__name__)

#: Model id recorded on every row this module produces. Not a model in
#: any real sense, but ``parser_model`` is the provenance column the
#: rest of the pipeline reads, and "which engine wrote this text" is
#: exactly the question it answers.
NATIVE_EXTRACTOR_ID = "local/pypdf-native"

#: User columns Curator's own ``build_interleaved_rows`` attaches
#: alongside the reserved schema. Mirrored exactly so a downstream
#: consumer cannot tell a native batch from a Nemotron-Parse one.
USER_COLUMNS: tuple[str, ...] = ("url", "page_number", "pdf_name", "element_class")

#: Written per task (never appended across workers, so no race) for any
#: document that passed triage as native but yielded no text at extract
#: time -- normally a sign the file changed on disk between the two runs.
REJECTS_DIRNAME = "native_rejects"


def build_native_interleaved_rows(
    sample_id: str,
    url: str,
    pdf_name: str,
    pages: list[dict[str, Any]],
    *,
    pdf_type: str | None = None,
    signals: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Convert pypdf per-page output into interleaved-schema rows.

    Mirrors the row shape of
    :func:`nemo_curator.stages.interleaved.pdf.nemotron_parse.utils.build_interleaved_rows`:
    a single metadata row at ``position=-1`` followed by content rows
    numbered from zero.

    The native path emits one ``text`` row per page rather than one per
    layout element, because pypdf has no layout model -- it returns a
    page's text as an undifferentiated block. ``source_ref`` therefore
    carries a ``bbox`` of ``None``, which is the honest answer, and
    ``element_class`` is a flat ``"Text"``. Pages that extracted empty
    are skipped rather than emitted as blank rows.
    """
    rows: list[dict[str, Any]] = [
        {
            "sample_id": sample_id,
            "position": -1,
            "modality": "metadata",
            "content_type": "application/json",
            "text_content": json.dumps(
                {
                    "url": url,
                    "pdf_name": pdf_name,
                    "num_pages": len(pages),
                    "pdf_type": pdf_type,
                    "signals": signals or {},
                    "extractor": NATIVE_EXTRACTOR_ID,
                },
                ensure_ascii=False,
                default=str,
            ),
            "binary_content": None,
            "source_ref": None,
            "url": url,
            "page_number": None,
            "pdf_name": pdf_name,
            "element_class": None,
        }
    ]

    position = 0
    for page_num, page in enumerate(pages):
        text = str(page.get("markdown") or "").strip()
        if not text:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "position": position,
                "modality": "text",
                "content_type": "text/markdown",
                "text_content": _scrub_surrogates(text),
                "binary_content": None,
                "source_ref": json.dumps({"page": page_num, "bbox": None}),
                "url": url,
                "page_number": page_num,
                "pdf_name": pdf_name,
                "element_class": "Text",
            }
        )
        position += 1

    return rows


def _to_interleaved_table(rows: list[dict[str, Any]]) -> pa.Table:
    """Backfill missing reserved columns, then build the Arrow table.

    ``INTERLEAVED_SCHEMA`` marks ``materialize_error`` (and others)
    nullable but still expects the column to exist. Curator's own
    postprocess stage backfills the same way before converting.
    """
    df = pd.DataFrame(rows)
    for col in INTERLEAVED_SCHEMA.names:
        if col not in df.columns:
            df[col] = None
    return pa.Table.from_pandas(df, preserve_index=False)


@dataclass
class NativePdfExtractStage(ProcessingStage[FileGroupTask, InterleavedBatch]):
    """Extract the embedded text layer of already-triaged native PDFs.

    Deliberately uses the local pypdf backend only, never
    :func:`packages.parser.stage.build_parser`. Routing to an OCR
    fallback here would silently reintroduce the GPU dependency this
    pipeline exists to avoid, and would make the native/deferred split
    a lie. Any document that turns out not to be extractable is
    rejected to a per-task sidecar for re-triage instead.

    Parameters
    ----------
    pdf_dir
        Directory holding the raw binaries named by the manifest.
    manifest_dir
        Parent of the ``native_rejects/`` sidecar directory.
    md_dir
        Consulted only when ``skip_existing`` is set, to drop documents
        whose markdown is already on disk.
    skip_existing
        Resume support: skip documents that already have a ``.md``.
    """

    pdf_dir: str
    manifest_dir: str
    md_dir: str | None = None
    skip_existing: bool = False
    name: str = "native_pdf_extract"
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))

    _local: ParserAlgorithm | None = field(default=None, init=False, repr=False)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        from packages.parser.pypdf import PypdfParser

        self._local = PypdfParser()

    def _write_rejects(self, task_id: str, rejects: list[dict[str, Any]]) -> None:
        if not rejects:
            return
        out_dir = Path(self.manifest_dir) / REJECTS_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{task_id}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rejects:
                fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        logger.warning(
            "NativePdfExtractStage: %d document(s) passed triage but yielded "
            "no text; recorded in %s for re-triage",
            len(rejects),
            path,
        )

    def process(self, task: FileGroupTask) -> InterleavedBatch | None:
        if self._local is None:
            self.setup(None)
        assert self._local is not None

        all_rows: list[dict[str, Any]] = []
        rejects: list[dict[str, Any]] = []

        for raw in task.data:
            try:
                entry = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except (TypeError, ValueError):
                logger.warning("NativePdfExtractStage: bad manifest entry %r", raw)
                continue

            file_name = str(entry.get("file_name") or "")
            if not file_name:
                continue
            url = str(entry.get("url") or "")
            sample_id = Path(file_name).stem

            if (
                self.skip_existing
                and self.md_dir
                and (Path(self.md_dir) / f"{sample_id}{MARKDOWN_EXTENSION}").exists()
            ):
                continue

            path = Path(self.pdf_dir) / file_name
            try:
                pdf_bytes = path.read_bytes()
            except OSError as exc:
                rejects.append(
                    {
                        "file_name": file_name,
                        "url": url,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            try:
                result = self._local.parse(pdf_bytes)
            except Exception as exc:
                # Triage already opened this file successfully, so a
                # raise here is unexpected. Contain it: one bad document
                # must never abort the surrounding Ray task.
                logger.warning(
                    "NativePdfExtractStage: %s raised on %s: %s",
                    type(self._local).__name__,
                    file_name,
                    exc,
                )
                rejects.append(
                    {
                        "file_name": file_name,
                        "url": url,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            pages = list(result.get("pages") or [])
            has_text = any(str(p.get("markdown") or "").strip() for p in pages)
            if not has_text:
                rejects.append(
                    {
                        "file_name": file_name,
                        "url": url,
                        "error": "native extraction produced no text; re-triage",
                    }
                )
                continue

            all_rows.extend(
                build_native_interleaved_rows(
                    sample_id,
                    url,
                    file_name,
                    pages,
                    pdf_type=entry.get("pdf_type"),
                    signals=entry.get("signals"),
                )
            )

        self._write_rejects(task.task_id, rejects)

        if not all_rows:
            return None

        return InterleavedBatch(
            task_id=f"{task.task_id}_native",
            dataset_name=task.dataset_name,
            data=_to_interleaved_table(all_rows),
            _metadata={**task._metadata, "extractor": NATIVE_EXTRACTOR_ID},
            _stage_perf=task._stage_perf,
        )


@dataclass
class InterleavedMarkdownSidecarStage(
    ProcessingStage[InterleavedBatch, InterleavedBatch]
):
    """Write ViLA's ``md/<doc>.md`` + ``.meta.json`` tier, pass batch through.

    The interleaved parquet is the canonical output, but the rest of
    this repo -- the extract, embed and reduce pipelines -- reads
    markdown per document via
    :class:`~packages.pipeline.io.MarkdownReader`. Emitting both keeps
    the new pipeline compatible with everything downstream instead of
    forking the corpus into two incompatible halves.

    Implemented as a pass-through rather than a terminal writer because
    a Curator pipeline is a linear chain: returning the batch unchanged
    lets Curator's :class:`InterleavedParquetWriterStage` still act as
    the real terminal stage.

    Markdown is reassembled from the ``text`` rows in ``position``
    order, one ``## Page N`` heading per page, matching the layout
    :class:`~packages.parser.pypdf.PypdfParser` produces so existing
    markdown consumers see no difference.
    """

    md_dir: str
    skip_existing: bool = True
    name: str = "interleaved_markdown_sidecar"
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.5))

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], []

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:
        Path(self.md_dir).mkdir(parents=True, exist_ok=True)

    def process(self, task: InterleavedBatch) -> InterleavedBatch:
        df = task.to_pandas()
        root = Path(self.md_dir)
        root.mkdir(parents=True, exist_ok=True)

        for sample_id, group in df.groupby("sample_id", sort=False):
            doc_name = str(sample_id)
            md_path = root / f"{doc_name}{MARKDOWN_EXTENSION}"
            meta_path = root / f"{doc_name}{META_EXTENSION}"
            if self.skip_existing and md_path.exists():
                continue

            ordered = group.sort_values("position")
            text_rows = ordered[ordered["modality"] == "text"]
            parts: list[str] = []
            for _, row in text_rows.iterrows():
                body = str(row.get("text_content") or "").strip()
                if not body:
                    continue
                page_no = row.get("page_number")
                heading = (
                    f"## Page {int(page_no) + 1}"
                    if pd.notna(page_no)
                    else "## Page"
                )
                parts.append(f"{heading}\n\n{body}")

            markdown = "\n\n".join(parts)
            # Same contract the rest of the repo enforces: never leave a
            # 0-byte .md behind for the extract pipeline to trip over.
            if not markdown.strip():
                logger.warning(
                    "InterleavedMarkdownSidecarStage: no text rows for %s; "
                    "skipping markdown write",
                    doc_name,
                )
                continue

            meta = self._build_meta(doc_name, ordered)
            md_path.write_text(_scrub_surrogates(markdown), encoding="utf-8")
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

        return task

    @staticmethod
    def _build_meta(doc_name: str, ordered: pd.DataFrame) -> dict[str, Any]:
        """Rebuild the meta sidecar from the sample's metadata row."""
        meta: dict[str, Any] = {
            "doc_name": doc_name,
            "parser_model": NATIVE_EXTRACTOR_ID,
            "parsed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "confidence": None,
        }
        meta_rows = ordered[ordered["modality"] == "metadata"]
        if not meta_rows.empty:
            try:
                payload = json.loads(str(meta_rows.iloc[0]["text_content"]))
            except (TypeError, ValueError):
                payload = {}
            meta.update(
                {
                    "url": payload.get("url"),
                    "detail_url": payload.get("url"),
                    "source": payload.get("pdf_name"),
                    "num_pages": payload.get("num_pages"),
                    "pdf_type": payload.get("pdf_type"),
                    "signals": payload.get("signals"),
                }
            )
        text_count = int((ordered["modality"] == "text").sum())
        meta["num_text_rows"] = text_count
        return meta


__all__ = [
    "NATIVE_EXTRACTOR_ID",
    "USER_COLUMNS",
    "InterleavedMarkdownSidecarStage",
    "NativePdfExtractStage",
    "build_native_interleaved_rows",
]
