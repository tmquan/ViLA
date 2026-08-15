"""Pipeline factories for the PDF triage / native-extraction split.

Two pipelines, registered by every PDF-bearing datasite:

``pdf_triage``
    Scan ``pdf/``, classify every document on CPU, and write
    ``manifests/native.jsonl`` + ``manifests/deferred.jsonl`` +
    ``manifests/triage_summary.json``. No extraction, no GPU.

``pdf_native``
    Read ``manifests/native.jsonl`` through Curator's own
    :class:`PDFPartitioningStage`, extract the embedded text layer, and
    write Curator interleaved parquet plus ViLA's ``md/`` per-doc tier.

Both are additive and never appear in a site's ``ALL_PIPELINES_ORDER``.
They were wired as explicit stages in the retired distributed anle CLI
(now under ``packages/datasites/anle/_legacy/``); the ``pdf_triage`` /
``pdf_native`` stages themselves (:mod:`packages.parser.triage`,
:mod:`packages.parser.native_interleaved`) remain live and are composed
directly by the in-process GB10 drivers.

The deferred manifest is left for a later GPU pass. Because it is
written in Curator's manifest format, that pass needs no glue::

    from nemo_curator.stages.interleaved.pdf.nemotron_parse.composite import (
        NemotronParsePDFReader,
    )

    NemotronParsePDFReader(
        manifest_path="data/<host>/manifests/deferred.jsonl",
        pdf_dir="data/<host>/pdf",
        model_path="nvidia/NVIDIA-Nemotron-Parse-v1.2",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.interleaved.io.writers import InterleavedParquetWriterStage
from nemo_curator.stages.interleaved.pdf.nemotron_parse.partitioning import (
    PDFPartitioningStage,
)

from packages.common.base import SiteLayout, build_layout
from packages.parser.native_interleaved import (
    InterleavedMarkdownSidecarStage,
    NativePdfExtractStage,
)
from packages.parser.triage import (
    DEFAULT_EXTENSIONS,
    NATIVE_MANIFEST,
    PdfSourceManifestStage,
    PdfTriageStage,
    TriageManifestWriter,
)

logger = logging.getLogger(__name__)


def _triage_cfg(cfg: Any) -> Any:
    """Return ``cfg.triage``, tolerating configs that predate the block.

    Site YAML files written before this pipeline existed have no
    ``triage:`` key. Rather than force every site to add one, fall back
    to the schema defaults so an un-migrated config still runs.
    """
    triage = cfg.get("triage") if hasattr(cfg, "get") else None
    if triage is None:
        from packages.common.schemas import TriageCfg

        return TriageCfg()
    return triage


def _manifest_dir(layout: SiteLayout, triage: Any) -> Path:
    subdir = str(_get(triage, "manifest_subdir", "manifests"))
    path = layout.site_root / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _interleaved_dir(layout: SiteLayout, triage: Any) -> Path:
    subdir = str(_get(triage, "interleaved_subdir", "interleaved"))
    path = layout.parquet_dir / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get(cfg: Any, key: str, default: Any) -> Any:
    """Read ``key`` from an OmegaConf node or a plain dataclass."""
    if hasattr(cfg, "get") and not isinstance(cfg, (str, bytes)):
        try:
            value = cfg.get(key, default)
        except TypeError:
            value = getattr(cfg, key, default)
    else:
        value = getattr(cfg, key, default)
    return default if value is None else value


def build_pdf_triage_pipeline(cfg: Any) -> Pipeline:
    """Classify ``pdf/`` into native and deferred manifests. CPU only."""
    layout = build_layout(cfg)
    triage = _triage_cfg(cfg)
    manifest_dir = _manifest_dir(layout, triage)

    extensions = tuple(_get(triage, "extensions", list(DEFAULT_EXTENSIONS)))
    limit = int(cfg.limit) if cfg.get("limit") else None

    return Pipeline(
        name=f"{cfg.host}-pdf-triage",
        description=(
            "PDF triage: classify native vs. deferred (OCR / repair) and "
            "write Curator-format manifests. No GPU, no rasterization."
        ),
        stages=[
            PdfSourceManifestStage(
                pdf_dir=str(layout.pdf_dir),
                manifest_dir=str(manifest_dir),
                extensions=extensions,
                pdfs_per_task=int(_get(triage, "pdfs_per_task", 32)),
                max_pdfs=limit,
                dataset_name=f"{cfg.host}-triage",
            ),
            PdfTriageStage(
                pdf_dir=str(layout.pdf_dir),
                min_chars=int(_get(triage, "min_local_chars", 50)),
                max_lossy=float(_get(triage, "max_lossy_score", 0.05)),
            ),
            TriageManifestWriter(manifest_dir=str(manifest_dir)),
        ],
        config={
            "host": str(cfg.host),
            "pdf_dir": str(layout.pdf_dir),
            "manifest_dir": str(manifest_dir),
        },
    )


def build_pdf_native_pipeline(cfg: Any) -> Pipeline:
    """Extract the native cohort into interleaved parquet + ``md/``.

    Raises
    ------
    FileNotFoundError
        When ``manifests/native.jsonl`` is missing, i.e. ``pdf_triage``
        has not been run for this site yet. Failing here with a
        pointed message beats letting Curator's partitioner open a
        nonexistent path deep inside a Ray worker.
    """
    layout = build_layout(cfg)
    triage = _triage_cfg(cfg)
    manifest_dir = _manifest_dir(layout, triage)
    manifest_path = manifest_dir / NATIVE_MANIFEST

    if not manifest_path.exists():
        msg = (
            f"native manifest not found: {manifest_path}\n"
            f"Produce it by running the pdf_triage stage first "
            f"(packages.pipeline.pdf_triage, composed by the in-process parse driver)."
        )
        raise FileNotFoundError(msg)

    interleaved_dir = _interleaved_dir(layout, triage)
    skip_existing = bool(_get(triage, "skip_existing", False))
    write_md = bool(_get(triage, "write_markdown_sidecar", True))
    limit = int(cfg.limit) if cfg.get("limit") else None

    stages: list[Any] = [
        PDFPartitioningStage(
            manifest_path=str(manifest_path),
            pdfs_per_task=int(_get(triage, "pdfs_per_task", 32)),
            max_pdfs=limit,
            dataset_name=f"{cfg.host}-native",
        ),
        NativePdfExtractStage(
            pdf_dir=str(layout.pdf_dir),
            manifest_dir=str(manifest_dir),
            md_dir=str(layout.md_dir),
            skip_existing=skip_existing,
        ),
    ]
    if write_md:
        stages.append(
            InterleavedMarkdownSidecarStage(
                md_dir=str(layout.md_dir),
                skip_existing=skip_existing,
            )
        )
    stages.append(
        InterleavedParquetWriterStage(
            path=str(interleaved_dir),
            # Page text carries no binary payload, so there is nothing
            # to fetch; leaving materialization on would make the writer
            # scan every row for absent image bytes.
            materialize_on_write=False,
        )
    )

    return Pipeline(
        name=f"{cfg.host}-pdf-native",
        description=(
            "Native PDF extraction: embedded text layer -> Curator "
            "interleaved parquet (+ md/ per-doc tier). No GPU."
        ),
        stages=stages,
        config={
            "host": str(cfg.host),
            "manifest_path": str(manifest_path),
            "interleaved_dir": str(interleaved_dir),
            "md_dir": str(layout.md_dir),
        },
    )


#: Registry fragment every PDF-bearing datasite merges into its own
#: ``PIPELINES`` dict, so the two pipelines stay identical across sites.
PDF_TRIAGE_PIPELINES: dict[str, Any] = {
    "pdf_triage": build_pdf_triage_pipeline,
    "pdf_native": build_pdf_native_pipeline,
}


__all__ = [
    "PDF_TRIAGE_PIPELINES",
    "build_pdf_native_pipeline",
    "build_pdf_triage_pipeline",
]
