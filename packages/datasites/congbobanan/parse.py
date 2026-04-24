"""Parser pipeline: PDF -> Markdown on disk.

Stage chain::

    FilePartitioningStage(pdf_dir, ext=[.pdf])
    -> DocumentIterateExtractStage(CongbobananDocumentIterator,
                                    CongbobananDocumentExtractor)
    -> PdfParseStage
    -> MarkdownPerDocWriter

Reads: ``data/<host>/pdf/*.pdf`` (+ sibling ``<case_id>.html`` /
``<case_id>.url`` sidecars written by the downloader).
Writes: ``data/<host>/md/<case_id>.md`` + ``<case_id>.meta.json``.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.download.base.iterator import (
    DocumentIterateExtractStage,
)

from packages.datasites.congbobanan._shared import build_layout
from packages.datasites.congbobanan.components import (
    CongbobananDocumentExtractor,
    CongbobananDocumentIterator,
)
from packages.parser.stage import PdfParseStage
from packages.pipeline.io import MarkdownPerDocWriter


def build_parse_pipeline(cfg: Any) -> Pipeline:
    """Return the Parser :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-parse",
        description="congbobanan Parser: PDFs -> <case_id>.md + <case_id>.meta.json.",
        stages=[
            FilePartitioningStage(
                file_paths=str(layout.pdf_dir),
                file_extensions=[".pdf"],
                files_per_partition=int(
                    cfg.get("stage_overrides", {}).get(
                        "parse_files_per_partition", 32
                    )
                ),
                limit=int(cfg.limit) if cfg.get("limit") else None,
            ),
            DocumentIterateExtractStage(
                iterator=CongbobananDocumentIterator(),
                extractor=CongbobananDocumentExtractor(cfg),
                add_filename_column=False,
            ),
            PdfParseStage(cfg=cfg),
            MarkdownPerDocWriter(
                path=str(layout.md_dir),
                doc_name_field="doc_name",
                markdown_field="markdown",
            ),
        ],
        config={"host": str(cfg.host), "md_dir": str(layout.md_dir)},
    )


__all__ = ["build_parse_pipeline"]
