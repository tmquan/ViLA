"""Parser pipeline: PDF -> Markdown on disk.

Stage chain::

    FilePartitioningStage(pdf_dir, ext=[.pdf,.docx,.doc])
    -> DocumentIterateExtractStage(LuutruDocumentIterator, LuutruDocumentExtractor)
    -> PdfParseStage
    -> MarkdownPerDocWriter

Reads: ``data/<host>/pdf/*.{pdf,docx,doc}`` (+ sibling HTML / URL sidecars).
Writes: ``data/<host>/md/<doc_name>.md`` + ``<doc_name>.meta.json``.

The markdown body goes to the ``.md`` file; every other non-bytes
column on the row (doc_name, source, detail_url, pdf_url, doc_number,
issue_date, issuing_authority, summary, legal_type, ...) is
JSON-serialised into the sibling ``.meta.json`` so the Extractor
pipeline can rehydrate the full row when it reads the markdown back.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.download.base.iterator import (
    DocumentIterateExtractStage,
)

from packages.datasites.luutru._shared import build_layout
from packages.datasites.luutru.components import (
    LuutruDocumentExtractor,
    LuutruDocumentIterator,
)
from packages.parser.stage import PdfParseStage
from packages.pipeline.io import MarkdownPerDocWriter


def build_parse_pipeline(cfg: Any) -> Pipeline:
    """Return the Parser :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-parse",
        description="luutru Parser: PDFs -> <doc_name>.md + <doc_name>.meta.json.",
        stages=[
            FilePartitioningStage(
                file_paths=str(layout.pdf_dir),
                file_extensions=[".pdf", ".docx", ".doc"],
                files_per_partition=int(
                    cfg.get("stage_overrides", {}).get(
                        "parse_files_per_partition", 8
                    )
                ),
                limit=int(cfg.limit) if cfg.get("limit") else None,
            ),
            DocumentIterateExtractStage(
                iterator=LuutruDocumentIterator(),
                extractor=LuutruDocumentExtractor(cfg),
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
