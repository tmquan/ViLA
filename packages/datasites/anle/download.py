"""Downloader pipeline: URLs -> PDFs on disk.

Stage chain::

    URLGenerationStage(AnleURLGenerator)
    -> DocumentDownloadStage(AnleDocumentDownloader)

Reads: ``cfg.scraper.listing_url`` (Oracle ADF).
Writes: ``data/<host>/pdf/<doc_name>.{pdf,docx,doc}`` +
        sibling ``<doc_name>.html`` / ``<doc_name>.url`` sidecars.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage

from packages.datasites.anle._shared import build_layout
from packages.datasites.anle.components import (
    AnleDocumentDownloader,
    AnleURLGenerator,
)


def build_download_pipeline(cfg: Any) -> Pipeline:
    """Return the Downloader :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-download",
        description="anle Downloader: URLs -> PDFs on disk.",
        stages=[
            URLGenerationStage(
                url_generator=AnleURLGenerator(cfg),
                limit=int(cfg.limit) if cfg.get("limit") else None,
            ),
            DocumentDownloadStage(
                downloader=AnleDocumentDownloader(
                    cfg=cfg,
                    download_dir=str(layout.pdf_dir),
                ),
            ),
        ],
        config={"host": str(cfg.host), "pdf_dir": str(layout.pdf_dir)},
    )


__all__ = ["build_download_pipeline"]
