"""Downloader pipeline: URLs -> PDFs on disk.

Stage chain::

    URLGenerationStage(LuutruURLGenerator)
    -> DocumentDownloadStage(LuutruDocumentDownloader)

Reads: ``cfg.scraper.listing_url`` (vanban.aspx GET pagination).
Writes: ``data/<host>/pdf/<doc_name>.{pdf,docx,doc}`` +
        sibling ``<doc_name>.html`` / ``<doc_name>.url`` sidecars.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage

from packages.datasites.luutru._shared import build_layout
from packages.datasites.luutru.components import (
    LuutruDocumentDownloader,
    LuutruURLGenerator,
)


def build_download_pipeline(cfg: Any) -> Pipeline:
    """Return the Downloader :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-download",
        description="luutru Downloader: URLs -> PDFs on disk.",
        stages=[
            URLGenerationStage(
                url_generator=LuutruURLGenerator(cfg),
                limit=int(cfg.limit) if cfg.get("limit") else None,
            ),
            DocumentDownloadStage(
                downloader=LuutruDocumentDownloader(
                    cfg=cfg,
                    download_dir=str(layout.pdf_dir),
                ),
            ),
        ],
        config={"host": str(cfg.host), "pdf_dir": str(layout.pdf_dir)},
    )


__all__ = ["build_download_pipeline"]
