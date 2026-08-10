"""Downloader pipeline: integer case IDs -> PDFs on disk.

Stage chain::

    URLGenerationStage(CongbobananURLGenerator)
    -> DocumentDownloadStage(CongbobananDocumentDownloader)

Reads: ``cfg.scraper.start_id`` .. ``cfg.scraper.end_id`` (no network
for URL enumeration; the downloader pays the HTTP cost per ID and
filters out ghost pages).
Writes: ``data/<host>/pdf/<case_id>.pdf`` + sibling ``<case_id>.html``
/ ``<case_id>.url`` sidecars.
"""

from __future__ import annotations

from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage

from packages.datasites.congbobanan._shared import build_layout
from packages.datasites.congbobanan.components import (
    CongbobananDocumentDownloader,
    CongbobananURLGenerator,
)


def build_download_pipeline(cfg: Any) -> Pipeline:
    """Return the Downloader :class:`Pipeline`."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-download",
        description="congbobanan Downloader: integer case IDs -> PDFs on disk.",
        stages=[
            URLGenerationStage(
                url_generator=CongbobananURLGenerator(cfg),
                limit=int(cfg.limit) if cfg.get("limit") else None,
            ),
            DocumentDownloadStage(
                downloader=CongbobananDocumentDownloader(
                    cfg=cfg,
                    download_dir=str(layout.pdf_dir),
                ),
            ),
        ],
        config={"host": str(cfg.host), "pdf_dir": str(layout.pdf_dir)},
    )


__all__ = ["build_download_pipeline"]
