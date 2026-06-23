"""Registry + dispatch for the dichvucong curation pipelines.

Stage chain (Curator + Ray; the deck's URLGenerator → Downloader →
Iterator → Extractor pattern):

    crawl   : URLGenerationStage(DichvucongURLGenerator)
              -> DocumentDownloadStage(DichvucongDocumentDownloader)   # pages/*.json
    extract : FilePartitioningStage(pages/, ext=[.json])
              -> DocumentIterateExtractStage(Iterator, Extractor)       # one row / procedure
              -> JsonlPerDocWriter(jsonl/)

``embed`` / ``reduce`` reuse the shared factories in
:mod:`packages.pipeline.factories` unchanged (the curated rows carry a
``procedure_name`` text field to embed). They are registered here so a
later step can be turned on without touching the CLI wiring.

The freshness / incremental mechanism (new vs. amended vs. withdrawn
procedures) is documented in ``wiki/DICHVUCONG.md`` §5 and implemented
by :func:`run_reconcile`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.file_partitioning import FilePartitioningStage
from nemo_curator.stages.text.download.base.download import DocumentDownloadStage
from nemo_curator.stages.text.download.base.iterator import (
    DocumentIterateExtractStage,
)
from nemo_curator.stages.text.download.base.url_generation import URLGenerationStage

from packages.datasites.dichvucong._shared import (
    EXTRACTOR_JSONL_FIELDS,
    build_layout,
    pages_dir,
)
from packages.datasites.dichvucong.components import (
    DichvucongDocumentDownloader,
    DichvucongDocumentExtractor,
    DichvucongDocumentIterator,
    DichvucongURLGenerator,
)
from packages.pipeline.io import JsonlPerDocWriter


def build_crawl_pipeline(cfg: Any) -> Pipeline:
    """Enumerate + download every search page as ``pages/*.json``."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-crawl",
        description="dichvucong Crawler: rest.jsp search pages -> pages/*.json.",
        stages=[
            URLGenerationStage(
                url_generator=DichvucongURLGenerator(cfg),
                limit=int(cfg.limit) if cfg.get("limit") else None,
            ),
            DocumentDownloadStage(
                downloader=DichvucongDocumentDownloader(
                    cfg=cfg,
                    download_dir=str(pages_dir(layout)),
                ),
            ),
        ],
        config={"host": str(cfg.host), "pages_dir": str(pages_dir(layout))},
    )


def build_extract_pipeline(cfg: Any) -> Pipeline:
    """Flatten cached page JSON into one curated row per procedure."""
    layout = build_layout(cfg)
    return Pipeline(
        name=f"{cfg.host}-extract",
        description="dichvucong Extractor: pages/*.json -> jsonl/<code>.jsonl.",
        stages=[
            FilePartitioningStage(
                file_paths=str(pages_dir(layout)),
                file_extensions=[".json"],
                files_per_partition=int(
                    cfg.get("stage_overrides", {}).get(
                        "extract_files_per_partition", 8
                    )
                ),
                limit=int(cfg.limit) if cfg.get("limit") else None,
            ),
            DocumentIterateExtractStage(
                iterator=DichvucongDocumentIterator(),
                extractor=DichvucongDocumentExtractor(cfg),
                add_filename_column=False,
            ),
            JsonlPerDocWriter(
                path=str(layout.jsonl_dir),
                doc_name_field="doc_name",
                fields=list(EXTRACTOR_JSONL_FIELDS),
            ),
        ],
        config={"host": str(cfg.host), "jsonl_dir": str(layout.jsonl_dir)},
    )


PIPELINES: dict[str, Callable[[Any], Pipeline]] = {
    "crawl": build_crawl_pipeline,
    "extract": build_extract_pipeline,
}

ALL_PIPELINES_ORDER: list[str] = ["crawl", "extract"]


def build_pipeline(cfg: Any, name: str) -> Pipeline:
    if name not in PIPELINES:
        raise ValueError(
            f"unknown pipeline: {name!r}; expected one of {sorted(PIPELINES)}"
        )
    return PIPELINES[name](cfg)


__all__ = [
    "ALL_PIPELINES_ORDER",
    "EXTRACTOR_JSONL_FIELDS",
    "PIPELINES",
    "build_crawl_pipeline",
    "build_extract_pipeline",
    "build_pipeline",
]
