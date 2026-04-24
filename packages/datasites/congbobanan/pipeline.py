"""Registry + dispatch for the five congbobanan curation pipelines.

One factory per pipeline lives in its own top-level file:

    download.py  -> build_download_pipeline   case IDs       -> PDFs
    parse.py     -> build_parse_pipeline      PDFs           -> markdown
    extract.py   -> build_extract_pipeline    markdown       -> JSONL
    embed.py     -> build_embed_pipeline      JSONL          -> embeddings parquet
    reduce.py    -> build_reduce_pipeline     embeddings     -> reduced parquet

Field constants are re-exported from :mod:`packages.datasites.congbobanan._shared`
for tests / notebooks that want a single import point.
"""

from __future__ import annotations

from typing import Any, Callable

from nemo_curator.pipeline import Pipeline

from packages.datasites.congbobanan._shared import (
    EMBEDDER_JSONL_READ_FIELDS,
    EMBEDDER_PARQUET_FIELDS,
    EXTRACTOR_JSONL_FIELDS,
    REDUCER_PARQUET_FIELDS,
)
from packages.datasites.congbobanan.download import build_download_pipeline
from packages.datasites.congbobanan.embed import build_embed_pipeline
from packages.datasites.congbobanan.extract import build_extract_pipeline
from packages.datasites.congbobanan.parse import build_parse_pipeline
from packages.datasites.congbobanan.reduce import build_reduce_pipeline


#: Pipeline name -> factory. Keys double as ``--pipeline`` CLI choices.
PIPELINES: dict[str, Callable[[Any], Pipeline]] = {
    "download": build_download_pipeline,
    "parse": build_parse_pipeline,
    "extract": build_extract_pipeline,
    "embed": build_embed_pipeline,
    "reduce": build_reduce_pipeline,
}


#: Default execution order when ``--pipeline all`` is selected.
ALL_PIPELINES_ORDER: list[str] = [
    "download",
    "parse",
    "extract",
    "embed",
    "reduce",
]


def build_pipeline(cfg: Any, name: str) -> Pipeline:
    """Return the named pipeline. ``name`` is one of :data:`PIPELINES`."""
    if name not in PIPELINES:
        raise ValueError(
            f"unknown pipeline: {name!r}; "
            f"expected one of {sorted(PIPELINES)}"
        )
    return PIPELINES[name](cfg)


__all__ = [
    "ALL_PIPELINES_ORDER",
    "EMBEDDER_JSONL_READ_FIELDS",
    "EMBEDDER_PARQUET_FIELDS",
    "EXTRACTOR_JSONL_FIELDS",
    "PIPELINES",
    "REDUCER_PARQUET_FIELDS",
    "build_download_pipeline",
    "build_embed_pipeline",
    "build_extract_pipeline",
    "build_parse_pipeline",
    "build_pipeline",
    "build_reduce_pipeline",
]
