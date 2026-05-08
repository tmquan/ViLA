"""Top-level pipeline dispatch for the pbgdpl crawler.

Two stages, run by name:

    harvest   -- walk ?page=1..N + ?lv=ID and write
                 listings.jsonl + taxonomy.json + html caches.
    detail    -- read listings.jsonl, fetch ?ItemID=X for every row,
                 parse + write qa.jsonl + manifest.json.
    all       -- run harvest then detail (default).

The two stages are decoupled so a partial crawl can be resumed
cheaply: re-running ``harvest`` re-uses cached listing fragments;
re-running ``detail`` re-uses cached item fragments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from packages.datasites.pbgdpl._shared import build_layout
from packages.datasites.pbgdpl.components import (
    PbgdplDetailDownloader,
    PbgdplHarvester,
)

logger = logging.getLogger(__name__)


PIPELINE_NAMES = ("harvest", "detail")
ALL_PIPELINES_ORDER = list(PIPELINE_NAMES)


def run_harvest(cfg: Any) -> Path:
    """Walk listings + taxonomy, write listings.jsonl. Returns its path."""
    layout = build_layout(cfg)
    harv = PbgdplHarvester(cfg, layout)
    state = harv.run()
    listings_path, _ = harv.write_outputs(state)
    logger.info(
        "harvest complete: items=%d global_pages=%d lv_pages=%d featured=%d",
        len(state.items), state.page_count, state.lv_pages_fetched,
        len(state.featured_ids),
    )
    return listings_path


def run_detail(cfg: Any) -> Path:
    """Fetch + parse every detail page, write qa.jsonl. Returns its path."""
    layout = build_layout(cfg)
    return PbgdplDetailDownloader(cfg, layout).run()


PIPELINES: dict[str, Callable[[Any], Path]] = {
    "harvest": run_harvest,
    "detail": run_detail,
}


def run_pipeline(cfg: Any, name: str) -> Path:
    """Dispatch to the named pipeline. ``name='all'`` is handled in __main__."""
    if name not in PIPELINES:
        raise ValueError(
            f"unknown pipeline {name!r}; choices: {list(PIPELINES) + ['all']}"
        )
    return PIPELINES[name](cfg)


__all__ = [
    "ALL_PIPELINES_ORDER",
    "PIPELINES",
    "PIPELINE_NAMES",
    "run_detail",
    "run_harvest",
    "run_pipeline",
]
