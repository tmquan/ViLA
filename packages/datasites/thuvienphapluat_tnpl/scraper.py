"""Top-level pipeline dispatch for the thuvienphapluat_tnpl crawler.

Three stages, run by name:

    harvest    -- fetch /tnpl/home, derive LinhVuc taxonomy + probe
                  range, write taxonomy.json + listings.jsonl.
    detail     -- read listings.jsonl, fetch /tnpl/{id}/x?tab=0 for
                  every probe id, parse + write terms.jsonl + manifest.json.
    translate  -- read terms.jsonl, run the NIM Nemotron 3 Super
                  120B-A12B translator over every Vietnamese-language
                  field, write terms_translated.jsonl + translation_manifest.json.
    all        -- run harvest then detail then translate (default).

The stages are decoupled so a partial run is cheap to resume:
re-running ``harvest`` re-uses the cached homepage; ``detail`` re-uses
cached item HTML; ``translate`` re-uses the per-row LLM cache under
``translations/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from packages.datasites.thuvienphapluat_tnpl._shared import build_layout
from packages.datasites.thuvienphapluat_tnpl.components import (
    TnplDetailDownloader,
    TnplHarvester,
    TnplTranslator,
)

logger = logging.getLogger(__name__)


PIPELINE_NAMES = ("harvest", "detail", "translate")
ALL_PIPELINES_ORDER = list(PIPELINE_NAMES)


def run_harvest(cfg: Any) -> Path:
    """Walk the homepage + write listings.jsonl. Returns its path."""
    layout = build_layout(cfg)
    harv = TnplHarvester(cfg, layout)
    state = harv.run()
    listings_path, _ = harv.write_outputs(state)
    logger.info(
        "harvest complete: taxonomy=%d, total_count=%s, bootstrap_ids=%d, "
        "probe=[%d, %d]",
        len(state.taxonomy), state.total_count, len(state.homepage_ids),
        state.probe_start, state.probe_end,
    )
    return listings_path


def run_detail(cfg: Any) -> Path:
    """Fetch + parse every detail page, write terms.jsonl. Returns its path."""
    layout = build_layout(cfg)
    return TnplDetailDownloader(cfg, layout).run()


def run_translate(cfg: Any) -> Path:
    """Translate every row, write terms_translated.jsonl. Returns its path."""
    layout = build_layout(cfg)
    return TnplTranslator(cfg, layout).run()


PIPELINES: dict[str, Callable[[Any], Path]] = {
    "harvest":   run_harvest,
    "detail":    run_detail,
    "translate": run_translate,
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
    "run_translate",
]
