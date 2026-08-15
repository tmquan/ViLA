"""phapdien.moj.gov.vn datasite -- crawler for Bộ Pháp Điển.

Source: https://phapdien.moj.gov.vn/ (the official codification of
Vietnamese law published by the Ministry of Justice; ~64K codified
articles across 42 topics × 202 đề-mục).

Why this crawler is shaped differently from anle / congbobanan:

* The portal exposes **no JSON / OData / SOAP API** for the codified
  content -- the only reachable surfaces are an ASP.NET WebForms
  jstree response (the topic / đề-mục tree) and a per-document AJAX
  endpoint that returns the codified HTML body. The crawler walks
  both surfaces in two stages (``tree`` + ``detail``).
* The output is article-shaped (one row per ``Điều``), not
  document-shaped: the scraper writes JSONL directly (no PDF parse /
  extract stage). Embeddings + 2-D coords are added by the in-process
  ``_embed_inproc`` / ``_reduce_inproc`` drivers on the GB10.

Top-level surface:

    scraper.py    -- tree walker + detail fetcher + run_pipeline dispatch
    analyze.py    -- post-crawl analytics (corpus stats + ontology)
    viz.py        -- matplotlib + mermaid visualisations
    ontology.py   -- bilingual VI/EN topic + đề-mục lexicon
    hf_export.py  -- materialise data + dataset card under hf/
    push_to_hf.py -- wrapper around packages.common.hf.run_push_cli
    __main__.py   -- CLI mirroring the other datasites

Run via::

    python -m packages.datasites.phapdien --pipeline all
    python -m packages.datasites.phapdien.analyze
    python -m packages.datasites.phapdien.hf_export
    python -m packages.datasites.phapdien.push_to_hf
"""

from packages.datasites.phapdien.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_pipeline,
)

__all__ = [
    "ALL_PIPELINES_ORDER",
    "PIPELINES",
    "run_pipeline",
]
