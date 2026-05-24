"""thuvienphapluat_tnpl datasite — Vietnamese legal-terminology dictionary.

Source: https://thuvienphapluat.vn/tnpl/ (the "Thuật ngữ pháp lý" /
"Legal Terminology" surface of THƯ VIỆN PHÁP LUẬT; ~16 247 Vietnamese-
language legal-terminology entries across 47 legal-domain (LinhVuc)
categories as of 2026-05). The deliverable on Hugging Face is
bilingual: every Vietnamese-language column is paired with a clean
English column produced by the NIM Nemotron 3 Super 120B-A12B
translator.

Why this crawler is shaped differently from anle / congbobanan:

* The portal exposes **no JSON / OData / SOAP API** for the term
  content. /tnpl/home server-renders only the most-recent ~20 ids
  plus the LinhVuc dropdown (47 closed values). The
  ``/tnpl/search?keyword=...&ddlField=...`` endpoint is a Solr fuzzy
  matcher that only returns ~4 near-matches per query. We probed
  ``sitemap.xml``, ``resitemap1..575.xml``, ``sitemap_tnpl.xml`` --
  none of them index /tnpl/ URLs.
* So the only viable harvest strategy is **brute-force sequential ID
  enumeration** over ``[1, max_id + id_buffer]`` where ``max_id`` is
  derived from the homepage's largest visible id. The parser tags
  missing/retracted ids as ``fetch_status="not_found"``.
* The detail surface is a single HTML page (no PDF / OCR), so the
  crawler is an HTML-fragment parser, not a PDF pipeline. There is
  no curator parse / extract stage in the critical path; the
  **embed + reduce** stage is in-process (``_embed_reduce_inproc.py``)
  and bilingual -- it embeds VI **and** EN definitions with the same
  multilingual encoder so paired VI<->EN cosine measures translation
  fidelity, and writes ``parquet/terms_reduced.parquet`` consumed by
  the analytics + viz stages.
* The dataset is bilingual, so a third **translate** stage drives the
  NIM Nemotron 3 Super 120B-A12B endpoint to fill the English
  columns. Translation is per-row, resumable from a tiny on-disk
  cache (``translations/<term_id>.json``).

Top-level surface:

    components/parser.py       -- HTML fragment -> dataclass record
    components/harvester.py    -- homepage walker (taxonomy + probe range)
    components/downloader.py   -- detail fetcher (?ItemID-style by id)
    components/translator.py   -- NIM chat-completion translator
    scraper.py                 -- run_harvest + run_detail + run_translate dispatch
    _embed_reduce_inproc.py    -- bilingual embed + PCA/t-SNE/UMAP +
                                  HDBSCAN -> parquet/terms_reduced.parquet
    analyze.py                 -- post-crawl analytics (-> analytics.json,
                                  picks up embedding stats when parquet exists)
    viz.py                     -- matplotlib PNG figures (ontology +
                                  bilingual embedding scatters)
    __main__.py                -- CLI mirroring the other datasites

Run via::

    python -m packages.datasites.thuvienphapluat_tnpl --pipeline all
    python -m packages.datasites.thuvienphapluat_tnpl._embed_reduce_inproc
    python -m packages.datasites.thuvienphapluat_tnpl.analyze
    python -m packages.datasites.thuvienphapluat_tnpl.viz
"""

from packages.datasites.thuvienphapluat_tnpl.components import (
    DEFAULT_DETAIL_URL_TEMPLATE,
    DEFAULT_ENDPOINT_URL,
    DEFAULT_INDEX_URL,
    DEFAULT_MODEL_ID,
    DetailRecord,
    HarvestState,
    LLMClient,
    TnplDetailDownloader,
    TnplHarvester,
    TnplTranslator,
    TranslationCache,
    TranslatorStats,
    parse_detail_fragment,
    parse_homepage_ids,
    parse_taxonomy,
    parse_total_count,
)
from packages.datasites.thuvienphapluat_tnpl.scraper import (
    ALL_PIPELINES_ORDER,
    PIPELINES,
    run_detail,
    run_harvest,
    run_pipeline,
    run_translate,
)

__all__ = [
    "ALL_PIPELINES_ORDER",
    "DEFAULT_DETAIL_URL_TEMPLATE",
    "DEFAULT_ENDPOINT_URL",
    "DEFAULT_INDEX_URL",
    "DEFAULT_MODEL_ID",
    "PIPELINES",
    "DetailRecord",
    "HarvestState",
    "LLMClient",
    "TnplDetailDownloader",
    "TnplHarvester",
    "TnplTranslator",
    "TranslationCache",
    "TranslatorStats",
    "parse_detail_fragment",
    "parse_homepage_ids",
    "parse_taxonomy",
    "parse_total_count",
    "run_detail",
    "run_harvest",
    "run_pipeline",
    "run_translate",
]
