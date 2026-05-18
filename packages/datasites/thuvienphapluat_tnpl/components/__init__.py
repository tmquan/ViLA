"""thuvienphapluat_tnpl crawler primitives.

The /tnpl/ portal exposes only an HTML interface (no JSON / OData /
SOAP API for the terminology data; no sitemap shard indexes /tnpl/
URLs -- see the package README's "Why no JSON API?" preface for the
probe results). We do not subclass Curator's PDF-oriented download
bases; instead this package ships four focused components:

    parser.py      -- pure-function HTML-fragment -> structured record
    harvester.py   -- homepage walker (taxonomy + bootstrap ids + probe
                      range) -> taxonomy.json + listings.jsonl
    downloader.py  -- per-id detail fetcher -> terms.jsonl
    translator.py  -- per-row NIM Nemotron 3 Super 120B-A12B translator
                      -> terms_translated.jsonl

The driver lives in :mod:`packages.datasites.thuvienphapluat_tnpl.scraper`
and is exposed via ``python -m packages.datasites.thuvienphapluat_tnpl``.
"""

from packages.datasites.thuvienphapluat_tnpl.components.downloader import (
    TnplDetailDownloader,
)
from packages.datasites.thuvienphapluat_tnpl.components.harvester import (
    DEFAULT_DETAIL_URL_TEMPLATE,
    DEFAULT_INDEX_URL,
    HarvestState,
    TnplHarvester,
)
from packages.datasites.thuvienphapluat_tnpl.components.parser import (
    DetailRecord,
    parse_detail_fragment,
    parse_homepage_ids,
    parse_taxonomy,
    parse_total_count,
)
from packages.datasites.thuvienphapluat_tnpl.components.translator import (
    DEFAULT_ENDPOINT_URL,
    DEFAULT_MODEL_ID,
    LLMClient,
    TnplTranslator,
    TranslationCache,
    TranslatorStats,
)

__all__ = [
    "DEFAULT_DETAIL_URL_TEMPLATE",
    "DEFAULT_ENDPOINT_URL",
    "DEFAULT_INDEX_URL",
    "DEFAULT_MODEL_ID",
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
]
