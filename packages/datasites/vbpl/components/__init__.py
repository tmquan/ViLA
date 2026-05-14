"""vbpl crawler primitives.

Five focused components, run as four ``--pipeline`` stages
(``harvest`` -> ``detail`` -> ``parse`` -> ``extract``):

    parser.py     -- pure-function sitemap XML + API-JSON -> dataclass
    harvester.py  -- /sitemap.xml walker (PoliteSession) -> sitemap.jsonl
    detail.py     -- per-ItemID Playwright fetcher       -> docs.jsonl
    parse.py      -- docs.jsonl + on-disk artefacts      -> md/<scope>/*.md
    extract.py    -- markdown + meta sidecars            -> jsonl/extract.jsonl

The driver lives in :mod:`packages.datasites.vbpl.scraper` and is
exposed via ``python -m packages.datasites.vbpl``.
"""

from packages.datasites.vbpl.components.detail import (
    DEFAULT_API_URL_SUBSTR,
    DEFAULT_WARMUP_URL,
    VbplDetailDownloader,
)
from packages.datasites.vbpl.components.extract import VbplDocumentExtractor
from packages.datasites.vbpl.components.harvester import (
    DEFAULT_SITEMAP_URL,
    VbplSitemapHarvester,
)
from packages.datasites.vbpl.components.parse import VbplDocumentParser
from packages.datasites.vbpl.components.parser import (
    DetailRecord,
    FilePath,
    SitemapEntry,
    detail_record_from_api_json,
    item_id_from_detail_url,
    parse_sitemap_index,
    parse_sitemap_urlset,
    scope_from_shard_url,
)

__all__ = [
    "DEFAULT_API_URL_SUBSTR",
    "DEFAULT_SITEMAP_URL",
    "DEFAULT_WARMUP_URL",
    "DetailRecord",
    "FilePath",
    "SitemapEntry",
    "VbplDetailDownloader",
    "VbplDocumentExtractor",
    "VbplDocumentParser",
    "VbplSitemapHarvester",
    "detail_record_from_api_json",
    "item_id_from_detail_url",
    "parse_sitemap_index",
    "parse_sitemap_urlset",
    "scope_from_shard_url",
]
